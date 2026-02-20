import argparse
import asyncio
import queue
import threading
import weave
import pyaudio
import numpy as np
from weave.integrations import patch_openai_realtime
from agents.realtime import RealtimeAgent, RealtimeRunner

DEFAULT_WEAVE_PROJECT = "ga-realtime-agents-example"

FORMAT = pyaudio.paInt16
RATE = 24000
CHUNK = 1024
MAX_INPUT_CHANNELS = 1
MAX_OUTPUT_CHANNELS = 1

INP_DEV_IDX = None
OUT_DEV_IDX = None

def parse_args():
    parser = argparse.ArgumentParser(description="Realtime agent with Weave logging")
    parser.add_argument(
        "--weave-project",
        default=DEFAULT_WEAVE_PROJECT,
        help=f"Weave project name (default: {DEFAULT_WEAVE_PROJECT})",
        dest="weave_project"
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="PyAudio input (mic) device index. Defaults to system default. Run mic_detect.py to list devices.",
        dest="input_device"
    )
    parser.add_argument(
        "--output-device",
        type=int,
        default=None,
        help="PyAudio output (speaker) device index. Defaults to system default. Run mic_detect.py to list devices.",
        dest="output_device"
    )
    return parser.parse_args()


def init_weave(project_name: str | None = None) -> None:
    name = project_name or DEFAULT_WEAVE_PROJECT
    weave.init(name)
    patch_openai_realtime()


def play_audio(output_stream: pyaudio.Stream, audio_output_queue: queue.Queue):
    """Runs in a separate thread because pyaudio's write() blocks until the
    sound card consumes the samples. Decoupling playback from the async event
    loop lets us flush the queue on interrupt without waiting for in-flight
    writes to finish."""
    while True:
        data = audio_output_queue.get()
        if data is None:
            break
        output_stream.write(data)


async def main(*, input_device_index: int | None = None, output_device_index: int | None = None):
    p = pyaudio.PyAudio()

    if input_device_index is None:
        input_device_index = int(p.get_default_input_device_info()['index'])
    if output_device_index is None:
        output_device_index = int(p.get_default_output_device_info()['index'])

    # Channel count must match the device's capabilities or pyaudio will error on open
    input_info = p.get_device_info_by_index(input_device_index)
    output_info = p.get_device_info_by_index(output_device_index)

    input_channels = min(int(input_info['maxInputChannels']), MAX_INPUT_CHANNELS)
    output_channels = min(int(output_info['maxOutputChannels']), MAX_OUTPUT_CHANNELS)

    mic = p.open(
        format=FORMAT,
        channels=input_channels,
        rate=RATE,
        input=True,
        output=False,
        frames_per_buffer=CHUNK,
        input_device_index=input_device_index,
        start=False,
    )
    speaker = p.open(
        format=FORMAT,
        channels=output_channels,
        rate=RATE,
        input=False,
        output=True,
        frames_per_buffer=CHUNK,
        output_device_index=output_device_index,
        start=False,
    )
    mic.start_stream()
    speaker.start_stream()

    # Audio goes through a queue so we can flush it when the user interrupts.
    # Writing directly to the speaker makes it impossible to cancel in-flight audio.
    audio_output_queue = queue.Queue()
    threading.Thread(
        target=play_audio, args=(speaker, audio_output_queue), daemon=True
    ).start()

    s_agent = RealtimeAgent(
        name="Speech Assistant",
        instructions="You are a tool using AI. Use tools to accomplish a task whenever possible"
    )

    s_runner = RealtimeRunner(s_agent, config={
        "model_settings": {
            "model_name": "gpt-realtime",
            "modalities": ["audio"],
            "output_modalities": ["audio"],
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "speed": 1.2,
            "turn_detection": {
                "type": "semantic_vad",
                "interrupt_response": True,
                "create_response": True,
            },
        }
    })
    print("--- Session Active (Speak into mic) ---")

    async with await s_runner.run() as session:
        async def send_mic_audio():
            try:
                while True:
                    raw_data = mic.read(CHUNK, exception_on_overflow=False)

                    audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float64)
                    rms = np.sqrt(np.mean(audio_data**2))
                    meter = int(min(rms / 50, 50))
                    print(f"Mic Level: {'█' * meter}{' ' * (50-meter)} |", end="\r")

                    await session.send_audio(raw_data)
                    await asyncio.sleep(0)
            except Exception:
                pass

        async def handle_events():
            async for event in session:
                if event.type == "audio":
                    audio_output_queue.put(event.audio.data)
                elif event.type == "audio_interrupted":
                    # User started speaking — discard buffered AI audio so it
                    # doesn't play over the user's voice
                    while not audio_output_queue.empty():
                        try:
                            audio_output_queue.get_nowait()
                        except queue.Empty:
                            break

        mic_task = asyncio.create_task(send_mic_audio())
        try:
            await handle_events()
        finally:
            mic_task.cancel()

    # Cleanup
    audio_output_queue.put(None)  # signal playback thread to exit
    mic.close()
    speaker.close()
    p.terminate()

if __name__ == "__main__":
    args = parse_args()
    init_weave(args.weave_project)
    asyncio.run(main(input_device_index=args.input_device, output_device_index=args.output_device))
