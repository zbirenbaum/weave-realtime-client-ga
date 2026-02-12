import weave
import asyncio
import pyaudio
import numpy as np
from weave.integrations import patch_openai_realtime
from agents.realtime import RealtimeAgent, RealtimeRunner


weave.init("ga-realtime-agents-example")
patch_openai_realtime()


# Required Audio Specs
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
CHUNK = 1024

async def main():
    p = pyaudio.PyAudio()
    # Setup Mic and Speaker
    mic = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=0)
    speaker = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK, output_device_index=1)

    agent = RealtimeAgent(
        name="Assistant",
        instructions="You are a tool using AI. Use tools to accomplish a task whenever possible"
    )
    runner = RealtimeRunner(agent, config={
        "model_settings": {
            "modalities": ["text", "audio"],
            "output_modalities": ["text", "audio"],
        }
    })

    print("--- Session Active (Speak into mic) ---")

    # The 'run' method returns the RealtimeSession object you provided
    async with await runner.run() as session:
        async def send_mic_audio():
            """Reads from hardware and uses the validated 'send_audio' method."""
            try:
                while True:
                    raw_data = mic.read(CHUNK, exception_on_overflow=False)

                    # Visual Volume Meter
                    audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float64)
                    rms = np.sqrt(np.mean(audio_data**2))
                    meter = int(min(rms / 50, 50))
                    print(f"Mic Level: {'█' * meter}{' ' * (50-meter)} |", end="\r")

                    # VALIDATED METHOD: session.send_audio
                    await session.send_audio(raw_data)

                    await asyncio.sleep(0)
            except Exception:
                pass

        async def handle_events():
            """Iterates over the session as an AsyncIterator."""
            async for event in session:
                # Based on your imports: RealtimeAudio contains the audio data
                if event.type == "audio":
                    # event.audio is the RealtimeModelEvent containing the bytes
                    speaker.write(event.audio.data)
                # Check for transcripts to print
                elif event.type == "transcript_delta":
                    print(event.delta, end="", flush=True)

        # Execute
        mic_task = asyncio.create_task(send_mic_audio())
        try:
            await handle_events()
        finally:
            mic_task.cancel()

    # Cleanup
    mic.close()
    speaker.close()
    p.terminate()

if __name__ == "__main__":
    asyncio.run(main())
