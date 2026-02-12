import asyncio
import base64
import json
import os
from typing import Any, Callable

import numpy as np
import pyaudio
import websockets
import json

import weave
weave.init('wandb/ga-example')
from weave.integrations import patch_openai_realtime
patch_openai_realtime()

# from weave.integrations import patch_openai_realtime
# patch_openai_realtime()

from tool_definitions import (
    calculate,
    get_weather,
    run_python_code,
    write_file,
)

# Audio settings matching OpenAI Realtime API requirements
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
CHUNK = 1024

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime"

# Map tool name -> callable for function call dispatch
TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "get_weather": get_weather,
    "calculate": calculate,
    "run_python_code": run_python_code,
    "write_file": write_file,
}

# Raw tool definitions for the Realtime API session config
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name to get weather for.",
                }
            },
            "required": ["city"],
        },
    },
    {
        "type": "function",
        "name": "calculate",
        "description": "Evaluate a math expression and return the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression to evaluate, e.g. '2 + 2'.",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "type": "function",
        "name": "run_python_code",
        "description": "Write and execute a Python script, returning its stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python source code to execute.",
                }
            },
            "required": ["code"],
        },
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Write content to a file on disk.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to write the file to.",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write into the file.",
                },
            },
            "required": ["file_path", "content"],
        },
    },
]


async def send_event(ws, event: dict) -> None:
    await ws.send(json.dumps(event))


async def configure_session(ws) -> None:
    event = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": "gpt-realtime",
            "instructions": (
                "You are a helpful AI assistant with access to tools. "
                "Use tools to accomplish tasks whenever possible. "
                "Speak clearly and briefly."
            ),
            "tools": TOOL_DEFINITIONS,
            "tool_choice": "auto",
            "audio": {
                "input": {
                  "format": {
                    "type": "audio/pcm",
                    "rate": 24000
                  },
                  "transcription": {
                    "model": "gpt-4o-transcribe"
                  },
                  "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                  }
                }
            },
        },
    }
    await send_event(ws, event)
    print("set session")


async def handle_function_call(ws, call_id: str, name: str, arguments: str) -> None:
    if not name:
        raise Exception("Did not get a function name")

    print(f"\n[Function Call] {name}({arguments})")
    tool_fn = TOOL_REGISTRY.get(name)
    print(TOOL_REGISTRY)
    if tool_fn is None:
        result = json.dumps({"error": f"Unknown function: {name}"})
    else:
        try:
            args = json.loads(arguments)
            result = tool_fn(**args)
            if asyncio.iscoroutine(result):
                result = await result
        except Exception as e:
            result = json.dumps({"error": str(e)})

    print(f"[Function Result] {result}")

    # Send the function call output back to the model
    await send_event(ws, {
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": result if isinstance(result, str) else json.dumps(result),
        },
    })

    # Trigger a new response so the model incorporates the function result
    await send_event(ws, {"type": "response.create"})


async def send_mic_audio(ws, mic) -> None:
    try:
        while True:
            raw_data = mic.read(CHUNK, exception_on_overflow=False)

            # Visual volume meter
            audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float64)
            rms = np.sqrt(np.mean(audio_data**2))
            meter = int(min(rms / 50, 50))
            print(f"Mic Level: {'█' * meter}{' ' * (50 - meter)} |", end="\r")

            # Base64-encode and send audio chunk
            b64_audio = base64.b64encode(raw_data).decode("utf-8")
            await send_event(ws, {
                "type": "input_audio_buffer.append",
                "audio": b64_audio,
            })

            await asyncio.sleep(0)
    except asyncio.CancelledError:
        pass


async def receive_events(ws, speaker) -> None:
    # Accumulate function call arguments across delta events
    pending_calls: dict[str, dict] = {}

    async for raw_message in ws:

# Use "a" for append mode
        with open("data.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(raw_message) + "\n")

        event = json.loads(raw_message)
        event_type = event.get("type", "")
        if event_type == "session.created":
            event = {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "output_modalities": ["audio"],
                    "instructions": (
                        "You are a helpful AI assistant with access to tools. "
                        "Use tools to accomplish tasks whenever possible. "
                        "Speak clearly and briefly."
                    ),
                    "tools": TOOL_DEFINITIONS,
                    "tool_choice": "auto",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "turn_detection": {"type": "server_vad"},
                        },
                        "output": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                        },
                    },
                },
            }
            await send_event(ws, event)
            print(raw_message)

        elif event_type == "session.updated":
            print(raw_message)

        elif event_type == "error":
            pass

        elif event_type == "input_audio_buffer.speech_started":
            pass

        elif event_type == "input_audio_buffer.speech_stopped":
            pass

        elif event_type == "input_audio_buffer.committed":
            pass

        elif event_type == "response.created":
            pass

        # Text output deltas
        elif event_type == "response.output_text.delta":
            pass

        elif event_type == "response.output_text.done":
            pass

        # Audio output deltas - decode and play
        elif event_type == "response.output_audio.delta":
            audio_bytes = base64.b64decode(event.get("delta", ""))
            speaker.write(audio_bytes)

        # Audio transcript deltas
        elif event_type == "response.output_audio_transcript.delta":
            pass

        elif event_type == "response.output_audio_transcript.done":
            pass

        # Function call argument deltas - accumulate
        elif event_type == "response.function_call_arguments.delta":
            item_id = event.get("item_id", "")
            if item_id not in pending_calls:
                pending_calls[item_id] = {
                    "call_id": event.get("call_id", ""),
                    "name": event.get("name", ""),
                    "arguments": "",
                }
            pending_calls[item_id]["arguments"] += event.get("delta", "")

        elif event_type == "response.function_call_arguments.done":
            item_id = event.get("item_id", "")
            call_info = pending_calls.pop(item_id, None)
            print(event)
            if call_info is None:
                # Fallback: use data directly from the done event
                call_info = {
                    "call_id": event.get("call_id"),
                    "name": event.get("name"),
                    "arguments": event.get("arguments"),
                }
            try:
                await handle_function_call(
                    ws,
                    call_info["call_id"],
                    call_info["name"],
                    call_info["arguments"],
                )
            except Exception as e:
                print(f"Failed to call function for message {call_info}: error - {e}")


        elif event_type == "response.done":
            response = event.get("response", {})

            # Check for function calls in the response output
            for output_item in response.get("output", []):
                if output_item.get("type") == "function_call":
                    item_id = output_item.get("id", "")
                    # Only handle if not already handled via delta events
                    if item_id not in pending_calls:
                        await handle_function_call(
                            ws,
                            output_item.get("call_id", ""),
                            output_item.get("name", ""),
                            output_item.get("arguments", ""),
                        )

        elif event_type == "rate_limits.updated":
            pass  # Silently ignore rate limit updates

        else:
            # Log unhandled events for debugging
            print(f"\n[Event: {event_type}]")

async def main():
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    p = pyaudio.PyAudio()
    mic = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=0,
    )
    speaker = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        output=True,
        frames_per_buffer=CHUNK,
        output_device_index=1,
    )

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    print("Connecting to OpenAI Realtime API...")

    async with websockets.connect(
        REALTIME_URL,
        additional_headers=headers,
    ) as ws:
        print("Connected! Configuring session...")
        await configure_session(ws)

        print("--- Session Active (Speak into mic) ---")

        mic_task = asyncio.create_task(send_mic_audio(ws, mic))
        await configure_session(ws)
        await configure_session(ws)
        try:
            await receive_events(ws, speaker)
        finally:
            mic_task.cancel()
            try:
                await mic_task
            except asyncio.CancelledError:
                pass

    mic.close()
    speaker.close()
    p.terminate()
    print("\nSession ended.")


if __name__ == "__main__":
    asyncio.run(main())
