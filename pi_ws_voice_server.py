import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from vosk import Model, KaldiRecognizer, SetLogLevel
from gpiozero import LED

from command_parser import check_light_command

SetLogLevel(-1)

MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", "/home/pi/Vosk_Project/vosk-model-small-en-us-0.15")
SAMPLE_RATE = 16000.0

LIGHT_PIN = 22
YELLOW_PIN = 27
RED_PIN = 17

app = FastAPI()
model = Model(MODEL_PATH)
thread_pool = ThreadPoolExecutor(max_workers=2)

light_actuator = LED(LIGHT_PIN)
yellow_led = LED(YELLOW_PIN)
red_led = LED(RED_PIN)

def apply_hardware_action(command: str):
    if command == "ON":
        light_actuator.on()
        yellow_led.off()
        red_led.off()
    elif command == "OFF":
        light_actuator.off()
        yellow_led.off()
        red_led.off()
    else:  # UNKNOWN
        yellow_led.on()
        red_led.off()

def get_hardware_payload():
    return {
        "green": light_actuator.is_lit,
        "yellow": yellow_led.is_lit,
        "red": red_led.is_lit
    }

@app.websocket("/ws/stream")
async def websocket_audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()
    
    # Each streaming connection gets its own Kaldi state
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetWords(True)

    # Send initial state
    await websocket.send_json({"type": "state", "hardware": get_hardware_payload()})

    try:
        while True:
            message = await websocket.receive()
            
            # Binary frame = Raw 16-bit 16kHz PCM audio
            if "bytes" in message and message["bytes"]:
                chunk = message["bytes"]
                accepted = await loop.run_in_executor(
                    thread_pool, recognizer.AcceptWaveform, chunk
                )
                
                if accepted:
                    result = json.loads(recognizer.Result())
                    transcript = result.get("text", "").strip()
                    if transcript:
                        command = check_light_command(transcript)
                        apply_hardware_action(command)
                        await websocket.send_json({
                            "type": "final",
                            "transcript": transcript,
                            "command": command,
                            "hardware": get_hardware_payload()
                        })
                else:
                    partial = json.loads(recognizer.PartialResult())
                    partial_text = partial.get("partial", "").strip()
                    if partial_text:
                        await websocket.send_json({
                            "type": "partial",
                            "transcript": partial_text
                        })

            # Text frame = Manual override or EOF trigger
            elif "text" in message and message["text"]:
                data = json.loads(message["text"])
                if data.get("action") == "manual_toggle":
                    target_state = data.get("state")
                    if target_state == "ON":
                        light_actuator.on()
                    elif target_state == "OFF":
                        light_actuator.off()
                    await websocket.send_json({
                        "type": "state",
                        "hardware": get_hardware_payload()
                    })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        red_led.on()
        await websocket.close(code=1011, reason=str(e))