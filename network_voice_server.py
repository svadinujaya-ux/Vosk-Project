"""Raspberry Pi server for laptop microphone voice commands with Wake-Word & Grammar.

The server receives short WAV recordings over the local network, verifies the wake
word ('Hey Vosk' / 'Hello'), transcribes commands with a constrained Kaldi grammar,
and controls the GPIO light.
"""

import argparse
import json
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import Optional

from command_parser import check_light_command


class VoiceCommandServer:
    WAKE_PHRASES = (
        "hey vosk",
        "hey boss",
        "hey box",
        "hey mosque",
        "hey lost",
        "hello vosk",
        "hello boss",
        "hey light",
        "hello light",
        "hello",
    )

    GRAMMAR_VOCAB = [
        "hey vosk", "hey boss", "hey box", "hey mosque", "hey lost",
        "hello vosk", "hello boss", "hey light", "hello light", "hello",
        "turn on", "turn off", "switch on", "switch off",
        "turn on the light", "turn off the light",
        "turn on the lights", "turn off the lights",
        "turn the light on", "turn the light off",
        "turn the lights on", "turn the lights off",
        "light on", "light off", "lights on", "lights off",
        "do not", "dont", "not", "no", "never",
        "[unk]",
    ]

    def __init__(
        self,
        model_path: str,
        light_pin: int = 22,
        yellow_pin: int = 27,
        red_pin: int = 17,
        active_low: bool = False,
        max_audio_bytes: int = 2_000_000,
        listen_timeout: float = 6.0,
        use_grammar: bool = True,
    ) -> None:
        try:
            from gpiozero import LED
            from vosk import KaldiRecognizer, Model, SetLogLevel
        except ImportError as error:
            raise RuntimeError(
                "Install dependencies on the Pi with: pip3 install vosk gpiozero"
            ) from error

        if max_audio_bytes <= 0:
            raise ValueError("max_audio_bytes must be greater than zero")

        SetLogLevel(-1)
        self.max_audio_bytes = max_audio_bytes
        self.listen_timeout = listen_timeout
        self.use_grammar = use_grammar
        self._grammar_json = json.dumps(self.GRAMMAR_VOCAB)

        print("Loading Vosk model into memory...")
        self.model = Model(model_path)
        self._recognizer_type = KaldiRecognizer

        # Hardware initialization (BCM Pinout)
        self.light = LED(light_pin, active_high=not active_low)
        self.yellow_led = LED(yellow_pin)
        self.red_led = LED(red_pin)
        self.light.off()
        self.yellow_led.off()
        self.red_led.off()

        # Thread-safe wake-word state machine
        self._lock = threading.Lock()
        self.is_awake = False
        self._timeout_timer: Optional[threading.Timer] = None

    def _has_wake_word(self, text: str) -> bool:
        return any(wake in text for wake in self.WAKE_PHRASES)

    def _cancel_timer(self) -> None:
        if self._timeout_timer is not None:
            self._timeout_timer.cancel()
            self._timeout_timer = None

    def _on_wake_timeout(self) -> None:
        """Called automatically by threading.Timer when the wake window expires."""
        with self._lock:
            if self.is_awake:
                self.is_awake = False
                self.yellow_led.off()
                print(
                    f"[TIMEOUT] No valid command within {self.listen_timeout:.0f}s. "
                    "Returning to standby.",
                    flush=True,
                )

    def _activate_wake_window(self) -> None:
        """Turn on Yellow LED and start/reset the countdown timer."""
        self._cancel_timer()
        self.is_awake = True
        self.yellow_led.on()
        self.red_led.off()
        self._timeout_timer = threading.Timer(
            self.listen_timeout, self._on_wake_timeout
        )
        self._timeout_timer.daemon = True
        self._timeout_timer.start()

    def _sleep_standby(self) -> None:
        """Exit wake state and turn off Yellow status LED."""
        self._cancel_timer()
        self.is_awake = False
        self.yellow_led.off()

    def process_wav(self, audio_bytes: bytes) -> dict[str, str]:
        if len(audio_bytes) > self.max_audio_bytes:
            raise ValueError("Audio recording is too large")

        with wave.open(BytesIO(audio_bytes), "rb") as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                raise ValueError("Audio must be mono, 16-bit PCM WAV")

            sample_rate = wav_file.getframerate()
            if self.use_grammar:
                recognizer = self._recognizer_type(
                    self.model, sample_rate, self._grammar_json
                )
            else:
                recognizer = self._recognizer_type(self.model, sample_rate)

            segments = []
            while data := wav_file.readframes(2000):
                if recognizer.AcceptWaveform(data):
                    part = json.loads(recognizer.Result()).get("text", "")
                    if part:
                        segments.append(part)

            final_part = json.loads(recognizer.FinalResult()).get("text", "")
            if final_part:
                segments.append(final_part)

        raw_transcript = " ".join(segments).replace("[unk]", "").strip()
        transcript = " ".join(raw_transcript.split())  # Normalize spaces

        with self._lock:
            self.red_led.off()

            if not transcript:
                return {
                    "transcript": "",
                    "command": "NO_SPEECH (AWAKE)" if self.is_awake else "NO_SPEECH (SLEEPING)",
                }

            woke_this_turn = False
            if not self.is_awake and self._has_wake_word(transcript):
                self._activate_wake_window()
                woke_this_turn = True

            # If still sleeping, ignore all spoken words
            if not self.is_awake:
                return {
                    "transcript": transcript,
                    "command": "IGNORED (SAY 'HEY VOSK' FIRST)",
                }

            # Evaluate light command while awake
            command = check_light_command(transcript)
            if command == "ON":
                self.light.on()
                self._sleep_standby()
                return {"transcript": transcript, "command": "ON"}

            if command == "OFF":
                self.light.off()
                self._sleep_standby()
                return {"transcript": transcript, "command": "OFF"}

            # If the user only said "Hey Vosk", keep the window open for the next clip
            if woke_this_turn:
                return {
                    "transcript": transcript,
                    "command": f"AWAKE (LISTENING FOR {self.listen_timeout:.0f}s...)",
                }

            # Awake, but said something unrelated: refresh timer and keep Yellow LED on
            self._activate_wake_window()
            return {
                "transcript": transcript,
                "command": "UNKNOWN (STILL LISTENING...)",
            }

    def close(self) -> None:
        with self._lock:
            self._cancel_timer()
            self.light.off()
            self.yellow_led.off()
            self.red_led.off()
            self.light.close()
            self.yellow_led.close()
            self.red_led.close()


def make_handler(controller: VoiceCommandServer):
    class RequestHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/audio":
                self.send_error(404, "Use POST /audio")
                return

            content_length_header = self.headers.get("Content-Length")
            try:
                content_length = int(content_length_header or "0")
            except ValueError:
                self.send_error(400, "Content-Length must be an integer")
                return

            if content_length <= 0 or content_length > controller.max_audio_bytes:
                self.send_error(413, "Invalid or oversized audio payload")
                return

            try:
                result = controller.process_wav(self.rfile.read(content_length))
            except (OSError, ValueError, wave.Error) as error:
                controller.red_led.on()
                self.send_error(400, str(error))
                return

            response = json.dumps(result).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
            print(
                f'[REQUEST] "{result["transcript"]}" -> {result["command"]}',
                flush=True,
            )

        def log_message(self, format: str, *args) -> None:
            pass  # Silence noisy HTTP logs to keep console clean

    return RequestHandler


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive laptop voice recordings and control a Pi GPIO light."
    )
    parser.add_argument(
        "--model",
        default="/home/pi/Vosk_Project/vosk-model-small-en-us-0.15",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Interface to listen on")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--light-pin", type=int, default=22)
    parser.add_argument("--yellow-pin", type=int, default=27)
    parser.add_argument("--red-pin", type=int, default=17)
    parser.add_argument(
        "--timeout",
        type=float,
        default=6.0,
        help="Seconds to stay awake after hearing 'Hey Vosk' (default: 6.0)",
    )
    parser.add_argument("--active-low", action="store_true")
    parser.add_argument(
        "--no-grammar",
        action="store_true",
        help="Disable constrained grammar and use full open vocabulary",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    controller: Optional[VoiceCommandServer] = None
    server: Optional[ThreadingHTTPServer] = None
    try:
        controller = VoiceCommandServer(
            model_path=arguments.model,
            light_pin=arguments.light_pin,
            yellow_pin=arguments.yellow_pin,
            red_pin=arguments.red_pin,
            active_low=arguments.active_low,
            listen_timeout=arguments.timeout,
            use_grammar=not arguments.no_grammar,
        )
        server = ThreadingHTTPServer(
            (arguments.host, arguments.port), make_handler(controller)
        )
        print("=" * 65)
        print(f"  SERVER READY: http://{arguments.host}:{arguments.port}/audio")
        print("  Say 'Hey Vosk' or 'Hello' from the laptop client to wake.")
        print("=" * 65)
        server.serve_forever()
        return 0
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Server stopped.")
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[STARTUP ERROR] {error}", file=sys.stderr)
        return 1
    finally:
        if server is not None:
            server.server_close()
        if controller is not None:
            controller.close()
            print("[SHUTDOWN] GPIO resources released safely.")


if __name__ == "__main__":
    raise SystemExit(main())