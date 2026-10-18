"""Low-latency Raspberry Pi voice light controller with wake-word & grammar constraints."""

import argparse
import json
import queue
import sys
import time
from typing import Optional

from command_parser import check_light_command


class LiveVoiceLightController:
    """Recognize wake words instantly via PartialResult and control GPIO pins."""

    # Because 'vosk' is not in standard English dictionaries, Kaldi maps the sound
    # "Hey Vosk" to phonetic equivalents like "hey boss" or "hey box".
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

    # Constrained grammar list: restricts Kaldi to only these phrases + [unk] for noise
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
        sample_rate: int = 16000,
        device: Optional[int] = None,
        active_low: bool = False,
        listen_timeout: float = 5.0,
        use_grammar: bool = True,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")

        try:
            import sounddevice as sd
            from gpiozero import LED
            from vosk import KaldiRecognizer, Model, SetLogLevel
        except ImportError as error:
            raise RuntimeError(
                "Missing dependencies. Install with: pip3 install vosk sounddevice gpiozero"
            ) from error

        SetLogLevel(-1)
        self._sounddevice = sd
        # Bounded queue prevents memory buildup if CPU momentarily spikes
        self._audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=30)
        self.sample_rate = sample_rate
        self.device = device

        model = Model(model_path)
        if use_grammar:
            self.recognizer = KaldiRecognizer(
                model, sample_rate, json.dumps(self.GRAMMAR_VOCAB)
            )
        else:
            self.recognizer = KaldiRecognizer(model, sample_rate)

        # Hardware outputs (BCM Pinout: Green=22, Yellow=27, Red=17)
        self.light = LED(light_pin, active_high=not active_low)
        self.yellow_led = LED(yellow_pin)
        self.red_led = LED(red_pin)
        self.light.off()
        self.yellow_led.off()
        self.red_led.off()

        # State machine tracking
        self.is_awake = False
        self.wake_timestamp = 0.0
        self.listen_timeout = listen_timeout

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"[AUDIO WARNING] {status}", file=sys.stderr)
        try:
            self._audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            pass  # Drop oldest/excess frame rather than lagging behind real-time

    def _clear_audio_queue(self) -> None:
        """Flush pending audio chunks and reset recognizer state after an action."""
        self.recognizer.Reset()
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def _has_wake_word(self, text: str) -> bool:
        return any(wake in text for wake in self.WAKE_PHRASES)

    def _wake_up(self, detected_text: str) -> None:
        self.is_awake = True
        self.wake_timestamp = time.time()
        self.yellow_led.on()
        print(f"\n[WAKE DETECTED] Heard '{detected_text}'")
        print(f"  >>> Listening for command ({self.listen_timeout:.0f}s window)...")

    def _sleep_standby(self, reason: str = "") -> None:
        if reason:
            print(f"  [{reason}] Returning to standby.\n")
        self.is_awake = False
        self.yellow_led.off()
        self._clear_audio_queue()

    def _try_execute_command(self, transcript: str) -> bool:
        command = check_light_command(transcript)
        if command == "ON":
            self.light.on()
            print("  [ACTION] 🟢 Light turned ON")
            return True
        if command == "OFF":
            self.light.off()
            print("  [ACTION] ⚪ Light turned OFF")
            return True
        return False

    def run(self) -> None:
        """Stream audio with 125ms blocks, instant partial wake detection, and timeout."""
        print("=" * 65)
        print("  READY: Say 'Hey Vosk', 'Hey Light', or 'Hello' to wake.")
        print("=" * 65)

        with self._sounddevice.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=2000,  # 125ms chunks for snappy real-time response
            device=self.device,
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
        ):
            while True:
                # 1. Check if active listening window timed out
                if self.is_awake and (time.time() - self.wake_timestamp > self.listen_timeout):
                    self._sleep_standby("TIMEOUT")

                try:
                    audio_data = self._audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # 2. Feed audio to Kaldi
                is_final = self.recognizer.AcceptWaveform(audio_data)

                if is_final:
                    result = json.loads(self.recognizer.Result())
                    transcript = result.get("text", "").replace("[unk]", "").strip()
                    if not transcript:
                        continue

                    # Wake up if wake word was in finalized utterance
                    if not self.is_awake and self._has_wake_word(transcript):
                        self._wake_up(transcript)

                    # If awake, evaluate finalized transcript for ON/OFF command
                    if self.is_awake:
                        print(f'  [HEARD] "{transcript}"')
                        if self._try_execute_command(transcript):
                            self._sleep_standby("DONE")
                        elif not self._has_wake_word(transcript):
                            print("  [IGNORED] Not a valid light command. Still listening...")

                else:
                    # 3. Check PartialResult for instant wake-word activation
                    partial_res = json.loads(self.recognizer.PartialResult())
                    partial_text = partial_res.get("partial", "").replace("[unk]", "").strip()

                    if not partial_text:
                        continue

                    if not self.is_awake and self._has_wake_word(partial_text):
                        self._wake_up(partial_text)

    def close(self) -> None:
        """Turn off LEDs and release GPIO pins safely."""
        self.light.off()
        self.yellow_led.off()
        self.red_led.off()
        self.light.close()
        self.yellow_led.close()
        self.red_led.close()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control a Raspberry Pi light with an offline wake word."
    )
    parser.add_argument(
        "--model",
        default="/home/pi/Vosk_Project/vosk-model-small-en-us-0.15",
        help="Path to the Vosk model",
    )
    parser.add_argument("--light-pin", type=int, default=22, help="Green LED/Relay BCM pin")
    parser.add_argument("--yellow-pin", type=int, default=27, help="Yellow status BCM pin")
    parser.add_argument("--red-pin", type=int, default=17, help="Red error BCM pin")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--timeout", type=float, default=5.0, help="Wake window in seconds")
    parser.add_argument("--device", type=int, default=None, help="Microphone device index")
    parser.add_argument("--active-low", action="store_true", help="Active-low relay support")
    parser.add_argument(
        "--no-grammar",
        action="store_true",
        help="Disable constrained grammar and use full open vocabulary",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    controller: Optional[LiveVoiceLightController] = None
    try:
        controller = LiveVoiceLightController(
            model_path=args.model,
            light_pin=args.light_pin,
            yellow_pin=args.yellow_pin,
            red_pin=args.red_pin,
            sample_rate=args.sample_rate,
            device=args.device,
            active_low=args.active_low,
            listen_timeout=args.timeout,
            use_grammar=not args.no_grammar,
        )
        controller.run()
        return 0
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] User stopped listening.")
        return 130
    except Exception as error:
        print(f"\n[STARTUP ERROR] {error}", file=sys.stderr)
        if controller is not None:
            controller.red_led.on()
            time.sleep(3.0)
        return 1
    finally:
        if controller is not None:
            controller.close()
            print("[SHUTDOWN] GPIO resources released safely.")


if __name__ == "__main__":
    raise SystemExit(main())