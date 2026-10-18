"""Test Raspberry Pi LED/Relay responses to WAV audio commands with timing & relay protection."""

import argparse
import os
import signal
import time
from pathlib import Path
from typing import Any, Callable, Optional

from command_parser import check_light_command
from transcriber import VoskAudioTranscriber


class AudioLedTester:
    """Transcribe audio files, protect relay hardware, and display results on three LEDs."""

    RED_PIN = 17
    YELLOW_PIN = 27
    GREEN_PIN = 22

    def __init__(
        self,
        model_path: str,
        display_hold_delay: float = 1.5,
        inter_file_delay: float = 0.5,
        relay_cooldown: float = 0.5,
        min_confidence: float = 0.55,
        led_factory: Optional[Callable[[int], Any]] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if display_hold_delay < 0 or inter_file_delay < 0:
            raise ValueError("LED delays cannot be negative")

        if led_factory is None:
            try:
                from gpiozero import LED
            except ImportError as error:
                raise RuntimeError(
                    "gpiozero is required. Install with: pip3 install gpiozero"
                ) from error
            led_factory = LED

        self.display_hold_delay = display_hold_delay
        self.inter_file_delay = inter_file_delay
        self.relay_cooldown = relay_cooldown
        self._last_toggle_time = 0.0
        self._sleep = sleep

        self.transcriber = VoskAudioTranscriber(
            model_path, min_confidence=min_confidence
        )
        self.red_led = led_factory(self.RED_PIN)
        self.yellow_led = led_factory(self.YELLOW_PIN)
        self.green_led = led_factory(self.GREEN_PIN)
        self.reset_leds()

    def reset_leds(self) -> None:
        self.red_led.off()
        self.yellow_led.off()
        self.green_led.off()

    def close(self) -> None:
        self.reset_leds()
        self.red_led.close()
        self.yellow_led.close()
        self.green_led.close()

    def _hardware_state(self) -> str:
        return (
            f"Red: {'ON' if self.red_led.is_lit else 'OFF'} | "
            f"Yellow: {'ON' if self.yellow_led.is_lit else 'OFF'} | "
            f"Green: {'ON' if self.green_led.is_lit else 'OFF'}"
        )

    def _safe_set_green(self, turn_on: bool) -> None:
        """Prevent relay contact arcing by enforcing cooldown and skipping redundant toggles."""
        if self.green_led.is_lit == turn_on:
            state_str = "ON" if turn_on else "OFF"
            print(f"  [RELAY]  -> Already {state_str} (skipped redundant GPIO toggle)")
            return

        elapsed = time.monotonic() - self._last_toggle_time
        if elapsed < self.relay_cooldown:
            self._sleep(self.relay_cooldown - elapsed)

        if turn_on:
            self.green_led.on()
        else:
            self.green_led.off()
        self._last_toggle_time = time.monotonic()

    def _show_error(self, error: Exception) -> None:
        self.red_led.on()
        self.yellow_led.off()
        print("  [STATUS] -> RED LIGHT (ERROR)")
        print(f"  [DETAIL] -> Failed to process file: {error}")
        print(f"  [HARDWARE] -> {self._hardware_state()}")
        self._sleep(self.display_hold_delay)
        self.red_led.off()

    def _show_no_speech(self) -> None:
        self.yellow_led.on()
        self.red_led.off()
        print(f"  [TIMING] -> Processed in {self.transcriber.last_inference_ms:.1f} ms")
        print("  [STATUS] -> YELLOW LIGHT (NO SPEECH / LOW CONFIDENCE)")
        print(f"  [HARDWARE] -> {self._hardware_state()}")
        self._sleep(self.display_hold_delay)
        self.yellow_led.off()

    def process_audio_file(self, file_path: str) -> str:
        """Process one WAV file and return its result category."""
        print(f"\n[FILE] {os.path.basename(file_path)}")

        try:
            transcript = self.transcriber.transcribe_file(
                file_path, early_command_checker=check_light_command
            )
        except Exception as error:
            self._show_error(error)
            return "ERROR"

        if not transcript:
            self._show_no_speech()
            return "NO_SPEECH"

        print(
            f'  [TRANSCRIPT] -> "{transcript}" '
            f"(conf: {self.transcriber.last_confidence:.2f}, "
            f"time: {self.transcriber.last_inference_ms:.1f} ms)"
        )
        command = check_light_command(transcript)

        if command == "ON":
            self._safe_set_green(True)
            self.yellow_led.off()
            self.red_led.off()
            print("  [STATUS] -> GREEN LIGHT TOGGLED: ON")
        elif command == "OFF":
            self._safe_set_green(False)
            self.yellow_led.off()
            self.red_led.off()
            print("  [STATUS] -> GREEN LIGHT TOGGLED: OFF")
        else:
            self.yellow_led.on()
            self.red_led.off()
            print("  [STATUS] -> YELLOW LIGHT (UNRECOGNIZED COMMAND)")

        print(f"  [HARDWARE] -> {self._hardware_state()}")
        self._sleep(self.display_hold_delay)
        if command == "UNKNOWN":
            self.yellow_led.off()
        return command

    def process_audio_folder(self, audio_folder: str) -> list[str]:
        audio_files = self.transcriber.get_audio_files(audio_folder)
        if not audio_files:
            raise FileNotFoundError(f"No .wav files found in directory: {audio_folder}")

        results = []
        for index, file_path in enumerate(audio_files, start=1):
            print(f"\n{'=' * 65}")
            print(f"  FILE [{index}/{len(audio_files)}]: {os.path.basename(file_path)}")
            print("=" * 65)
            results.append(self.process_audio_file(file_path))
            if index < len(audio_files):
                self._sleep(self.inter_file_delay)
        return results


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Raspberry Pi LED responses to WAV audio commands."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("audio_file", nargs="?", help="WAV file to test")
    input_group.add_argument("--audio-folder", help="Folder of WAV files to test")
    parser.add_argument(
        "--model",
        default="/home/pi/Vosk_Project/vosk-model-small-en-us-0.15",
        help="Path to the Vosk model",
    )
    parser.add_argument(
        "--hold-delay",
        type=float,
        default=1.5,
        help="Seconds to hold LED result (default: 1.5, use 0 for instant testing)",
    )
    parser.add_argument(
        "--inter-file-delay",
        type=float,
        default=0.5,
        help="Seconds between files in folder mode (default: 0.5)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.55,
        help="Minimum word confidence score 0.0-1.0 (default: 0.55)",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    tester: Optional[AudioLedTester] = None

    # Handle systemd/service termination signals cleanly
    def _handle_sigterm(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        tester = AudioLedTester(
            model_path=arguments.model,
            display_hold_delay=arguments.hold_delay,
            inter_file_delay=arguments.inter_file_delay,
            min_confidence=arguments.min_confidence,
        )
        if arguments.audio_file:
            if not Path(arguments.audio_file).is_file():
                raise FileNotFoundError(f"Audio file not found: {arguments.audio_file}")
            tester.process_audio_file(arguments.audio_file)
        else:
            tester.process_audio_folder(arguments.audio_folder)
        return 0
    except KeyboardInterrupt:
        print("\n[ABORTED] Execution terminated cleanly.")
        return 130
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        print(f"\n[STARTUP ERROR] {error}")
        if tester is not None:
            tester.red_led.on()
            tester._sleep(tester.display_hold_delay)
        return 1
    finally:
        if tester is not None:
            tester.close()
            print("[SHUTDOWN] GPIO resources released cleanly.")


if __name__ == "__main__":
    raise SystemExit(main())