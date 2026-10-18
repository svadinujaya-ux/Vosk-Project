"""Fast, real-world Vosk WAV transcriber with auto-resampling, silence trimming, and confidence checks."""

import array
import json
import math
import os
import time
import wave
from pathlib import Path
from typing import Callable, Optional
from vosk import Model, KaldiRecognizer, SetLogLevel

# Silence Kaldi debug logs
SetLogLevel(-1)


class VoskAudioTranscriber:
    TARGET_SAMPLE_RATE = 16000

    GRAMMAR_VOCAB = [
        "hey vosk", "hey boss", "hey box", "hey light", "hello",
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
        use_grammar: bool = True,
        min_confidence: float = 0.55,
        silence_rms_threshold: int = 120,
    ) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        print("Loading Vosk model into memory...")
        self.model = Model(model_path)
        self.min_confidence = min_confidence
        self.silence_rms_threshold = silence_rms_threshold

        # Pre-allocate a single persistent 16kHz recognizer for consistent speed
        if use_grammar:
            self.recognizer = KaldiRecognizer(
                self.model, self.TARGET_SAMPLE_RATE, json.dumps(self.GRAMMAR_VOCAB)
            )
        else:
            self.recognizer = KaldiRecognizer(self.model, self.TARGET_SAMPLE_RATE)
        self.recognizer.SetWords(True)

        self.last_inference_ms: float = 0.0
        self.last_confidence: float = 1.0

    @staticmethod
    def _normalize_to_16k_mono_pcm(
        raw_bytes: bytes, channels: int, sample_width: int, src_rate: int
    ) -> bytes:
        """Convert arbitrary PCM WAV data into 16-bit, 16kHz Mono PCM."""
        if sample_width != 2:
            raise ValueError(f"Expected 16-bit PCM (2 bytes), got {sample_width * 8}-bit")

        samples = array.array("h")
        samples.frombytes(raw_bytes)

        # 1. Downmix stereo/multi-channel to Mono by averaging channels
        if channels > 1:
            mono_samples = array.array(
                "h",
                (
                    sum(samples[i : i + channels]) // channels
                    for i in range(0, len(samples), channels)
                ),
            )
            samples = mono_samples

        # 2. Resample to 16,000 Hz if recorded at 44.1kHz, 48kHz, etc.
        if src_rate != VoskAudioTranscriber.TARGET_SAMPLE_RATE and len(samples) > 1:
            ratio = src_rate / VoskAudioTranscriber.TARGET_SAMPLE_RATE
            target_len = int(len(samples) / ratio)
            resampled = array.array("h", [0] * target_len)
            max_idx = len(samples) - 1
            for i in range(target_len):
                pos = i * ratio
                idx = int(pos)
                frac = pos - idx
                nxt = idx + 1 if idx < max_idx else max_idx
                resampled[i] = int((1.0 - frac) * samples[idx] + frac * samples[nxt])
            samples = resampled

        return samples.tobytes()

    def _trim_silence(self, pcm_bytes: bytes, frame_samples: int = 800) -> bytes:
        """Trim leading and trailing silence blocks so WAV length doesn't skew latency."""
        if not pcm_bytes:
            return pcm_bytes

        samples = array.array("h")
        samples.frombytes(pcm_bytes)
        total = len(samples)
        if total <= frame_samples:
            return pcm_bytes

        # Find first active speech block
        start_idx = 0
        for i in range(0, total, frame_samples):
            block = samples[i : i + frame_samples]
            rms = math.sqrt(sum(s * s for s in block) / len(block))
            if rms >= self.silence_rms_threshold:
                start_idx = max(0, i - frame_samples)  # Keep 50ms lead-in pad
                break
        else:
            return b""  # Entire file is silent

        # Find last active speech block
        end_idx = total
        for i in range(total - frame_samples, start_idx, -frame_samples):
            block = samples[i : i + frame_samples]
            rms = math.sqrt(sum(s * s for s in block) / max(1, len(block)))
            if rms >= self.silence_rms_threshold:
                end_idx = min(total, i + 2 * frame_samples)  # Keep 100ms tail pad
                break

        return samples[start_idx:end_idx].tobytes()

    def _extract_verified_text(self, res_json: str) -> tuple[str, float]:
        """Extract recognized text and filter out low-confidence acoustic guesses."""
        data = json.loads(res_json)
        text = data.get("text", "").replace("[unk]", "").strip()
        if not text:
            return "", 0.0

        words_info = [
            w for w in data.get("result", []) if w.get("word") != "[unk]"
        ]
        if not words_info:
            return text, 1.0

        avg_conf = sum(w.get("conf", 1.0) for w in words_info) / len(words_info)
        if avg_conf < self.min_confidence:
            return "", avg_conf
        return " ".join(text.split()), avg_conf

    def transcribe_file(
        self,
        audio_path: str,
        early_command_checker: Optional[Callable[[str], str]] = None,
    ) -> str:
        """
        Transcribe a WAV file with normalized 16kHz audio, silence trimming,
        persistent recognizer reuse, and optional early command exit.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        start_time = time.perf_counter()

        with wave.open(audio_path, "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            src_rate = wf.getframerate()
            raw_frames = wf.readframes(wf.getnframes())

        # Normalize to 16kHz Mono and strip dead air at ends
        pcm_16k = self._normalize_to_16k_mono_pcm(
            raw_frames, channels, sample_width, src_rate
        )
        trimmed_pcm = self._trim_silence(pcm_16k)

        if not trimmed_pcm:
            self.last_inference_ms = (time.perf_counter() - start_time) * 1000.0
            self.last_confidence = 0.0
            return ""

        # Reset persistent recognizer instead of reallocating C++ graphs
        self.recognizer.Reset()

        segments: list[str] = []
        confidences: list[float] = []
        chunk_bytes = 4000  # 2000 samples = 125ms at 16kHz

        for offset in range(0, len(trimmed_pcm), chunk_bytes):
            chunk = trimmed_pcm[offset : offset + chunk_bytes]
            if self.recognizer.AcceptWaveform(chunk):
                seg_text, conf = self._extract_verified_text(self.recognizer.Result())
                if seg_text:
                    segments.append(seg_text)
                    confidences.append(conf)
                    # Optional early exit: stop reading remaining file once command is clear
                    if early_command_checker is not None:
                        if early_command_checker(" ".join(segments)) in ("ON", "OFF"):
                            break

        final_text, final_conf = self._extract_verified_text(
            self.recognizer.FinalResult()
        )
        if final_text:
            segments.append(final_text)
            confidences.append(final_conf)

        self.last_inference_ms = (time.perf_counter() - start_time) * 1000.0
        self.last_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )
        return " ".join(segments).strip()

    def get_audio_files(self, folder_path: str) -> list[str]:
        """Find all .wav files in the folder in deterministic alphabetical order."""
        folder = Path(folder_path)
        if not folder.is_dir():
            raise FileNotFoundError(f"Audio folder not found: {folder_path}")
        return sorted(str(p) for p in folder.glob("*.wav"))