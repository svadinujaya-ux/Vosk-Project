"""PC Desktop GUI Frontend for the Raspberry Pi Vosk Voice Server.

Requires only `sounddevice` (pip install sounddevice) and standard Python libraries.
"""

import array
import json
import math
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
import urllib.error
import urllib.request
import wave
from io import BytesIO
from typing import Optional

try:
    import sounddevice as sd
except ImportError as error:
    raise RuntimeError(
        "Install sounddevice on your PC first: pip install sounddevice"
    ) from error


class PCVoiceClientApp:
    SAMPLE_RATE = 16000
    BLOCK_SIZE = 1024

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Raspberry Pi Voice Controller — PC Frontend")
        self.root.geometry("680x620")
        self.root.configure(bg="#0f172a")
        self.root.resizable(False, False)

        # Runtime state
        self.is_listening = False
        self.is_manual_recording = False
        self.audio_queue: queue.Queue[bytes] = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None

        # UI Variables
        self.server_url_var = tk.StringVar(value="http://192.168.1.50:8765")
        self.threshold_var = tk.IntVar(value=500)
        self.silence_var = tk.DoubleVar(value=0.8)
        self.status_text_var = tk.StringVar(value="Idle — Click 'Start Hands-Free' or 'Push to Talk'")
        self.last_transcript_var = tk.StringVar(value='Waiting for speech...')
        self.rms_level_var = tk.DoubleVar(value=0.0)

        self.devices = self._get_input_devices()
        self.selected_device_var = tk.StringVar(
            value=self.devices[0] if self.devices else "Default Microphone"
        )

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _get_input_devices(self) -> list[str]:
        """Return a list of available microphone input devices on the PC."""
        device_list = ["Default Microphone"]
        try:
            for idx, dev in enumerate(sd.query_devices()):
                if dev.get("max_input_channels", 0) > 0:
                    device_list.append(f"[{idx}] {dev['name']}")
        except Exception:
            pass
        return device_list

    def _get_selected_device_index(self) -> Optional[int]:
        val = self.selected_device_var.get()
        if val.startswith("[") and "]" in val:
            try:
                return int(val[1 : val.index("]")])
            except ValueError:
                return None
        return None

    def _build_ui(self) -> None:
        # Header
        header = tk.Frame(self.root, bg="#1e293b", pady=12, padx=16)
        header.pack(fill="x")
        tk.Label(
            header,
            text="🎙️ Raspberry Pi Vosk Voice Controller",
            font=("Segoe UI", 14, "bold"),
            bg="#1e293b",
            fg="#f8fafc",
        ).pack(side="left")

        # Connection & Device Settings Card
        conn_frame = tk.Frame(self.root, bg="#1e293b", pady=12, padx=16)
        conn_frame.pack(fill="x", padx=16, pady=(14, 8))

        tk.Label(
            conn_frame, text="Pi Server URL:", bg="#1e293b", fg="#94a3b8", font=("Segoe UI", 10)
        ).grid(row=0, column=0, sticky="w", pady=4)
        tk.Entry(
            conn_frame,
            textvariable=self.server_url_var,
            width=32,
            font=("Consolas", 10),
            bg="#0f172a",
            fg="#f8fafc",
            insertbackground="#f8fafc",
            relief="flat",
        ).grid(row=0, column=1, sticky="w", padx=8, pady=4, ipady=4)

        tk.Label(
            conn_frame, text="PC Microphone:", bg="#1e293b", fg="#94a3b8", font=("Segoe UI", 10)
        ).grid(row=1, column=0, sticky="w", pady=4)
        device_menu = ttk.Combobox(
            conn_frame,
            textvariable=self.selected_device_var,
            values=self.devices,
            state="readonly",
            width=42,
        )
        device_menu.grid(row=1, column=1, columnspan=2, sticky="w", padx=8, pady=4)

        # Virtual Raspberry Pi LED Status Panel
        led_frame = tk.Frame(self.root, bg="#1e293b", pady=14, padx=16)
        led_frame.pack(fill="x", padx=16, pady=8)

        tk.Label(
            led_frame,
            text="RASPBERRY PI HARDWARE STATUS",
            bg="#1e293b",
            fg="#64748b",
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        indicators = tk.Frame(led_frame, bg="#1e293b")
        indicators.pack(fill="x")

        self.red_canvas, self.red_oval = self._create_led_widget(
            indicators, "RED (Error / Offline)", "#450a0a"
        )
        self.yellow_canvas, self.yellow_oval = self._create_led_widget(
            indicators, "YELLOW (Awake / Unknown)", "#422006"
        )
        self.green_canvas, self.green_oval = self._create_led_widget(
            indicators, "GREEN (Light GPIO 22)", "#052e16"
        )

        # Microphone Level & Threshold Calibration Card
        audio_frame = tk.Frame(self.root, bg="#1e293b", pady=12, padx=16)
        audio_frame.pack(fill="x", padx=16, pady=8)

        tk.Label(
            audio_frame,
            text="Live Mic Volume (RMS):",
            bg="#1e293b",
            fg="#94a3b8",
            font=("Segoe UI", 9),
        ).grid(row=0, column=0, sticky="w")

        self.level_canvas = tk.Canvas(
            audio_frame, width=420, height=18, bg="#0f172a", highlightthickness=0
        )
        self.level_canvas.grid(row=0, column=1, padx=10, pady=4)
        self.level_bar = self.level_canvas.create_rectangle(0, 0, 0, 18, fill="#38bdf8", width=0)
        self.threshold_line = self.level_canvas.create_line(70, 0, 70, 18, fill="#f43f5e", width=2)

        tk.Label(
            audio_frame,
            text="Voice Trigger Threshold:",
            bg="#1e293b",
            fg="#94a3b8",
            font=("Segoe UI", 9),
        ).grid(row=1, column=0, sticky="w")

        thresh_slider = tk.Scale(
            audio_frame,
            from_=100,
            to=2500,
            orient="horizontal",
            variable=self.threshold_var,
            bg="#1e293b",
            fg="#f8fafc",
            highlightthickness=0,
            troughcolor="#0f172a",
            length=420,
            command=self._update_threshold_marker,
        )
        thresh_slider.grid(row=1, column=1, padx=10)

        # Controls Row
        btn_frame = tk.Frame(self.root, bg="#0f172a", pady=8)
        btn_frame.pack(fill="x", padx=16)

        self.auto_btn = tk.Button(
            btn_frame,
            text="▶ Start Hands-Free Listening",
            font=("Segoe UI", 11, "bold"),
            bg="#0284c7",
            fg="white",
            activebackground="#0369a1",
            activeforeground="white",
            relief="flat",
            padx=18,
            pady=8,
            command=self._toggle_hands_free,
        )
        self.auto_btn.pack(side="left", expand=True, fill="x", padx=(0, 6))

        self.ptt_btn = tk.Button(
            btn_frame,
            text="🎙️ Hold or Click to Speak Now",
            font=("Segoe UI", 11, "bold"),
            bg="#334155",
            fg="white",
            activebackground="#475569",
            activeforeground="white",
            relief="flat",
            padx=18,
            pady=8,
            command=self._trigger_push_to_talk,
        )
        self.ptt_btn.pack(side="left", expand=True, fill="x", padx=(6, 0))

        # Status & Last Transcript Banner
        status_box = tk.Frame(self.root, bg="#1e293b", pady=10, padx=16)
        status_box.pack(fill="x", padx=16, pady=6)

        tk.Label(
            status_box,
            textvariable=self.status_text_var,
            bg="#1e293b",
            fg="#38bdf8",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")

        tk.Label(
            status_box,
            textvariable=self.last_transcript_var,
            bg="#1e293b",
            fg="#f8fafc",
            font=("Segoe UI", 11),
        ).pack(anchor="w", pady=(4, 0))

        # Event Log Box
        log_frame = tk.Frame(self.root, bg="#1e293b", padx=12, pady=10)
        log_frame.pack(fill="both", expand=True, padx=16, pady=(6, 16))

        self.log_widget = tk.Text(
            log_frame,
            height=8,
            bg="#090d16",
            fg="#cbd5e1",
            font=("Consolas", 9),
            relief="flat",
            state="disabled",
        )
        self.log_widget.pack(fill="both", expand=True)
        self._update_threshold_marker()

    def _create_led_widget(self, parent: tk.Widget, label: str, off_color: str):
        frame = tk.Frame(parent, bg="#1e293b")
        frame.pack(side="left", expand=True)
        canvas = tk.Canvas(frame, width=28, height=28, bg="#1e293b", highlightthickness=0)
        canvas.pack(side="left", padx=(0, 8))
        oval = canvas.create_oval(3, 3, 25, 25, fill=off_color, outline="#334155", width=2)
        tk.Label(frame, text=label, bg="#1e293b", fg="#e2e8f0", font=("Segoe UI", 9, "bold")).pack(
            side="left"
        )
        return canvas, oval

    def _set_leds(
        self,
        red: Optional[bool] = None,
        yellow: Optional[bool] = None,
        green: Optional[bool] = None,
    ) -> None:
        if red is not None:
            self.red_canvas.itemconfig(self.red_oval, fill="#ef4444" if red else "#450a0a")
        if yellow is not None:
            self.yellow_canvas.itemconfig(
                self.yellow_oval, fill="#facc15" if yellow else "#422006"
            )
        if green is not None:
            self.green_canvas.itemconfig(self.green_oval, fill="#22c55e" if green else "#052e16")

    def _update_threshold_marker(self, *_args) -> None:
        x_pos = min(420, int((self.threshold_var.get() / 2500.0) * 420))
        self.level_canvas.coords(self.threshold_line, x_pos, 0, x_pos, 18)

    def _update_level_bar(self, rms: float) -> None:
        width = min(420, int((rms / 2500.0) * 420))
        color = "#22c55e" if rms >= self.threshold_var.get() else "#38bdf8"
        self.level_canvas.coords(self.level_bar, 0, 0, width, 18)
        self.level_canvas.itemconfig(self.level_bar, fill=color)

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", f"[{timestamp}] {message}\n")
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")

    def _toggle_hands_free(self) -> None:
        if self.is_listening:
            self.is_listening = False
            self.auto_btn.configure(text="▶ Start Hands-Free Listening", bg="#0284c7")
            self.status_text_var.set("Stopped.")
            self._update_level_bar(0)
        else:
            self.is_listening = True
            self.auto_btn.configure(text="⏹ Stop Hands-Free Listening", bg="#e11d48")
            self.status_text_var.set("Listening for voice above threshold...")
            self.worker_thread = threading.Thread(target=self._audio_loop, daemon=True)
            self.worker_thread.start()

    def _trigger_push_to_talk(self) -> None:
        if not self.is_listening:
            self._toggle_hands_free()
        self.is_manual_recording = True
        self.status_text_var.set("🎙️ Recording utterance now... Speak clearly!")

    def _audio_loop(self) -> None:
        """Background thread capturing PC microphone audio and dispatching WAV clips."""
        def callback(indata, frames, time_info, status) -> None:
            self.audio_queue.put(bytes(indata))

        device_idx = self._get_selected_device_index()

        try:
            with sd.RawInputStream(
                samplerate=self.SAMPLE_RATE,
                blocksize=self.BLOCK_SIZE,
                dtype="int16",
                channels=1,
                device=device_idx,
                callback=callback,
            ):
                started = False
                silent_for = 0.0
                recorded_for = 0.0
                audio_chunks: list[bytes] = []
                pre_roll: list[bytes] = []

                while self.is_listening:
                    try:
                        data = self.audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    duration = len(data) / 2 / self.SAMPLE_RATE
                    samples = array.array("h")
                    samples.frombytes(data)
                    rms = (
                        math.sqrt(sum(s * s for s in samples) / len(samples))
                        if samples
                        else 0.0
                    )

                    self.root.after(0, self._update_level_bar, rms)

                    if rms >= self.threshold_var.get() or self.is_manual_recording:
                        if not started:
                            started = True
                            audio_chunks.extend(pre_roll)
                            self.root.after(
                                0,
                                self.status_text_var.set,
                                "🔴 Voice detected — Recording...",
                            )
                        self.is_manual_recording = False
                        silent_for = 0.0
                    elif started:
                        silent_for += duration
                    else:
                        # Keep 250ms pre-roll buffer so first syllable isn't clipped
                        pre_roll.append(data)
                        if len(pre_roll) > 4:
                            pre_roll.pop(0)

                    if started:
                        audio_chunks.append(data)
                        recorded_for += duration

                    if started and (
                        silent_for >= self.silence_var.get() or recorded_for >= 6.0
                    ):
                        wav_bytes = self._to_wav(audio_chunks)
                        started = False
                        silent_for = 0.0
                        recorded_for = 0.0
                        audio_chunks.clear()
                        pre_roll.clear()

                        self.root.after(
                            0, self.status_text_var.set, "⚡ Sending audio to Raspberry Pi..."
                        )
                        self._send_to_pi(wav_bytes)
                        if self.is_listening:
                            self.root.after(
                                0,
                                self.status_text_var.set,
                                "Listening for voice above threshold...",
                            )

        except Exception as err:
            self.root.after(0, self._handle_stream_error, str(err))

    def _handle_stream_error(self, err_msg: str) -> None:
        self.is_listening = False
        self.auto_btn.configure(text="▶ Start Hands-Free Listening", bg="#0284c7")
        self.status_text_var.set(f"Audio Error: {err_msg}")
        self._set_leds(red=True)
        self._log(f"[AUDIO ERROR] {err_msg}")

    def _to_wav(self, chunks: list[bytes]) -> bytes:
        output = BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.SAMPLE_RATE)
            wav_file.writeframes(b"".join(chunks))
        return output.getvalue()

    def _send_to_pi(self, audio_bytes: bytes) -> None:
        url = self.server_url_var.get().strip().rstrip("/") + "/audio"
        request = urllib.request.Request(
            url,
            data=audio_bytes,
            headers={"Content-Type": "audio/wav"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            self.root.after(0, self._on_server_error, str(error))
            return

        self.root.after(0, self._on_server_response, result)

    def _on_server_error(self, error_msg: str) -> None:
        self._set_leds(red=True)
        self.last_transcript_var.set(f"Network Error: Cannot reach Pi")
        self._log(f"[NETWORK ERROR] {error_msg}")

    def _on_server_response(self, result: dict) -> None:
        transcript = result.get("transcript", "")
        command = result.get("command", "UNKNOWN")

        self._set_leds(red=False)
        display_text = transcript if transcript else "(No decipherable speech)"
        self.last_transcript_var.set(f'Heard: "{display_text}"  ➔  {command}')
        self._log(f'Heard "{display_text}" -> {command}')

        if command == "ON":
            self._set_leds(green=True, yellow=False)
        elif command == "OFF":
            self._set_leds(green=False, yellow=False)
        elif "AWAKE" in command or "UNKNOWN" in command:
            self._set_leds(yellow=True)
        else:
            self._set_leds(yellow=False)

    def _on_close(self) -> None:
        self.is_listening = False
        self.root.destroy()


if __name__ == "__main__":
    app_root = tk.Tk()
    app = PCVoiceClientApp(app_root)
    app_root.mainloop()