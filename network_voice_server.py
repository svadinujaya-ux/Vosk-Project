"""Raspberry Pi Voice & Mobile App Server (Supports both Mobile Web App and PC App).

- GET  /        -> Serves the touch-friendly Mobile App interface
- GET  /status  -> Returns live LED / Relay hardware states
- POST /audio   -> Processes 16-bit PCM WAV voice commands (from Phone or PC)
- POST /manual  -> Toggles the light directly via touch buttons
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


MOBILE_APP_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="mobile-web-app-capable" content="yes">
  <title>Pi Voice Controller</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #0284c7;
    }
    * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
    body {
      margin: 0; padding: 16px;
      background: var(--bg); color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      display: flex; flex-direction: column; align-items: center;
      min-height: 100vh; user-select: none;
    }
    .container { width: 100%; max-width: 420px; display: flex; flex-direction: column; gap: 14px; }
    .header {
      background: var(--card); padding: 14px 16px; border-radius: 12px;
      text-align: center; font-weight: 700; font-size: 1.15rem;
    }
    .card {
      background: var(--card); padding: 14px 16px; border-radius: 12px;
      display: flex; flex-direction: column; gap: 10px;
    }
    .card-title { font-size: 0.75rem; font-weight: 700; color: var(--muted); letter-spacing: 0.05em; }
    .leds { display: flex; justify-content: space-around; padding: 6px 0; }
    .led-item { display: flex; flex-direction: column; align-items: center; gap: 6px; font-size: 0.75rem; color: var(--muted); }
    .led-dot { width: 24px; height: 24px; border-radius: 50%; border: 2px solid #334155; transition: 0.2s; }
    .led-red { background: #450a0a; }
    .led-red.on { background: #ef4444; box-shadow: 0 0 12px #ef4444; }
    .led-yellow { background: #422006; }
    .led-yellow.on { background: #facc15; box-shadow: 0 0 12px #facc15; }
    .led-green { background: #052e16; }
    .led-green.on { background: #22c55e; box-shadow: 0 0 12px #22c55e; }

    .mic-section {
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      padding: 20px 10px; gap: 14px;
    }
    .mic-btn {
      width: 120px; height: 120px; border-radius: 50%; border: none;
      background: var(--accent); color: white; font-size: 2.5rem;
      display: flex; align-items: center; justify-content: center;
      box-shadow: 0 6px 20px rgba(2, 132, 199, 0.4);
      cursor: pointer; transition: transform 0.15s, background 0.2s;
    }
    .mic-btn.recording {
      background: #e11d48;
      box-shadow: 0 0 28px rgba(225, 29, 72, 0.7);
      transform: scale(1.08);
    }
    .status-text { font-size: 0.95rem; font-weight: 600; color: #38bdf8; text-align: center; }
    .transcript-box {
      font-size: 0.95rem; color: var(--text); text-align: center;
      background: #090d16; padding: 12px; border-radius: 8px; min-height: 44px;
    }
    .manual-btns { display: flex; gap: 10px; }
    .btn {
      flex: 1; padding: 12px; border: none; border-radius: 8px;
      font-weight: 700; font-size: 0.95rem; cursor: pointer; color: white;
    }
    .btn-on { background: #16a34a; }
    .btn-off { background: #475569; }
    .warn-box {
      display: none; background: #450a0a; border: 1px solid #ef4444;
      color: #fecaca; padding: 12px; border-radius: 10px; font-size: 0.82rem; line-height: 1.4;
    }
    code { background: #090d16; padding: 2px 5px; border-radius: 4px; color: #38bdf8; word-break: break-all; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">🎙️ Pi Voice Controller</div>

    <div id="httpsWarn" class="warn-box">
      <b>⚠️ Enable Phone Microphone Over Local Wi-Fi:</b><br>
      Mobile browsers block the mic on <code>http://</code> unless enabled:<br>
      1. Copy and open: <code>chrome://flags/#unsafely-treat-insecure-origin-as-secure</code><br>
      2. Type <code id="originUrl"></code> in the box, set to <b>Enabled</b>, and tap <b>Relaunch</b>.
    </div>

    <div class="card">
      <div class="card-title">RASPBERRY PI HARDWARE STATUS</div>
      <div class="leds">
        <div class="led-item"><div id="ledRed" class="led-dot led-red"></div><span>ERROR</span></div>
        <div class="led-item"><div id="ledYellow" class="led-dot led-yellow"></div><span>AWAKE / WAIT</span></div>
        <div class="led-item"><div id="ledGreen" class="led-dot led-green"></div><span>LIGHT (GPIO 22)</span></div>
      </div>
    </div>

    <div class="card mic-section">
      <button id="micBtn" class="mic-btn">🎙️</button>
      <div id="statusText" class="status-text">Tap to Speak (3s) or Hold to Record</div>
      <div id="transcriptBox" class="transcript-box">Say "Hey Vosk, turn on the light"</div>
    </div>

    <div class="card">
      <div class="card-title">MANUAL TOUCH OVERRIDE</div>
      <div class="manual-btns">
        <button class="btn btn-on" onclick="sendManual('ON')">💡 Turn ON</button>
        <button class="btn btn-off" onclick="sendManual('OFF')">🌑 Turn OFF</button>
      </div>
    </div>
  </div>

<script>
  const micBtn = document.getElementById('micBtn');
  const statusText = document.getElementById('statusText');
  const transcriptBox = document.getElementById('transcriptBox');
  const ledRed = document.getElementById('ledRed');
  const ledYellow = document.getElementById('ledYellow');
  const ledGreen = document.getElementById('ledGreen');

  document.getElementById('originUrl').textContent = window.location.origin;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    document.getElementById('httpsWarn').style.display = 'block';
  }

  let audioCtx = null;
  let mediaStream = null;
  let processor = null;
  let recordedChunks = [];
  let isRecording = false;
  let autoStopTimer = null;

  function setLeds(red, yellow, green) {
    if (red !== null) ledRed.classList.toggle('on', red);
    if (yellow !== null) ledYellow.classList.toggle('on', yellow);
    if (green !== null) ledGreen.classList.toggle('on', green);
  }

  async function refreshStatus() {
    try {
      const res = await fetch('/status');
      const data = await res.json();
      setLeds(data.red, data.yellow, data.green);
    } catch (e) {
      setLeds(true, null, null);
    }
  }
  setInterval(refreshStatus, 2500);
  refreshStatus();

  async function sendManual(state) {
    try {
      const res = await fetch('/manual', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({action: state})
      });
      const data = await res.json();
      setLeds(false, data.yellow, data.green);
      transcriptBox.textContent = `Manual Switch ➔ ${data.command}`;
    } catch (e) {
      setLeds(true, null, null);
      statusText.textContent = 'Network Error: Cannot reach Pi';
    }
  }

  async function startRecording() {
    if (isRecording) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      document.getElementById('httpsWarn').style.display = 'block';
      statusText.textContent = 'Mic blocked on HTTP — see red instructions above';
      return;
    }
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioCtx.createMediaStreamSource(mediaStream);
      processor = audioCtx.createScriptProcessor(4096, 1, 1);
      recordedChunks = [];

      processor.onaudioprocess = (e) => {
        if (!isRecording) return;
        const input = e.inputBuffer.getChannelData(0);
        recordedChunks.push(new Float32Array(input));
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);

      isRecording = true;
      micBtn.classList.add('recording');
      statusText.textContent = '🔴 Listening... Tap again when done';

      autoStopTimer = setTimeout(() => {
        if (isRecording) stopRecording();
      }, 4000);
    } catch (err) {
      statusText.textContent = 'Mic Error: ' + err.message;
      setLeds(true, null, null);
    }
  }

  async function stopRecording() {
    if (!isRecording) return;
    isRecording = false;
    clearTimeout(autoStopTimer);
    micBtn.classList.remove('recording');
    statusText.textContent = '⚡ Processing on Raspberry Pi...';

    const srcRate = audioCtx ? audioCtx.sampleRate : 48000;
    if (processor) processor.disconnect();
    if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
    if (audioCtx) await audioCtx.close();

    const wavBlob = encodeWAV(recordedChunks, srcRate, 16000);
    await uploadAudio(wavBlob);
  }

  micBtn.addEventListener('click', () => {
    if (isRecording) stopRecording();
    else startRecording();
  });

  function encodeWAV(buffers, srcRate, targetRate) {
    let totalLen = buffers.reduce((acc, b) => acc + b.length, 0);
    let flat = new Float32Array(totalLen);
    let offset = 0;
    for (let b of buffers) { flat.set(b, offset); offset += b.length; }

    // Downsample to 16000 Hz
    const ratio = srcRate / targetRate;
    const outLen = Math.floor(flat.length / ratio);
    const pcm16 = new Int16Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const pos = i * ratio;
      const idx = Math.floor(pos);
      const frac = pos - idx;
      const s = (1 - frac) * flat[idx] + frac * (flat[idx + 1] || flat[idx]);
      const clamped = Math.max(-1, Math.min(1, s));
      pcm16[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7FFF;
    }

    const buffer = new ArrayBuffer(44 + pcm16.length * 2);
    const view = new DataView(buffer);
    const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };

    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + pcm16.length * 2, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, targetRate, true);
    view.setUint32(28, targetRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, 'data');
    view.setUint32(40, pcm16.length * 2, true);

    let byteOff = 44;
    for (let i = 0; i < pcm16.length; i++, byteOff += 2) {
      view.setInt16(byteOff, pcm16[i], true);
    }
    return new Blob([buffer], { type: 'audio/wav' });
  }

  async function uploadAudio(wavBlob) {
    try {
      const res = await fetch('/audio', {
        method: 'POST',
        headers: { 'Content-Type': 'audio/wav' },
        body: wavBlob
      });
      const data = await res.json();
      const heard = data.transcript || '(No decipherable speech)';
      transcriptBox.textContent = `"${heard}" ➔ ${data.command}`;
      statusText.textContent = 'Tap to Speak';
      refreshStatus();
    } catch (e) {
      statusText.textContent = 'Network Error: Cannot reach Pi';
      setLeds(true, null, null);
    }
  }
</script>
</body>
</html>
"""


class VoiceCommandServer:
    WAKE_PHRASES = (
        "hey vosk", "hey boss", "hey box", "hey mosque", "hey lost",
        "hello vosk", "hello boss", "hey light", "hello light", "hello",
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

        SetLogLevel(-1)
        self.max_audio_bytes = max_audio_bytes
        self.listen_timeout = listen_timeout
        self.use_grammar = use_grammar
        self._grammar_json = json.dumps(self.GRAMMAR_VOCAB)

        print("Loading Vosk model into memory...")
        self.model = Model(model_path)
        self._recognizer_type = KaldiRecognizer

        self.light = LED(light_pin, active_high=not active_low)
        self.yellow_led = LED(yellow_pin)
        self.red_led = LED(red_pin)
        self.light.off()
        self.yellow_led.off()
        self.red_led.off()

        self._lock = threading.Lock()
        self.is_awake = False
        self._timeout_timer: Optional[threading.Timer] = None

    def get_status(self) -> dict:
        return {
            "green": self.light.is_lit,
            "yellow": self.yellow_led.is_lit,
            "red": self.red_led.is_lit,
            "awake": self.is_awake,
        }

    def set_manual(self, action: str) -> dict:
        with self._lock:
            self._sleep_standby()
            self.red_led.off()
            if action == "ON":
                self.light.on()
            elif action == "OFF":
                self.light.off()
            return {
                "command": action,
                **self.get_status(),
            }

    def _has_wake_word(self, text: str) -> bool:
        return any(wake in text for wake in self.WAKE_PHRASES)

    def _cancel_timer(self) -> None:
        if self._timeout_timer is not None:
            self._timeout_timer.cancel()
            self._timeout_timer = None

    def _on_wake_timeout(self) -> None:
        with self._lock:
            if self.is_awake:
                self.is_awake = False
                self.yellow_led.off()
                print("[TIMEOUT] Wake window expired. Returning to standby.", flush=True)

    def _activate_wake_window(self) -> None:
        self._cancel_timer()
        self.is_awake = True
        self.yellow_led.on()
        self.red_led.off()
        self._timeout_timer = threading.Timer(self.listen_timeout, self._on_wake_timeout)
        self._timeout_timer.daemon = True
        self._timeout_timer.start()

    def _sleep_standby(self) -> None:
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
        transcript = " ".join(raw_transcript.split())

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

            if not self.is_awake:
                return {
                    "transcript": transcript,
                    "command": "IGNORED (SAY 'HEY VOSK' FIRST)",
                }

            command = check_light_command(transcript)
            if command == "ON":
                self.light.on()
                self._sleep_standby()
                return {"transcript": transcript, "command": "ON"}

            if command == "OFF":
                self.light.off()
                self._sleep_standby()
                return {"transcript": transcript, "command": "OFF"}

            if woke_this_turn:
                return {
                    "transcript": transcript,
                    "command": f"AWAKE (LISTENING FOR {self.listen_timeout:.0f}s...)",
                }

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
        def _send_json(self, status_code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/" or self.path.startswith("/index"):
                body = MOBILE_APP_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/status":
                self._send_json(200, controller.get_status())
            else:
                self.send_error(404, "Not Found")

        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length") or "0")
            if content_length <= 0 or content_length > controller.max_audio_bytes:
                self.send_error(413, "Invalid or oversized payload")
                return

            raw_body = self.rfile.read(content_length)

            if self.path == "/manual":
                try:
                    data = json.loads(raw_body.decode("utf-8"))
                    action = data.get("action", "")
                    if action not in ("ON", "OFF"):
                        raise ValueError("action must be ON or OFF")
                    result = controller.set_manual(action)
                    print(f"[MANUAL TOUCH] Light turned {action}", flush=True)
                    self._send_json(200, result)
                except Exception as error:
                    self.send_error(400, str(error))
                return

            if self.path == "/audio":
                try:
                    result = controller.process_wav(raw_body)
                except (OSError, ValueError, wave.Error) as error:
                    controller.red_led.on()
                    self.send_error(400, str(error))
                    return

                self._send_json(200, result)
                print(
                    f'[VOICE] "{result["transcript"]}" -> {result["command"]}',
                    flush=True,
                )
                return

            self.send_error(404, "Use POST /audio or POST /manual")

        def log_message(self, format: str, *args) -> None:
            pass

    return RequestHandler


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Pi Voice Server with built-in Mobile Web App and PC support."
    )
    parser.add_argument(
        "--model",
        default="/home/pi/Vosk_Project/vosk-model-small-en-us-0.15",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--light-pin", type=int, default=22)
    parser.add_argument("--yellow-pin", type=int, default=27)
    parser.add_argument("--red-pin", type=int, default=17)
    parser.add_argument("--timeout", type=float, default=6.0)
    parser.add_argument("--active-low", action="store_true")
    parser.add_argument("--no-grammar", action="store_true")
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
        print(f"  MOBILE APP URL : http://<PI_IP_ADDRESS>:{arguments.port}/")
        print(f"  PC CLIENT URL  : http://<PI_IP_ADDRESS>:{arguments.port}/audio")
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