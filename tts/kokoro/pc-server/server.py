"""Kokoro TTS HTTP server (CPU, optional CUDA) for the benchmark rig.

Three ONNX engines in one process:
  stock   - stock Kokoro v1.0 (voices such as am_adam; Russian via espeak-ng phonemes);
  ru-base - Russian fine-tune zaakirio/kokoro-ru (voices sveta, masha);
  ru-dima - same fine-tune, separate checkpoint (voice dima).

Listens on 127.0.0.1 only (no authentication - do not expose it).
  GET  /health            -> {"ok":true,...}
  POST /tts?voice=NAME    -> body = UTF-8 text, response = WAV (24 kHz)

Requirements: pip install kokoro-onnx onnxruntime(-gpu) soundfile numpy
Files expected in $KOKORO_DIR (default ./models/kokoro):
  kokoro-v1.0.onnx, voices.bin   (kokoro-onnx release assets)
  ru-base.onnx, ru-dima.onnx     (ONNX exports of kokoro-ru)
  sveta.bin, masha.bin, dima.bin (float32 style packs, shape [510,1,256])
Env: KOKORO_DIR, KOKORO_PORT (11400), KOKORO_THREADS (16), KOKORO_GPU_MEM_GB (2),
     KOKORO_DEFAULT_VOICE (am_adam), CUDA_DLL_DIRS (os.pathsep-separated extra
     directories with cuDNN/cuBLAS DLLs, prepended to PATH).
Run:  python3 server.py
"""
import io
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

_extra = os.environ.get("CUDA_DLL_DIRS", "")
if _extra:
    os.environ["PATH"] = _extra + os.pathsep + os.environ.get("PATH", "")

import numpy as np  # noqa: E402
import onnxruntime as rt  # noqa: E402
import soundfile as sf  # noqa: E402
from kokoro_onnx import Kokoro  # noqa: E402

DIR = os.environ.get("KOKORO_DIR", "./models/kokoro")
THREADS = int(os.environ.get("KOKORO_THREADS", "16"))
PORT = int(os.environ.get("KOKORO_PORT", "11400"))
GPU_MEM_LIMIT = int(float(os.environ.get("KOKORO_GPU_MEM_GB", "2")) * 1024**3)
DEFAULT_VOICE = os.environ.get("KOKORO_DEFAULT_VOICE", "am_adam")

STOCK = os.path.join(DIR, "kokoro-v1.0.onnx")
RU_BASE = os.path.join(DIR, "ru-base.onnx")
RU_DIMA = os.path.join(DIR, "ru-dima.onnx")
VOICES = os.path.join(DIR, "voices.bin")   # placeholder table; custom styles are passed as arrays
PACKS = {name: os.path.join(DIR, f"{name}.bin") for name in ("sveta", "masha", "dima")}


def make(model_path: str):
    """Build a Kokoro engine on CUDA with a memory cap; fall back to CPU on any failure."""
    try:
        so = rt.SessionOptions()
        opts = {"device_id": 0, "gpu_mem_limit": GPU_MEM_LIMIT,
                "arena_extend_strategy": "kSameAsRequested"}
        sess = rt.InferenceSession(model_path, so, providers=[
            ("CUDAExecutionProvider", opts), "CPUExecutionProvider"])
        prov = sess.get_providers()[0]
        print("%s -> %s" % (os.path.basename(model_path), prov))
        if prov != "CUDAExecutionProvider":
            raise RuntimeError("CUDA execution provider not active")
        return Kokoro.from_session(sess, VOICES), "gpu"
    except Exception as e:  # noqa: BLE001
        print("%s: GPU unavailable (%s) - using CPU" % (os.path.basename(model_path), e))
    so = rt.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.inter_op_num_threads = 1
    sess = rt.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    return Kokoro.from_session(sess, VOICES), "cpu"


def pack(path: str):
    return np.fromfile(path, dtype=np.float32).reshape(510, 1, 256)


t0 = time.time()
stock, where_stock = make(STOCK)
ru_base, where_base = make(RU_BASE)
ru_dima, where_dima = make(RU_DIMA)
print("models loaded in %.1fs (stock=%s base=%s dima=%s)"
      % (time.time() - t0, where_stock, where_base, where_dima))

_lock = threading.Lock()   # espeak is not thread-safe: serialize all synthesis


def synth(text: str, voice: str) -> bytes:
    if voice in ("sveta", "masha"):
        eng, style = ru_base, pack(PACKS[voice])
    elif voice == "dima":
        eng, style = ru_dima, pack(PACKS["dima"])
    else:
        eng, style = stock, voice            # stock voices by name (am_adam, ...)
    with _lock:
        audio, sr = eng.create(text, voice=style, lang="ru", speed=1.0)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"   # keep-alive: avoid paying connection setup per request

    def log_message(self, *a):
        pass

    def _send(self, code, body=b"", ctype="application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            return self._send(200, b'{"ok":true}', "application/json")
        self._send(404)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/tts":
            return self._send(404)
        q = parse_qs(u.query)
        n = int(self.headers.get("Content-Length") or 0)
        text = self.rfile.read(n).decode("utf-8", "replace") if n else ""
        if not text.strip():
            return self._send(400)
        voice = (q.get("voice") or [DEFAULT_VOICE])[0]
        try:
            return self._send(200, synth(text, voice), "audio/wav")
        except Exception as e:  # noqa: BLE001
            return self._send(500, str(e).encode("utf-8", "replace"), "text/plain")


if __name__ == "__main__":
    # Warm up every engine: the first call is several times slower than later ones.
    synth("Прогрев.", DEFAULT_VOICE)
    synth("Прогрев.", "sveta")
    synth("Прогрев.", "dima")
    print("kokoro ready, listening on 127.0.0.1:%d" % PORT)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
