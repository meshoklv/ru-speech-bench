"""Check that onnxruntime picks the CUDA execution provider for a Kokoro model.

Loads a Kokoro ONNX model with CUDA (falling back to CPU), then synthesizes one
Russian sentence three times and prints the wall time of each run.

Usage:
    KOKORO_DIR=./models/kokoro python3 gpu_probe.py
Needs: kokoro-onnx, onnxruntime-gpu (matching your CUDA major version), numpy.
"""
import os
import time

import numpy as np
import onnxruntime as rt
from kokoro_onnx import Kokoro

DIR = os.environ.get("KOKORO_DIR", "./models/kokoro")

opts = {"device_id": 0, "gpu_mem_limit": 2 * 1024**3, "arena_extend_strategy": "kSameAsRequested"}
t0 = time.time()
so = rt.SessionOptions()
sess = rt.InferenceSession(os.path.join(DIR, "ru-base.onnx"), so,
                           providers=[("CUDAExecutionProvider", opts), "CPUExecutionProvider"])
print("active provider:", sess.get_providers()[0], "| session %.1fs" % (time.time() - t0))
k = Kokoro.from_session(sess, os.path.join(DIR, "voices.bin"))
style = np.fromfile(os.path.join(DIR, "sveta.bin"), dtype=np.float32).reshape(510, 1, 256)
text = "Алло, добрый день! Меня зовут Диана, я звоню из клиники. Удобно сейчас говорить?"
for i in range(3):
    t0 = time.time()
    a, sr = k.create(text, voice=style, lang="ru")
    print("run %d: %.2fs (%.1fs of audio)" % (i + 1, time.time() - t0, len(a) / sr))
