#!/usr/bin/env python3
"""Generate testdata/ fixtures for the real-audio integration tests.

Usage: python3 scripts/make_testdata.py <path-to-16k-mono-wav>

Copies the wav to testdata/speech_16k.wav and computes per-frame reference
probabilities with onnxruntime on the prepared model, matching the Rust
pipeline exactly: 512-sample frames with 64-sample carried context, and a
zero-padded trailing partial frame.

testdata/ is git-ignored: use any real 16 kHz mono speech recording you have
the rights to. The tests skip (loudly) when the fixtures are absent.
"""
import os
import shutil
import sys
import wave

import numpy as np
import onnxruntime as ort

CONTEXT, FRAME = 64, 512
ROOT = os.path.join(os.path.dirname(__file__), "..")
TESTDATA = os.path.join(ROOT, "testdata")

if len(sys.argv) != 2:
    sys.exit(__doc__)

src = sys.argv[1]
with wave.open(src, "rb") as w:
    assert w.getframerate() == 16000, f"need 16 kHz, got {w.getframerate()}"
    assert w.getnchannels() == 1, f"need mono, got {w.getnchannels()} channels"
    assert w.getsampwidth() == 2, "need 16-bit PCM"
    raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
x = (raw / 32768.0).astype(np.float32)

os.makedirs(TESTDATA, exist_ok=True)
dst = os.path.join(TESTDATA, "speech_16k.wav")
if not os.path.exists(dst) or not os.path.samefile(src, dst):
    shutil.copy(src, dst)

sess = ort.InferenceSession(
    os.path.join(ROOT, "models", "silero_vad_16k_v6.onnx"),
    providers=["CPUExecutionProvider"],
)
state = np.zeros((2, 1, 128), dtype=np.float32)
ctx = np.zeros(CONTEXT, dtype=np.float32)
probs = []
for i in range(0, len(x), FRAME):
    frame = x[i : i + FRAME]
    if len(frame) < FRAME:  # zero-pad the trailing partial frame, as the Rust side does
        frame = np.pad(frame, (0, FRAME - len(frame)))
    out = sess.run(None, {"input": np.concatenate([ctx, frame])[None, :], "state": state})
    probs.append(float(out[0].ravel()[0]))
    state = out[1]
    ctx = frame[-CONTEXT:]

np.savetxt(os.path.join(TESTDATA, "speech_16k_ort_probs.txt"), probs, fmt="%.9g")
p = np.array(probs)
print(f"wrote {len(x)} samples, {len(p)} frame probs (speech>=0.5: {(p >= 0.5).sum()}, mean {p.mean():.3f})")
