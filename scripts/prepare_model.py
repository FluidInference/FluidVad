#!/usr/bin/env python3
"""Prepare the embedded FluidVad model from upstream silero-vad.

1. Download silero_vad_16k_op15.onnx (16 kHz-only, opset 15).
2. Bake `sr` = 16000 as an initializer and fix input shapes
   (input [1, 576] = 64-sample context + 512 new samples, state [2, 1, 128]).
3. Constant-fold with onnxruntime's basic optimizer — this eliminates every
   `If` node, which pure-Rust runtimes (tract) cannot type-check because the
   upstream branches disagree on rank (e.g. [1,128] vs [1,128,1]).
4. Verify bit-exact parity against the unmodified upstream model.

Output: models/silero_vad_16k_v6.onnx + models/parity_ref.txt
"""
import os
import urllib.request

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

UPSTREAM = "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad_16k_op15.onnx"
CONTEXT = 64
FRAME = 512
MODELS = os.path.join(os.path.dirname(__file__), "..", "models")

src = os.path.join(MODELS, "silero_vad_16k_op15.onnx")
out = os.path.join(MODELS, "silero_vad_16k_v6.onnx")

if not os.path.exists(src):
    print(f"downloading {UPSTREAM}")
    # download to a temp path and rename so an interrupted download never
    # leaves a truncated file that the exists() check would trust forever
    tmp = src + ".part"
    urllib.request.urlretrieve(UPSTREAM, tmp)
    os.replace(tmp, src)

m = onnx.load(src)
g = m.graph
if not any(i.name == "sr" for i in g.initializer):
    g.initializer.append(numpy_helper.from_array(np.array(16000, dtype=np.int64), name="sr"))
inputs = [i for i in g.input if i.name != "sr"]
del g.input[:]
g.input.extend(inputs)
for i in g.input:
    dims = i.type.tensor_type.shape.dim
    if i.name == "input":
        dims[0].ClearField("dim_param"); dims[0].dim_value = 1
        dims[1].ClearField("dim_param"); dims[1].dim_value = CONTEXT + FRAME
    elif i.name == "state":
        dims[1].ClearField("dim_param"); dims[1].dim_value = 1
baked = os.path.join(MODELS, "_sr_baked.onnx")
onnx.save(m, baked)

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
so.optimized_model_filepath = out
ort.InferenceSession(baked, so, providers=["CPUExecutionProvider"])
os.remove(baked)

m2 = onnx.load(out)
ifs = [n.name for n in m2.graph.node if n.op_type == "If"]
assert not ifs, f"If nodes survived optimization: {ifs}"
print("ops:", sorted({n.op_type for n in m2.graph.node}))

# Parity: fixed model vs unmodified upstream, chained state over 20 frames
rng = np.random.RandomState(42)
orig = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
fixed = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
sr = np.array(16000, dtype=np.int64)
s1 = s2 = np.zeros((2, 1, 128), dtype=np.float32)
first = None
for i in range(20):
    x = (rng.randn(1, CONTEXT + FRAME) * 0.1).astype(np.float32)
    if i == 0:
        # zero context on the first frame so the Rust wrapper (fresh state)
        # can reproduce this input exactly from the 512 new samples
        x[:, :CONTEXT] = 0.0
    r1 = orig.run(None, {"input": x, "state": s1, "sr": sr})
    r2 = fixed.run(None, {"input": x, "state": s2})
    if first is None:
        first = (x, r2[0].ravel()[0])
    assert np.abs(r1[0] - r2[0]).max() == 0.0, "prob mismatch"
    assert np.abs(r1[1] - r2[1]).max() == 0.0, "state mismatch"
    s1, s2 = r1[1], r2[1]
print("parity vs upstream: bit-exact over 20 chained frames")

with open(os.path.join(MODELS, "parity_ref.txt"), "w") as f:
    f.write(" ".join(f"{v:.9g}" for v in first[0].ravel()) + "\n")
    f.write(f"{first[1]:.9g}\n")
print(f"wrote {out} ({os.path.getsize(out)} bytes) + parity_ref.txt (prob={first[1]:.9g})")
