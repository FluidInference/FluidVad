# Test fixtures

- `speech_16k.wav` — 11 s excerpt of John F. Kennedy's 1961 inaugural address
  ("ask not what your country can do for you…"), 16 kHz mono 16-bit PCM.
  A work of the United States government — **public domain**. The same clip is
  used as a sample by [whisper.cpp](https://github.com/ggml-org/whisper.cpp).
- `speech_16k_ort_probs.txt` — per-frame reference probabilities computed by
  onnxruntime on the prepared model; regenerate with
  `python3 scripts/make_testdata.py testdata/speech_16k.wav`.
