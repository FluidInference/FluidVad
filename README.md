# FluidVad

[![npm](https://img.shields.io/npm/v/@fluidinference/fluidvad.svg)](https://www.npmjs.com/package/@fluidinference/fluidvad)

**Voice activity detection for npm — Silero VAD (v6) compiled to WebAssembly.
Model bundled, zero config, no runtime downloads.** Works in the browser,
Node, and Electron (both processes) on macOS and Windows.

- **No native modules** — pure wasm. Nothing to `electron-rebuild`, no
  per-arch prebuilds, no onnxruntime peer dependency, no extra binaries to sign.
- **Model embedded** — the 1.3 MB Silero v6 16 kHz graph ships inside the
  `.wasm` (5.3 MB raw, 2.3 MB gzipped total). `npm install` and go; nothing
  fetched at runtime, fully offline.
- **Streaming + offline** — `SpeechStart`/`SpeechEnd` events with hysteresis,
  or whole-buffer segmentation. ~150× real-time; a 32 ms frame costs well
  under a millisecond.

```bash
npm i @fluidinference/fluidvad
pnpm add @fluidinference/fluidvad
bun add @fluidinference/fluidvad
```

📦 [`@fluidinference/fluidvad` on npm](https://www.npmjs.com/package/@fluidinference/fluidvad)

## Microphone (browser / Electron renderer)

```js
import { MicVad } from "@fluidinference/fluidvad/mic";

const mic = new MicVad({
  onSpeechStart: (t) => console.log("speech started", t),
  onSpeechEnd: (audio, start, end) => {
    // audio: Float32Array, 16 kHz mono, whole utterance incl. pre-roll
    console.log(`utterance ${start.toFixed(2)}s → ${end.toFixed(2)}s`);
  },
});
await mic.start();
```

## Buffers (Node / Electron main / browser)

```js
import { createVad } from "@fluidinference/fluidvad";

const vad = await createVad({ threshold: 0.5 });

// streaming: push any chunk size, get boundary events
const events = vad.push(samples); // Float32Array, 16 kHz mono
// [{ isStart: true, sampleIndex: 15872, timeSeconds: 0.99 }, ...]

// offline: segment a whole buffer
const segments = vad.segment(samples);
// [{ startTime: 0.9, endTime: 4.21 }, ...]
```

Input is always **16 kHz mono f32** in `[-1, 1]`. The model consumes
512-sample frames (32 ms); `push` buffers partial frames internally.

## Electron

Runnable example in [`examples/electron`](examples/electron) (mic UI +
headless smoke mode, CI-tested on macOS and Windows).

- **Main / preload (Node env):** `createVad()` works as-is; the wasm is read
  from disk (asar-transparent).
- **Renderer with `contextIsolation`:** the renderer cannot `fetch()` `file://`
  URLs, so hand the wasm bytes over from the preload:

```js
// preload.cjs
const wasmPath = require.resolve("@fluidinference/fluidvad/dist/fluidvad_bg.wasm");
contextBridge.exposeInMainWorld("fluidvad", { wasmBytes: new Uint8Array(fs.readFileSync(wasmPath)) });

// renderer
const mic = new MicVad({ load: { wasm: window.fluidvad.wasmBytes }, onSpeechEnd: ... });
```

- CSP: add `'wasm-unsafe-eval'` to `script-src` (compiles wasm without enabling JS `eval`).
- macOS mic: call `systemPreferences.askForMediaAccess("microphone")` from main and
  set `NSMicrophoneUsageDescription` when packaging.

## Fidelity

Benchmarked with the same protocol as FluidAudio's `vad-benchmark`
(per-chunk probability >= threshold, clip = speech at >=10% active chunks;
`examples/vad_benchmark.rs`). Apple M-series, native CPU, single thread.

**musan_mini50** (25 LibriVox speech + 25 MUSAN noise clips):

| Backend | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| FluidAudio silero (CoreML) | 82.0% | 73.5% | 100% | 84.7% |
| FluidAudio FSMN-VAD | 98.0% | 96.2% | 100% | 98.0% |
| **FluidVad (Silero v6, tract)** | **98.0%** | **96.2%** | **100%** | **98.0%** |

**Full MUSAN** (426 speech + 930 noise clips, 66.7 h):

| Backend | Noise rejected | False-positive rate | Speech recall |
|---|---|---|---|
| FluidAudio silero (CoreML) | 69.8% | 30.2% | — |
| FluidAudio FSMN-VAD | 81.9% | 18.1% | — |
| **FluidVad (Silero v6, tract)** | **94.8%** | **5.2%** | **100%** |

~327x real-time native, ~150x in wasm (Node). FluidAudio numbers from
[FluidAudio#653](https://github.com/FluidInference/FluidAudio/pull/653)
(its MUSAN noise run used 774 clips vs 930 here).

## Configuration

| Option | Default | Meaning |
|---|---|---|
| `threshold` | 0.5 | entry threshold (frame is speech at ≥) |
| `negativeThreshold` | `threshold - 0.15` | exit threshold (hysteresis) |
| `minSpeechDuration` | 0.15 s | drop shorter speech runs |
| `minSilenceDuration` | 0.75 s | silence needed to close a segment |
| `maxSpeechDuration` | 14 s | force-split longer segments at the best silence |
| `speechPadding` | 0.1 s | padding around each segment |

## Development

The wasm is built from a Rust core (`src/`) using [tract](https://github.com/sonos/tract)
for CPU inference — no onnxruntime anywhere.

Upstream Silero ONNX contains `If` nodes whose branches disagree on rank —
onnxruntime broadcasts through it, strict runtimes cannot. We bake
`sr = 16000` as a constant, fix the input shapes, and constant-fold with
onnxruntime's basic optimizer, which eliminates every `If`
(`scripts/prepare_model.py`, **bit-exact** with upstream). The result is
pre-compiled to NNEF (`examples/export_nnef.rs`) so the shipped wasm only
carries tract's lightweight loader. Per-frame parity vs onnxruntime is
asserted in tests (`tests/model_parity.rs`); the hysteresis / segmentation
state machines (adapted from [FluidAudio](https://github.com/FluidInference/FluidAudio))
are unit-tested with synthetic probability sequences.

```bash
cargo test --release              # core + parity tests
./scripts/build_npm.sh            # build the npm package into npm/
python3 scripts/prepare_model.py  # regenerate model artifacts (needs onnx, onnxruntime)
cd examples/electron && npm i && FLUIDVAD_SMOKE=1 npx electron .   # headless check
```

## License

MIT. The bundled Silero VAD model is © Silero Team, MIT-licensed
([SILERO_LICENSE](SILERO_LICENSE), [snakers4/silero-vad](https://github.com/snakers4/silero-vad)).
