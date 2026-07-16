# FluidVad

Silero VAD (v6) in pure Rust — **model bundled, zero config, no runtime downloads**.
One core, two targets: native ([crates.io](https://crates.io/crates/fluidvad)) and
WebAssembly ([npm](https://www.npmjs.com/package/@fluidinference/fluidvad)).

- **CPU-only, pure Rust** — ONNX inference via [tract](https://github.com/sonos/tract);
  no onnxruntime, no C++ linkage, no platform-specific binaries.
- **Model embedded** — the 1.3 MB Silero v6 16 kHz graph ships inside the
  binary/`.wasm`. `npm install` / `cargo add` and go; nothing fetched at runtime.
- **Streaming + offline** — `SpeechStart`/`SpeechEnd` events with hysteresis, or
  whole-buffer segmentation. State machine behavior matches
  [FluidAudio](https://github.com/FluidInference/FluidAudio)'s Swift `VadManager`.
- **Fast enough everywhere** — ~150× real-time in wasm (Node, Apple M-series),
  faster native. A 32 ms frame costs well under a millisecond.

## npm (browser + Node)

```bash
npm i @fluidinference/fluidvad
```

Microphone with callbacks (browser):

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

Buffers (browser or Node):

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

## Electron

Works in both processes with no native modules — nothing to `electron-rebuild`,
no per-arch prebuilds, no extra binaries to sign. Runnable example in
[`examples/electron`](examples/electron) (mic UI + headless smoke mode, CI-tested
on macOS and Windows).

- **Main / preload (Node env):** `import { createVad } from "@fluidinference/fluidvad"`
  works as-is; the wasm is read from disk (asar-transparent).
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

## Rust

```bash
cargo add fluidvad
```

```rust
use fluidvad::{SileroModel, VadStreamer, VadSegmentationConfig, segment_speech};

// offline
let model = SileroModel::new()?;
let segments = segment_speech(&model, &samples, &VadSegmentationConfig::default())?;

// streaming
let mut streamer = VadStreamer::with_model(model, VadSegmentationConfig::default());
for result in streamer.push(&chunk)? {
    if let Some(event) = result.event {
        println!("{:?} at {:.2}s", event.kind, event.time_seconds());
    }
}
```

Input is always **16 kHz mono f32** in `[-1, 1]`. The model consumes
512-sample frames (32 ms); `push` buffers partial frames internally.

## Configuration

| Option | Default | Meaning |
|---|---|---|
| `threshold` | 0.5 | entry threshold (frame is speech at ≥) |
| `negativeThreshold` | `threshold - 0.15` | exit threshold (hysteresis) |
| `minSpeechDuration` | 0.15 s | drop shorter speech runs |
| `minSilenceDuration` | 0.75 s | silence needed to close a segment |
| `maxSpeechDuration` | 14 s | force-split longer segments at the best silence |
| `speechPadding` | 0.1 s | padding around each segment |

## How the model is prepared

Upstream `silero_vad_16k_op15.onnx` contains `If` nodes whose branches disagree
on rank — onnxruntime broadcasts through it, strict runtimes cannot. We bake
`sr = 16000` as a constant, fix the input shapes, and constant-fold with
onnxruntime's basic optimizer, which eliminates every `If`
(`scripts/prepare_model.py`, **bit-exact** with upstream). The result is
pre-compiled to NNEF (`examples/export_nnef.rs`) so the shipped library only
carries tract's lightweight loader. Per-frame parity vs onnxruntime is
asserted in CI-runnable tests (`tests/model_parity.rs`).

## Building

```bash
cargo test --release            # native tests
./scripts/build_npm.sh          # wasm + npm package into npm/
python3 scripts/prepare_model.py  # regenerate the model artifacts (needs onnx, onnxruntime)
```

## License

MIT. The bundled Silero VAD model is © Silero Team, MIT-licensed
([SILERO_LICENSE](SILERO_LICENSE), [snakers4/silero-vad](https://github.com/snakers4/silero-vad)).
