/**
 * Renderer (contextIsolation on, no nodeIntegration).
 * The wasm bytes come from the preload bridge; ESM comes from node_modules
 * via relative file:// imports (Electron ≥28 supports renderer ESM).
 */
import { createVad } from "./node_modules/@fluidinference/fluidvad/index.js";
import { MicVad } from "./node_modules/@fluidinference/fluidvad/mic.js";

const status = document.getElementById("status");
const bar = document.getElementById("bar");
const log = document.getElementById("log");
const say = (line) => (log.textContent = `${line}\n${log.textContent}`.slice(0, 4000));

const loadOptions = { wasm: window.fluidvad.wasmBytes };

if (window.fluidvad.isSmoke) {
  // headless verification: wasm loads + inference runs in this renderer
  try {
    const vad = await createVad(undefined, loadOptions);
    const events = vad.push(new Float32Array(32000)); // 2 s of silence
    if (vad.isSpeaking) throw new Error("silence must not trigger speech");
    window.fluidvad.reportSmoke({ ok: true, frames: 32000 / 512, isSpeaking: vad.isSpeaking, events: events.length });
  } catch (e) {
    window.fluidvad.reportSmoke({ ok: false, error: String(e && e.stack ? e.stack : e) });
  }
} else {
  // live microphone
  const mic = new MicVad({
    load: loadOptions,
    onSpeechStart: (t) => {
      status.textContent = "speaking";
      status.className = "speaking";
      say(`speech start @ ${t.toFixed(2)}s`);
    },
    onSpeechEnd: (audio, start, end) => {
      status.textContent = "listening";
      status.className = "";
      say(`utterance ${start.toFixed(2)}s → ${end.toFixed(2)}s (${(audio.length / 16000).toFixed(2)}s audio)`);
    },
    onFrame: (p) => (bar.style.width = `${Math.round(p * 100)}%`),
  });

  try {
    await mic.start();
    status.textContent = "listening";
    say("mic started — speak!");
  } catch (e) {
    status.textContent = `mic error: ${e.message}`;
    say(String(e));
  }
}
