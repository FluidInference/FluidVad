/**
 * @fluidinference/fluidvad — environment-aware entry point.
 *
 * The wasm-pack "web" build expects `fetch(new URL(...))`, which Node cannot
 * do for file:// URLs; in Node we read the wasm from disk instead.
 */
import init, { initSync, FluidVad, VadOptions } from "./dist/fluidvad.js";

let ready = null;

/** Initialize the wasm module once. Safe to call multiple times. */
export async function load() {
  if (!ready) {
    ready = (async () => {
      const isNode =
        typeof process !== "undefined" && process.versions && process.versions.node;
      if (isNode) {
        const { readFile } = await import("node:fs/promises");
        const { fileURLToPath } = await import("node:url");
        const wasmPath = fileURLToPath(new URL("./dist/fluidvad_bg.wasm", import.meta.url));
        initSync({ module: await readFile(wasmPath) });
      } else {
        await init();
      }
    })();
  }
  await ready;
}

/** Convenience: load the wasm and construct a FluidVad in one call. */
export async function createVad(options) {
  await load();
  if (options === undefined) return new FluidVad();
  const o = new VadOptions();
  if (options.threshold !== undefined) o.threshold = options.threshold;
  if (options.minSpeechDuration !== undefined) o.minSpeechDuration = options.minSpeechDuration;
  if (options.minSilenceDuration !== undefined) o.minSilenceDuration = options.minSilenceDuration;
  if (options.maxSpeechDuration !== undefined) o.maxSpeechDuration = options.maxSpeechDuration;
  if (options.speechPadding !== undefined) o.speechPadding = options.speechPadding;
  if (options.negativeThreshold !== undefined) o.negativeThreshold = options.negativeThreshold;
  return new FluidVad(o);
}

export { FluidVad, VadOptions };
