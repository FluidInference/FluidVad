/**
 * @fluidinference/fluidvad — environment-aware entry point.
 *
 * Resolution order for the wasm binary:
 *   1. explicit `wasm` passed to load()/createVad() — bytes, a Response, or a
 *      compiled WebAssembly.Module. The escape hatch for Electron renderers
 *      (contextIsolation) and unusual bundlers: read the file however you
 *      like and hand it over.
 *   2. Node (incl. Electron main/preload): read from disk next to this module
 *      (asar-transparent in Electron).
 *   3. Browser: `fetch(new URL('./dist/fluidvad_bg.wasm', import.meta.url))`
 *      — bundlers (Vite/webpack) rewrite this to an emitted asset.
 */
import init, { initSync, FluidVad, VadOptions } from "./dist/fluidvad.js";

let ready = null;

/**
 * Initialize the wasm module once. Safe to call multiple times; the first
 * call wins (a differing `wasm` argument on a later call is ignored).
 *
 * @param {object} [options]
 * @param {BufferSource | WebAssembly.Module | Response | Promise<Response>} [options.wasm]
 *   Explicit wasm source, bypassing environment detection.
 */
export async function load(options) {
  if (!ready) {
    ready = (async () => {
      const explicit = options && options.wasm;
      if (explicit) {
        // init() accepts bytes, Response, Request, URL, or WebAssembly.Module
        await init({ module_or_path: await explicit });
        return;
      }
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

/**
 * Convenience: load the wasm and construct a FluidVad in one call.
 *
 * @param {import('./index.d.ts').VadConfig} [options]
 * @param {Parameters<typeof load>[0]} [loadOptions] e.g. `{ wasm: bytes }`.
 */
export async function createVad(options, loadOptions) {
  await load(loadOptions);
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
