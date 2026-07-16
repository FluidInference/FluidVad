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
import init, { initSync, FluidVad, VadOptions, VadEvent, Segment } from "./dist/fluidvad.js";

let ready = null;

async function initialize(options) {
  const explicit = options && options.wasm;
  if (explicit) {
    // init() accepts bytes, Response, Request, URL, or WebAssembly.Module
    await init({ module_or_path: await explicit });
    return;
  }
  const isNode = typeof process !== "undefined" && process.versions && process.versions.node;
  if (isNode) {
    const { readFile } = await import("node:fs/promises");
    const { fileURLToPath } = await import("node:url");
    const wasmPath = fileURLToPath(new URL("./dist/fluidvad_bg.wasm", import.meta.url));
    initSync({ module: await readFile(wasmPath) });
  } else {
    await init();
  }
}

/**
 * Initialize the wasm module once. Safe to call multiple times; the first
 * successful call wins (a differing `wasm` argument on a later call is
 * ignored). A failed attempt is not cached — calling again retries.
 *
 * @param {object} [options]
 * @param {BufferSource | WebAssembly.Module | Response | Promise<Response>} [options.wasm]
 *   Explicit wasm source, bypassing environment detection.
 */
export async function load(options) {
  if (!ready) {
    ready = initialize(options).catch((e) => {
      ready = null; // don't brick future calls on a transient failure
      throw e;
    });
  }
  await ready;
}

/** VadConfig keys forwarded to the wasm VadOptions (same names on both sides). */
const CONFIG_KEYS = [
  "threshold",
  "minSpeechDuration",
  "minSilenceDuration",
  "maxSpeechDuration",
  "speechPadding",
  "negativeThreshold",
];

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
  for (const key of CONFIG_KEYS) {
    if (options[key] !== undefined) o[key] = options[key];
  }
  return new FluidVad(o);
}

export { FluidVad, VadOptions, VadEvent, Segment };
