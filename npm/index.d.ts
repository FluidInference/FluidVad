export { FluidVad, VadOptions, VadEvent, Segment } from "./dist/fluidvad.js";

/**
 * User-facing configuration; all fields optional, defaults match Silero.
 * Out-of-range values (NaN, negatives, thresholds outside [0, 1]) make the
 * FluidVad constructor throw.
 */
export interface VadConfig {
  /** Entry threshold: a frame is speech at probability ≥ this (default 0.5). */
  threshold?: number;
  /**
   * Minimum speech run to keep, seconds (default 0.15).
   * Applies to `segment()` only — streaming `push()` events fire immediately.
   */
  minSpeechDuration?: number;
  /** Silence needed to close speech, seconds (default 0.75). */
  minSilenceDuration?: number;
  /**
   * Split segments longer than this, seconds (default 14).
   * Applies to `segment()` only — streaming `push()` never force-splits.
   */
  maxSpeechDuration?: number;
  /** Padding added around each segment / boundary event, seconds (default 0.1). */
  speechPadding?: number;
  /**
   * Pin the exit threshold of the hysteresis (default: threshold − 0.15).
   * Only affects when speech *ends*; the entry threshold stays `threshold`.
   * Must not exceed `threshold`.
   */
  negativeThreshold?: number;
}

export interface LoadOptions {
  /**
   * Explicit wasm source (bytes, Response, URL, or compiled Module),
   * bypassing environment detection — the escape hatch for Electron
   * renderers with contextIsolation and for unusual bundlers.
   */
  wasm?: BufferSource | WebAssembly.Module | Response | URL | Promise<Response>;
}

/** Initialize the wasm module once. Safe to call multiple times (first call wins). */
export function load(options?: LoadOptions): Promise<void>;

/** Load the wasm and construct a FluidVad in one call. */
export function createVad(
  options?: VadConfig,
  loadOptions?: LoadOptions
): Promise<import("./dist/fluidvad.js").FluidVad>;
