export { FluidVad, VadOptions, VadEvent, Segment } from "./dist/fluidvad.js";

/** User-facing configuration; all fields optional, defaults match Silero. */
export interface VadConfig {
  /** Entry threshold (default 0.5). */
  threshold?: number;
  /** Minimum speech run to keep, seconds (default 0.15). */
  minSpeechDuration?: number;
  /** Silence needed to close a segment, seconds (default 0.75). */
  minSilenceDuration?: number;
  /** Split segments longer than this, seconds (default 14). */
  maxSpeechDuration?: number;
  /** Padding added around each segment, seconds (default 0.1). */
  speechPadding?: number;
  /** Pin the exit threshold (default: threshold - 0.15). */
  negativeThreshold?: number;
}

/** Initialize the wasm module once. Safe to call multiple times. */
export function load(): Promise<void>;

/** Load the wasm and construct a FluidVad in one call. */
export function createVad(options?: VadConfig): Promise<import("./dist/fluidvad.js").FluidVad>;
