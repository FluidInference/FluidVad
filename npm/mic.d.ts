import type { LoadOptions, VadConfig } from "./index.js";

export interface MicVadOptions {
  /** Called when speech starts (seconds since capture start). */
  onSpeechStart?: (time: number) => void;
  /**
   * Called with the utterance audio (16 kHz mono), trimmed to exactly
   * [startTime, endTime] — `audio.length === (endTime - startTime) * 16000`.
   */
  onSpeechEnd?: (audio: Float32Array, startTime: number, endTime: number) => void;
  /** Called every 32 ms frame with the current speech probability. */
  onFrame?: (probability: number) => void;
  /** VAD configuration. */
  vad?: VadConfig;
  /** wasm loading override (e.g. `{ wasm: bytes }` in Electron renderers). */
  load?: LoadOptions;
  /**
   * Override the worklet module URL — needed with bundlers that don't rewrite
   * `new URL(..., import.meta.url)` (e.g. esbuild).
   */
  workletUrl?: string | URL;
  /** Extra getUserMedia audio constraints. */
  audioConstraints?: MediaTrackConstraints;
}

/** Browser microphone VAD with speech-boundary callbacks. */
export class MicVad {
  constructor(options?: MicVadOptions);
  start(): Promise<void>;
  stop(): Promise<void>;
  readonly isSpeaking: boolean;
}
