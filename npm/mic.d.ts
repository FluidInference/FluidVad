import type { VadConfig } from "./index.js";

export interface MicVadOptions {
  /** Called when speech starts (seconds since capture start). */
  onSpeechStart?: (time: number) => void;
  /** Called with the full utterance audio (16 kHz mono, includes pre-roll). */
  onSpeechEnd?: (audio: Float32Array, startTime: number, endTime: number) => void;
  /** Called every 32 ms frame with the current speech probability. */
  onFrame?: (probability: number) => void;
  /** VAD configuration. */
  vad?: VadConfig;
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
