/**
 * MicVad — browser microphone VAD with speech-boundary callbacks.
 *
 * Captures the mic through an AudioWorklet at 16 kHz, runs FluidVad on the
 * main thread (sub-millisecond per 32 ms frame), buffers audio while speaking
 * (plus pre-roll), and emits whole utterances on speech end. The emitted
 * audio is trimmed to exactly [startTime, endTime].
 */
import { createVad, FluidVad } from "./index.js";

const SAMPLE_RATE = 16000;

export class MicVad {
  /**
   * @param {object} [options]
   * @param {(time: number) => void} [options.onSpeechStart]
   * @param {(audio: Float32Array, startTime: number, endTime: number) => void} [options.onSpeechEnd]
   * @param {(probability: number) => void} [options.onFrame]
   * @param {import('./index.js').VadConfig} [options.vad] VAD configuration.
   * @param {import('./index.js').LoadOptions} [options.load] wasm loading override
   *   (e.g. `{ wasm: bytes }` in Electron renderers with contextIsolation).
   * @param {string | URL} [options.workletUrl] override the worklet module URL —
   *   needed with bundlers that don't rewrite `new URL(..., import.meta.url)`
   *   (e.g. esbuild); point it at your emitted copy of `worklet.js`.
   * @param {MediaTrackConstraints} [options.audioConstraints] extra getUserMedia constraints.
   */
  constructor(options = {}) {
    this.options = options;
    this._context = null;
    this._stream = null;
    this._node = null;
    this._vad = null;
    this._startPromise = null;
    this._frameSize = 512; // updated from the wasm constant after load
    this._framesSeen = 0;
    /** buffered frames while speaking (plus rolling pre-roll), with absolute sample offsets */
    this._buffer = [];
    this._speechStartSample = 0;
    // pre-roll must cover the backdated start: padding + the min-speech
    // confirmation window + one frame of margin
    const pad = (options.vad && options.vad.speechPadding) ?? 0.1;
    const minSpeech = (options.vad && options.vad.minSpeechDuration) ?? 0.15;
    this._prerollSamples = Math.ceil((pad + minSpeech + 0.064) * SAMPLE_RATE);
  }

  /** Request the microphone and start emitting callbacks. */
  start() {
    if (!this._startPromise) {
      this._startPromise = this._start().catch((e) => {
        this._startPromise = null;
        this._teardown();
        throw e;
      });
    }
    return this._startPromise;
  }

  async _start() {
    this._vad = await createVad(this.options.vad, this.options.load);
    this._frameSize = FluidVad.frameSize();

    this._stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        ...(this.options.audioConstraints || {}),
      },
    });

    // stop() may have been called while we were awaiting — abort cleanly
    if (!this._startPromise) {
      this._teardown();
      return;
    }

    // 16 kHz context: the browser resamples the mic natively
    this._context = new AudioContext({ sampleRate: SAMPLE_RATE });
    const workletUrl = this.options.workletUrl ?? new URL("./worklet.js", import.meta.url);
    await this._context.audioWorklet.addModule(workletUrl);

    if (!this._startPromise) {
      this._teardown();
      return;
    }

    const source = this._context.createMediaStreamSource(this._stream);
    this._node = new AudioWorkletNode(this._context, "fluidvad-processor", {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      channelCount: 1,
      processorOptions: { frameSize: this._frameSize },
    });
    this._node.port.onmessage = (e) => this._onFrame(e.data);
    source.connect(this._node);
  }

  _onFrame(frame) {
    if (!this._vad) return; // late message after stop()
    const frameStart = this._framesSeen * this._frameSize;
    this._framesSeen += 1;

    const events = this._vad.push(frame);
    this.options.onFrame?.(this._vad.lastProbability);

    this._buffer.push({ frame, start: frameStart });
    if (!this._vad.isSpeaking && !events.some((e) => !e.isStart)) {
      // rolling pre-roll: keep just enough to cover the backdated speech start
      const cutoff = frameStart + this._frameSize - this._prerollSamples;
      while (this._buffer.length && this._buffer[0].start + this._frameSize <= cutoff) {
        this._buffer.shift();
      }
    }

    for (const event of events) {
      if (event.isStart) {
        this._speechStartSample = event.sampleIndex;
        this.options.onSpeechStart?.(event.timeSeconds);
      } else {
        const audio = this._extract(this._speechStartSample, event.sampleIndex);
        // keep only frames at/after the boundary as the next pre-roll
        this._buffer = this._buffer.filter(
          (f) => f.start + this._frameSize > event.sampleIndex
        );
        this.options.onSpeechEnd?.(
          audio,
          this._speechStartSample / SAMPLE_RATE,
          event.sampleIndex / SAMPLE_RATE
        );
      }
    }
  }

  /** Concatenate buffered frames and slice to exactly [startSample, endSample). */
  _extract(startSample, endSample) {
    const available = this._buffer.filter(
      (f) => f.start + this._frameSize > startSample && f.start < endSample
    );
    if (available.length === 0) return new Float32Array(0);
    const base = available[0].start;
    const out = new Float32Array(
      available[available.length - 1].start + this._frameSize - base
    );
    for (const f of available) out.set(f.frame, f.start - base);
    const from = Math.max(0, startSample - base);
    const to = Math.min(out.length, endSample - base);
    return out.slice(from, to);
  }

  /** True while the state machine is inside speech. */
  get isSpeaking() {
    return this._vad ? this._vad.isSpeaking : false;
  }

  /** Stop capture and release the microphone. Safe during a pending start(). */
  async stop() {
    const pending = this._startPromise;
    this._startPromise = null;
    if (pending) {
      try {
        await pending;
      } catch {
        return; // start() already failed and tore down
      }
    }
    this._teardown();
  }

  _teardown() {
    if (this._node) {
      this._node.port.onmessage = null; // silence queued worklet messages
      this._node.disconnect();
    }
    this._stream?.getTracks().forEach((t) => t.stop());
    this._context?.close().catch(() => {});
    this._vad?.reset();
    this._vad = null;
    this._context = null;
    this._stream = null;
    this._node = null;
    this._buffer = [];
    this._framesSeen = 0;
  }
}
