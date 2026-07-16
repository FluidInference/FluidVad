/**
 * MicVad — browser microphone VAD with speech-boundary callbacks.
 *
 * Captures the mic through an AudioWorklet at 16 kHz, runs FluidVad on the
 * main thread (sub-millisecond per 32 ms frame), buffers audio while speaking
 * (plus pre-roll), and emits whole utterances on speech end.
 */
import { createVad } from "./index.js";

const SAMPLE_RATE = 16000;
const FRAME_SIZE = 512;

export class MicVad {
  /**
   * @param {object} [options]
   * @param {(time: number) => void} [options.onSpeechStart]
   * @param {(audio: Float32Array, startTime: number, endTime: number) => void} [options.onSpeechEnd]
   * @param {(probability: number) => void} [options.onFrame]
   * @param {import('./index.js').VadConfig} [options.vad] VAD configuration.
   * @param {MediaTrackConstraints} [options.audioConstraints] extra getUserMedia constraints.
   */
  constructor(options = {}) {
    this.options = options;
    this._context = null;
    this._stream = null;
    this._node = null;
    this._vad = null;
    this._speechBuffer = [];
    this._preroll = [];
    this._prerollFrames = Math.ceil(
      ((options.vad && options.vad.speechPadding) ?? 0.1) * SAMPLE_RATE / FRAME_SIZE
    ) + 1;
    this._speechStartTime = 0;
  }

  /** Request the microphone and start emitting callbacks. */
  async start() {
    if (this._context) return;
    this._vad = await createVad(this.options.vad);

    this._stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        ...(this.options.audioConstraints || {}),
      },
    });

    // 16 kHz context: the browser resamples the mic natively
    this._context = new AudioContext({ sampleRate: SAMPLE_RATE });
    await this._context.audioWorklet.addModule(new URL("./worklet.js", import.meta.url));

    const source = this._context.createMediaStreamSource(this._stream);
    this._node = new AudioWorkletNode(this._context, "fluidvad-processor", {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      channelCount: 1,
    });
    this._node.port.onmessage = (e) => this._onFrame(e.data);
    source.connect(this._node);
  }

  _onFrame(frame) {
    const events = this._vad.push(frame);
    this.options.onFrame?.(this._vad.lastProbability);

    if (this._vad.isSpeaking) {
      this._speechBuffer.push(frame);
    } else if (this._speechBuffer.length === 0) {
      // rolling pre-speech padding
      this._preroll.push(frame);
      if (this._preroll.length > this._prerollFrames) this._preroll.shift();
    }

    for (const event of events) {
      if (event.isStart) {
        this._speechStartTime = event.timeSeconds;
        // seed the utterance with the pre-roll
        this._speechBuffer = [...this._preroll, frame];
        this._preroll = [];
        this.options.onSpeechStart?.(event.timeSeconds);
      } else {
        const audio = concat(this._speechBuffer);
        this._speechBuffer = [];
        this.options.onSpeechEnd?.(audio, this._speechStartTime, event.timeSeconds);
      }
    }
  }

  /** True while the state machine is inside speech. */
  get isSpeaking() {
    return this._vad ? this._vad.isSpeaking : false;
  }

  /** Stop capture and release the microphone. */
  async stop() {
    this._node?.disconnect();
    this._stream?.getTracks().forEach((t) => t.stop());
    await this._context?.close();
    this._vad?.reset();
    this._context = this._stream = this._node = null;
    this._speechBuffer = [];
    this._preroll = [];
  }
}

function concat(frames) {
  const out = new Float32Array(frames.length * FRAME_SIZE);
  frames.forEach((f, i) => out.set(f, i * FRAME_SIZE));
  return out;
}
