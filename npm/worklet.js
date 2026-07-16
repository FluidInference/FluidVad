/**
 * FluidVad capture worklet: batches mono input into fixed-size frames and
 * posts them to the main thread. Runs inside AudioWorkletGlobalScope —
 * registered by `mic.js` via `audioWorklet.addModule`.
 *
 * The AudioContext is created at 16 kHz (mic.js), so the browser handles
 * resampling; this processor only frames the stream.
 */
const FRAME_SIZE = 512;

class FluidVadProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Float32Array(FRAME_SIZE);
    this._fill = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;

    let offset = 0;
    while (offset < channel.length) {
      const take = Math.min(FRAME_SIZE - this._fill, channel.length - offset);
      this._buffer.set(channel.subarray(offset, offset + take), this._fill);
      this._fill += take;
      offset += take;
      if (this._fill === FRAME_SIZE) {
        // transfer a copy; _buffer is reused
        const frame = this._buffer.slice();
        this.port.postMessage(frame, [frame.buffer]);
        this._fill = 0;
      }
    }
    return true;
  }
}

registerProcessor("fluidvad-processor", FluidVadProcessor);
