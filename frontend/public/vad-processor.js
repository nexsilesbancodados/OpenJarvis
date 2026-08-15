/**
 * AudioWorklet: downsample the mic to 16 kHz PCM16 and report loudness.
 *
 * Served as a static asset on purpose. Worklet module fetches are checked
 * against `script-src`, which allows 'self' but not blob: in either the
 * backend or the Tauri CSP — loading this from a blob URL fails, and widening
 * script-src to blob: to work around that would be the wrong trade.
 *
 * Runs on the audio thread, so it stays arithmetic only: no allocation per
 * sample, no JSON, no main-thread round trips. It posts a small message per
 * ~20 ms frame and lets the main thread decide what the loudness means
 * (src/lib/vad.ts). Keeping the decision out of here is what makes the
 * awkward cases — clicks, breaths, echo — testable without a microphone.
 */

const TARGET_RATE = 16000;
// 20 ms at 16 kHz. Small enough that barge-in feels immediate, large enough
// that RMS is a stable measure rather than sample noise.
const FRAME_SAMPLES = 320;

class VadProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / TARGET_RATE;
    this.buffer = new Int16Array(FRAME_SAMPLES);
    this.filled = 0;
    // Fractional read position into the current input block, carried across
    // blocks so resampling does not drift or click at block boundaries.
    this.cursor = 0;
    this.muted = false;

    this.port.onmessage = (event) => {
      const data = event.data || {};
      if (typeof data.muted === 'boolean') this.muted = data.muted;
    };
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel || channel.length === 0) return true;
    if (this.muted) return true;

    // Linear-interpolated decimation. Good enough for speech at 16 kHz and
    // cheap enough to run on the audio thread without risking underruns.
    let position = this.cursor;
    while (position < channel.length) {
      const index = Math.floor(position);
      const frac = position - index;
      const a = channel[index];
      const b = index + 1 < channel.length ? channel[index + 1] : a;
      const sample = a + (b - a) * frac;

      const clamped = sample < -1 ? -1 : sample > 1 ? 1 : sample;
      this.buffer[this.filled] = clamped * 0x7fff;
      this.filled += 1;

      if (this.filled === FRAME_SAMPLES) {
        let sum = 0;
        for (let i = 0; i < FRAME_SAMPLES; i += 1) {
          const value = this.buffer[i] / 0x7fff;
          sum += value * value;
        }
        // Copy: the buffer is reused immediately, and a transferred view
        // would be detached out from under the next frame.
        const pcm = this.buffer.slice();
        this.port.postMessage(
          { rms: Math.sqrt(sum / FRAME_SAMPLES), pcm: pcm.buffer },
          [pcm.buffer],
        );
        this.filled = 0;
      }
      position += this.ratio;
    }
    this.cursor = position - channel.length;
    return true;
  }
}

registerProcessor('vad-processor', VadProcessor);
