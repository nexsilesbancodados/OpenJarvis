/**
 * Voice activity detection — the decision half, with no audio APIs in it.
 *
 * The worklet measures loudness; this decides what that loudness *means*. It
 * is pure so the awkward cases can be tested without a microphone: a cough
 * must not open a turn, a breath mid-sentence must not close one, and the
 * assistant's own voice leaking into the mic must not be heard as barge-in.
 *
 * Two thresholds, not one. A single threshold chatters: any level hovering at
 * the boundary flips start/stop every frame. Speech opens at the higher
 * threshold and only closes below the lower one, so the state is stable
 * exactly where a naive detector is worst.
 *
 * Time is passed in rather than read, so tests are deterministic.
 */

export interface VadConfig {
  /** RMS above which speech may start. */
  startThreshold: number;
  /** RMS below which speech may stop. Must be < startThreshold. */
  stopThreshold: number;
  /** Loudness must persist this long before a turn opens — rejects clicks. */
  minSpeechMs: number;
  /** Silence must persist this long before a turn closes — allows breaths. */
  silenceHangoverMs: number;
  /** Hard cap on one utterance, so a stuck detector cannot listen forever. */
  maxUtteranceMs: number;
  /**
   * While the assistant is speaking, require this much more energy to treat
   * input as barge-in. Echo cancellation is good but not perfect, and a false
   * barge-in — the assistant interrupting itself — is far more damaging than
   * a missed one.
   */
  duckedMultiplier: number;
}

export const DEFAULT_VAD: VadConfig = {
  startThreshold: 0.028,
  stopThreshold: 0.014,
  minSpeechMs: 120,
  silenceHangoverMs: 620,
  maxUtteranceMs: 30_000,
  duckedMultiplier: 2.2,
};

export type VadEvent = 'speech_start' | 'speech_end' | null;

interface Frame {
  /** Root-mean-square amplitude of the frame, 0..1. */
  rms: number;
  /** Monotonic milliseconds. */
  now: number;
  /** True while the assistant's audio is playing. */
  assistantSpeaking: boolean;
}

export class Vad {
  private readonly cfg: VadConfig;
  private speaking = false;
  /** When loudness first crossed the start threshold, or null. */
  private candidateSince: number | null = null;
  /** When loudness first fell below the stop threshold, or null. */
  private quietSince: number | null = null;
  private startedAt = 0;
  /**
   * Set when a turn is cut short by the length cap. Blocks a new turn until
   * the input has genuinely gone quiet at least once. Without it, constant
   * noise — a fan, a stuck gain stage — would be chopped into an endless
   * series of max-length utterances, each one a transcription request. The
   * cap has to bound the damage, not just the length.
   */
  private awaitingQuiet = false;

  constructor(config: Partial<VadConfig> = {}) {
    this.cfg = { ...DEFAULT_VAD, ...config };
  }

  get isSpeaking(): boolean {
    return this.speaking;
  }

  /** Feed one frame. Returns an event only on a transition. */
  push({ rms, now, assistantSpeaking }: Frame): VadEvent {
    const gate = assistantSpeaking ? this.cfg.duckedMultiplier : 1;
    const startAt = this.cfg.startThreshold * gate;
    const stopAt = this.cfg.stopThreshold * gate;

    if (!this.speaking) {
      if (rms < stopAt) {
        // Genuine quiet clears a cap-induced lockout.
        this.awaitingQuiet = false;
      }
      if (this.awaitingQuiet || rms < startAt) {
        this.candidateSince = null;
        return null;
      }
      if (this.candidateSince === null) {
        this.candidateSince = now;
        return null;
      }
      // Loud for long enough to be a voice rather than a click.
      if (now - this.candidateSince >= this.cfg.minSpeechMs) {
        this.speaking = true;
        this.startedAt = now;
        this.candidateSince = null;
        this.quietSince = null;
        return 'speech_start';
      }
      return null;
    }

    // Speaking: a pause is not the end of a turn until it outlasts a breath.
    if (rms >= stopAt) {
      this.quietSince = null;
    } else if (this.quietSince === null) {
      this.quietSince = now;
    } else if (now - this.quietSince >= this.cfg.silenceHangoverMs) {
      return this.close();
    }

    if (now - this.startedAt >= this.cfg.maxUtteranceMs) {
      this.awaitingQuiet = true;
      return this.close();
    }
    return null;
  }

  /** Force the turn closed — the socket dropped, the user hit mute. */
  reset(): VadEvent {
    this.candidateSince = null;
    return this.speaking ? this.close() : null;
  }

  private close(): VadEvent {
    this.speaking = false;
    this.quietSince = null;
    this.candidateSince = null;
    return 'speech_end';
  }
}

/** RMS of a frame of float samples in −1..1. */
export function rms(samples: Float32Array): number {
  if (samples.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < samples.length; i += 1) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}
