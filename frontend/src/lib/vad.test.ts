import { describe, expect, it } from 'vitest';
import { DEFAULT_VAD, Vad, rms } from './vad';

const LOUD = 0.2;
const QUIET = 0.001;

/** Feed frames at a fixed cadence and collect the transitions. */
function drive(
  vad: Vad,
  frames: Array<{ rms: number; ms: number; assistantSpeaking?: boolean }>,
  step = 20,
): string[] {
  const events: string[] = [];
  let now = 0;
  for (const frame of frames) {
    for (let elapsed = 0; elapsed < frame.ms; elapsed += step) {
      const event = vad.push({
        rms: frame.rms,
        now,
        assistantSpeaking: frame.assistantSpeaking ?? false,
      });
      if (event) events.push(event);
      now += step;
    }
  }
  return events;
}

describe('rms', () => {
  it('is zero for silence and positive for signal', () => {
    expect(rms(new Float32Array(128))).toBe(0);
    expect(rms(new Float32Array([0.5, -0.5]))).toBeCloseTo(0.5);
  });

  it('handles an empty frame without dividing by zero', () => {
    expect(rms(new Float32Array())).toBe(0);
  });
});

describe('turn boundaries', () => {
  it('opens and closes a normal utterance', () => {
    const events = drive(new Vad(), [
      { rms: QUIET, ms: 200 },
      { rms: LOUD, ms: 900 },
      { rms: QUIET, ms: 900 },
    ]);
    expect(events).toEqual(['speech_start', 'speech_end']);
  });

  it('ignores a click too short to be a voice', () => {
    const events = drive(new Vad(), [
      { rms: QUIET, ms: 200 },
      { rms: LOUD, ms: 40 },
      { rms: QUIET, ms: 600 },
    ]);
    expect(events).toEqual([]);
  });

  it('does not end the turn on a breath mid-sentence', () => {
    const events = drive(new Vad(), [
      { rms: LOUD, ms: 500 },
      { rms: QUIET, ms: 250 }, // shorter than the hangover
      { rms: LOUD, ms: 500 },
      { rms: QUIET, ms: 900 },
    ]);
    expect(events).toEqual(['speech_start', 'speech_end']);
  });

  it('does not chatter at the threshold boundary', () => {
    // A level sitting between stop and start would flip a single-threshold
    // detector on every frame.
    const between = (DEFAULT_VAD.startThreshold + DEFAULT_VAD.stopThreshold) / 2;
    const events = drive(new Vad(), [
      { rms: LOUD, ms: 300 },
      { rms: between, ms: 1500 },
    ]);
    expect(events).toEqual(['speech_start']);
  });

  it('closes a runaway utterance rather than listening forever', () => {
    const vad = new Vad({ maxUtteranceMs: 500 });
    const events = drive(vad, [{ rms: LOUD, ms: 1200 }]);
    expect(events).toEqual(['speech_start', 'speech_end']);
    expect(vad.isSpeaking).toBe(false);
  });

  it('does not chop constant noise into endless utterances', () => {
    // A fan or a stuck gain stage never goes quiet. Reopening a turn each
    // time the cap fires would mean a transcription request every 500ms.
    const vad = new Vad({ maxUtteranceMs: 500 });
    const events = drive(vad, [{ rms: LOUD, ms: 5000 }]);
    expect(events).toEqual(['speech_start', 'speech_end']);
  });

  it('resumes after the input genuinely goes quiet', () => {
    const vad = new Vad({ maxUtteranceMs: 500 });
    const events = drive(vad, [
      { rms: LOUD, ms: 1200 },
      { rms: QUIET, ms: 400 },
      { rms: LOUD, ms: 400 },
    ]);
    expect(events).toEqual(['speech_start', 'speech_end', 'speech_start']);
  });
});

describe('barge-in while the assistant talks', () => {
  it('needs more energy than when it is silent', () => {
    // Loud enough to open normally, not loud enough to be barge-in.
    const marginal = DEFAULT_VAD.startThreshold * 1.4;

    expect(
      drive(new Vad(), [{ rms: marginal, ms: 600 }]),
    ).toEqual(['speech_start']);

    expect(
      drive(new Vad(), [
        { rms: marginal, ms: 600, assistantSpeaking: true },
      ]),
    ).toEqual([]);
  });

  it('still opens for genuinely loud speech', () => {
    const events = drive(new Vad(), [
      { rms: LOUD, ms: 500, assistantSpeaking: true },
    ]);
    expect(events).toEqual(['speech_start']);
  });

  it('leaked assistant audio does not open a turn', () => {
    // Echo cancellation leaves a residue below the ducked threshold.
    const residue = DEFAULT_VAD.startThreshold * 1.1;
    const events = drive(new Vad(), [
      { rms: residue, ms: 3000, assistantSpeaking: true },
    ]);
    expect(events).toEqual([]);
  });
});

describe('reset', () => {
  it('closes an open turn', () => {
    const vad = new Vad();
    drive(vad, [{ rms: LOUD, ms: 400 }]);
    expect(vad.isSpeaking).toBe(true);
    expect(vad.reset()).toBe('speech_end');
    expect(vad.isSpeaking).toBe(false);
  });

  it('is a no-op when nothing is open', () => {
    expect(new Vad().reset()).toBeNull();
  });

  it('leaves the detector usable afterwards', () => {
    const vad = new Vad();
    drive(vad, [{ rms: LOUD, ms: 400 }]);
    vad.reset();
    const events = drive(vad, [
      { rms: QUIET, ms: 100 },
      { rms: LOUD, ms: 400 },
    ]);
    expect(events).toEqual(['speech_start']);
  });
});
