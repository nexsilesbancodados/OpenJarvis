/**
 * React binding for the continuous voice session.
 *
 * Replaces the record → stop-on-silence → send → speak → record-again loop for
 * voice mode. That loop cannot be interrupted: the microphone is closed while
 * the assistant talks, so there is no moment at which the user speaking can be
 * heard. Here the mic stays open the whole time and the client decides, frame
 * by frame, whether the user has started talking over the answer.
 *
 * The hook owns no audio itself — it is a thin lifecycle wrapper so the UI can
 * stay declarative and `VoiceClient` can stay testable.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { VoiceClient, type VoiceState } from '../lib/voice-client';

export interface UseVoiceSessionOptions {
  /** Called with the user's transcribed utterance, once per turn. */
  onUserTurn?: (text: string) => void;
  /**
   * Called when a turn ends. `text` is what the assistant actually said —
   * for an interrupted turn that is the spoken prefix, not the full reply,
   * so the transcript on screen matches what was heard.
   */
  onAssistantTurn?: (text: string, interrupted: boolean) => void;
  model?: string;
  voice?: string;
}

export interface VoiceSessionHandle {
  state: VoiceState;
  active: boolean;
  muted: boolean;
  /** Latest user transcript, cleared when the next turn starts. */
  transcript: string;
  /** Assistant reply as it streams, for on-screen captions. */
  reply: string;
  /** Mic amplitude 0..1, for the orb. */
  level: number;
  error: string;
  start: () => Promise<void>;
  stop: () => Promise<void>;
  toggleMute: () => void;
  /** Stop the current assistant response without closing the session. */
  cancel: () => void;
  /** Send typed text through the same turn machinery. */
  say: (text: string) => void;
}

export function useVoiceSession(
  options: UseVoiceSessionOptions = {},
): VoiceSessionHandle {
  const [state, setState] = useState<VoiceState>('idle');
  const [muted, setMuted] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [reply, setReply] = useState('');
  const [level, setLevel] = useState(0);
  const [error, setError] = useState('');

  const clientRef = useRef<VoiceClient | null>(null);
  // Held in a ref so a re-render never tears down a live session just because
  // a parent passed a new inline callback.
  const optionsRef = useRef(options);
  optionsRef.current = options;

  // The orb wants amplitude at frame rate; React does not. Coalesce to one
  // state update per animation frame so a 50 Hz signal cannot cause 50
  // re-renders a second on a machine that is also running inference.
  const levelRef = useRef(0);
  const rafRef = useRef<number | null>(null);
  const pumpLevel = useCallback(() => {
    if (rafRef.current !== null) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      setLevel(levelRef.current);
    });
  }, []);

  const stop = useCallback(async () => {
    const client = clientRef.current;
    clientRef.current = null;
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    await client?.stop();
    setState('idle');
    setLevel(0);
    setMuted(false);
  }, []);

  const start = useCallback(async () => {
    if (clientRef.current) return;
    setError('');
    setTranscript('');
    setReply('');

    const client = new VoiceClient({
      model: optionsRef.current.model,
      voice: optionsRef.current.voice,
      onState: setState,
      onLevel: (rms) => {
        levelRef.current = rms;
        pumpLevel();
      },
      onTranscript: (text) => {
        setTranscript(text);
        setReply('');
        if (text.trim()) optionsRef.current.onUserTurn?.(text);
      },
      onDelta: (_delta, full) => setReply(full),
      onTurnEnd: (text, interrupted) => {
        setReply(text);
        optionsRef.current.onAssistantTurn?.(text, interrupted);
      },
      onError: setError,
    });
    clientRef.current = client;

    try {
      await client.start();
    } catch (err) {
      clientRef.current = null;
      setError(err instanceof Error ? err.message : 'Could not start voice mode.');
      setState('idle');
    }
  }, [pumpLevel]);

  const toggleMute = useCallback(() => {
    setMuted((current) => {
      const next = !current;
      clientRef.current?.setMuted(next);
      return next;
    });
  }, []);

  const say = useCallback((text: string) => {
    clientRef.current?.say(text);
  }, []);

  const cancel = useCallback(() => {
    clientRef.current?.cancel();
  }, []);

  // Release the microphone if the component unmounts mid-conversation —
  // otherwise the recording indicator stays lit and the socket stays open.
  useEffect(() => () => void stop(), [stop]);

  return {
    state,
    active: state !== 'idle',
    muted,
    transcript,
    reply,
    level,
    error,
    start,
    stop,
    toggleMute,
    cancel,
    say,
  };
}
