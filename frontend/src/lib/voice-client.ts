/**
 * Client for `WS /v1/voice/session` — microphone in, spoken answer out.
 *
 * Nothing here is imported by the existing UI yet; wire it up when the voice
 * surface is ready. Usage:
 *
 *     const voice = new VoiceClient({ onState: setOrbState });
 *     await voice.start();     // asks for the mic, opens the socket
 *     voice.setMuted(true);    // stops listening and speaking
 *     await voice.stop();      // releases the mic and closes the socket
 *
 * Three things this file exists to get right, all of them things that are
 * invisible in code review and obvious in use:
 *
 * 1. **Echo.** `getUserMedia` is asked for echo cancellation, noise
 *    suppression and auto gain explicitly. Without them the assistant hears
 *    itself through the speakers and interrupts itself mid-sentence.
 * 2. **Barge-in latency.** Speech detection runs on 20 ms frames in a
 *    worklet, so "stop talking" happens within a frame or two rather than
 *    after the current sentence.
 * 3. **Gapless playback.** Audio arrives one sentence at a time and is
 *    scheduled back-to-back on the Web Audio clock. Using an `<audio>`
 *    element per chunk leaves an audible seam between sentences.
 */

import { getApiKey, getBase } from './api';
import { Vad, type VadConfig } from './vad';

export type VoiceState =
  | 'idle'
  | 'connecting'
  | 'listening'
  | 'user_speaking'
  | 'thinking'
  | 'tool'
  | 'speaking'
  | 'error';

export interface VoiceClientOptions {
  onState?: (state: VoiceState) => void;
  /** Interim and final transcripts of what the user said. */
  onTranscript?: (text: string) => void;
  /** Reply text as it streams, for on-screen captions. */
  onDelta?: (text: string, full: string) => void;
  /** A turn ended. `interrupted` means the user cut in. */
  onTurnEnd?: (text: string, interrupted: boolean) => void;
  onError?: (message: string) => void;
  /** Drives the orb's amplitude, 0..1. Called ~50×/s while listening. */
  onLevel?: (rms: number) => void;
  deviceId?: string;
  voice?: string;
  model?: string;
  vad?: Partial<VadConfig>;
}

const WORKLET_URL = '/vad-processor.js';
const SOCKET_OPEN_TIMEOUT_MS = 10_000;
// 20 ms per frame. Ten frames covers the detector's confirmation delay
// with margin, at a memory cost of ~6 kB.
const PREROLL_FRAMES = 10;

export class VoiceClient {
  private readonly opts: VoiceClientOptions;
  private readonly vad: Vad;

  private socket: WebSocket | null = null;
  private stream: MediaStream | null = null;
  private context: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;

  private playback: AudioContext | null = null;
  private playHead = 0;
  private playing = 0;

  /** Recent frames held back before the detector confirms speech. */
  private readonly preroll: ArrayBuffer[] = [];

  private state: VoiceState = 'idle';
  private full = '';
  private muted = false;
  private stopped = false;

  constructor(options: VoiceClientOptions = {}) {
    this.opts = options;
    this.vad = new Vad(options.vad);
  }

  get currentState(): VoiceState {
    return this.state;
  }

  // -- lifecycle ------------------------------------------------------

  async start(): Promise<void> {
    if (this.socket) return;
    this.stopped = false;
    this.setState('connecting');
    try {
      await this.openMicrophone();
      await this.openSocket();
    } catch (error) {
      this.setState('error');
      this.opts.onError?.(describe(error));
      await this.stop();
      throw error;
    }
  }

  async stop(): Promise<void> {
    this.stopped = true;
    this.vad.reset();

    this.node?.port.close();
    this.node?.disconnect();
    this.source?.disconnect();
    this.stream?.getTracks().forEach((track) => track.stop());
    await this.context?.close().catch(() => {});
    await this.playback?.close().catch(() => {});

    if (this.socket && this.socket.readyState <= WebSocket.OPEN) {
      this.socket.close(1000, 'client stopped');
    }

    this.node = null;
    this.source = null;
    this.stream = null;
    this.context = null;
    this.playback = null;
    this.socket = null;
    this.playing = 0;
    this.setState('idle');
  }

  setMuted(muted: boolean): void {
    this.muted = muted;
    this.node?.port.postMessage({ muted });
    if (muted) {
      this.vad.reset();
      this.stopPlayback();
    }
    this.send({ type: 'mute', value: muted });
  }

  /** Stop the current answer without it counting as an interruption. */
  cancel(): void {
    this.stopPlayback();
    this.send({ type: 'cancel' });
  }

  /** Send typed text through the same turn machinery as speech. */
  say(text: string): void {
    if (text.trim()) this.send({ type: 'text', text });
  }

  // -- microphone -----------------------------------------------------

  private async openMicrophone(): Promise<void> {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('This browser cannot reach the microphone.');
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        // Not defaults — without these the assistant hears itself.
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
        ...(this.opts.deviceId ? { deviceId: { exact: this.opts.deviceId } } : {}),
      },
    });

    this.context = new AudioContext();
    await this.context.audioWorklet.addModule(WORKLET_URL);
    this.source = this.context.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.context, 'vad-processor');
    this.node.port.onmessage = (event) => this.onFrame(event.data);
    this.source.connect(this.node);
    // Deliberately not connected to the destination: routing the mic to the
    // speakers is how you build a feedback loop.
  }

  private onFrame(data: { rms: number; pcm: ArrayBuffer }): void {
    if (this.stopped || this.muted) return;
    this.opts.onLevel?.(data.rms);

    const speakingBefore = this.vad.isSpeaking;
    const event = this.vad.push({
      rms: data.rms,
      now: performance.now(),
      assistantSpeaking: this.state === 'speaking',
    });

    // The detector needs a few frames of loudness before it will call
    // something speech, so by the time it says yes the first syllable has
    // already gone past. Keeping a short rolling buffer and flushing it on
    // speech_start is the difference between "abre o Chrome" and "bre o
    // Chrome" reaching the transcriber.
    if (!speakingBefore && event !== 'speech_start') {
      this.preroll.push(data.pcm);
      if (this.preroll.length > PREROLL_FRAMES) this.preroll.shift();
    }

    if (event === 'speech_start') {
      // Stop our own audio immediately rather than waiting for the server to
      // acknowledge — the round trip is the difference between "it stopped"
      // and "it talked over me".
      this.stopPlayback();
      this.send({ type: 'speech_start' });
      for (const frame of this.preroll) this.sendAudio(frame);
      this.preroll.length = 0;
    }

    if (this.vad.isSpeaking) this.sendAudio(data.pcm);

    if (event === 'speech_end') {
      this.preroll.length = 0;
      this.send({ type: 'speech_end' });
    }
  }

  private sendAudio(pcm: ArrayBuffer): void {
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(pcm);
  }

  // -- socket ---------------------------------------------------------

  private async openSocket(): Promise<void> {
    const base = getBase() || window.location.origin;
    const url = new URL('/v1/voice/session', base);
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    url.searchParams.set('format', 'pcm16');
    if (this.opts.voice) url.searchParams.set('voice', this.opts.voice);
    if (this.opts.model) url.searchParams.set('model', this.opts.model);
    const key = getApiKey();
    if (key) url.searchParams.set('token', key);

    const socket = new WebSocket(url.toString());
    socket.binaryType = 'arraybuffer';
    this.socket = socket;

    await new Promise<void>((resolve, reject) => {
      let settled = false;
      const timeout = window.setTimeout(() => {
        if (settled) return;
        settled = true;
        socket.close();
        reject(new Error('O serviço de voz demorou demais para responder.'));
      }, SOCKET_OPEN_TIMEOUT_MS);
      const finish = (error?: Error) => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timeout);
        if (error) reject(error);
        else resolve();
      };
      socket.onopen = () => finish();
      socket.onerror = () => finish(new Error('Não foi possível conectar ao serviço de voz.'));
      socket.onclose = () => finish(new Error('A conexão de voz foi encerrada.'));
    });

    socket.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        void this.enqueueAudio(event.data);
        return;
      }
      try {
        this.onMessage(JSON.parse(event.data as string));
      } catch {
        /* a frame we do not understand is not worth breaking the session */
      }
    };
    socket.onclose = () => {
      if (!this.stopped) {
        this.setState('error');
        this.opts.onError?.('The voice connection dropped.');
      }
    };
  }

  private onMessage(message: Record<string, unknown>): void {
    switch (message.type) {
      case 'state':
        this.setState(message.value as VoiceState);
        break;
      case 'transcript':
        this.opts.onTranscript?.(String(message.text ?? ''));
        break;
      case 'delta': {
        const text = String(message.text ?? '');
        this.full += text;
        this.opts.onDelta?.(text, this.full);
        break;
      }
      case 'interrupted':
        this.stopPlayback();
        this.opts.onTurnEnd?.(String(message.spoken ?? ''), true);
        this.full = '';
        break;
      case 'turn_complete':
        this.opts.onTurnEnd?.(String(message.text ?? ''), false);
        this.full = '';
        break;
      case 'error':
        this.opts.onError?.(String(message.message ?? 'Something went wrong.'));
        break;
      default:
        break;
    }
  }

  private send(payload: Record<string, unknown>): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(payload));
    }
  }

  // -- playback -------------------------------------------------------

  private async enqueueAudio(data: ArrayBuffer): Promise<void> {
    if (this.stopped || this.muted || data.byteLength === 0) return;
    if (!this.playback) this.playback = new AudioContext();

    let buffer: AudioBuffer;
    try {
      buffer = await this.playback.decodeAudioData(data.slice(0));
    } catch {
      return; // a chunk we cannot decode is dropped, not fatal
    }
    // The interrupt may have landed while we were decoding.
    if (this.stopped || this.muted || !this.playback) return;

    const source = this.playback.createBufferSource();
    source.buffer = buffer;
    source.connect(this.playback.destination);

    // Schedule against the audio clock, not wall time: back-to-back sentences
    // with no seam, which is what "fluid" means here.
    const now = this.playback.currentTime;
    const startAt = Math.max(now, this.playHead);
    source.start(startAt);
    this.playHead = startAt + buffer.duration;

    this.playing += 1;
    source.onended = () => {
      this.playing = Math.max(0, this.playing - 1);
    };
  }

  private stopPlayback(): void {
    // Closing the context is the only way to stop sources already scheduled
    // on the audio clock; a fresh one is created for the next chunk.
    const context = this.playback;
    this.playback = null;
    this.playHead = 0;
    this.playing = 0;
    void context?.close().catch(() => {});
  }

  private setState(state: VoiceState): void {
    if (state === this.state) return;
    this.state = state;
    this.opts.onState?.(state);
  }
}

function describe(error: unknown): string {
  if (error instanceof DOMException && error.name === 'NotAllowedError') {
    return 'I need permission to use the microphone.';
  }
  if (error instanceof DOMException && error.name === 'NotFoundError') {
    return 'No microphone was found.';
  }
  return error instanceof Error ? error.message : 'Could not start voice mode.';
}
