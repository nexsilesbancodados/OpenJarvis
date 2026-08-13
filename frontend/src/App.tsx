import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { useLocation, useNavigate } from 'react-router';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Activity,
  ArrowUpRight,
  Bot,
  BrainCircuit,
  CalendarDays,
  Check,
  CheckCircle2,
  ChevronDown,
  Circle,
  CloudSun,
  Cpu,
  Database,
  FileText,
  FolderOpen,
  Gauge,
  Globe2,
  HardDrive,
  Mail,
  LayoutDashboard,
  Library,
  Menu,
  MessageCircle,
  Mic,
  Mic2,
  Monitor,
  Music2,
  MoreHorizontal,
  Pause,
  Play,
  Plus,
  RefreshCw,
  Search,
  Send,
  Settings,
  ShieldCheck,
  Sparkles,
  Square,
  TerminalSquare,
  Thermometer,
  Volume2,
  VolumeX,
  Workflow,
  X,
  Zap,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { useAppStore } from './lib/store';
import {
  checkHealth,
  fetchEnergy,
  fetchManagedAgents,
  fetchRecommendedModel,
  fetchSavings,
  fetchModels,
  fetchServerInfo,
  fetchTelemetry,
  getMemoryStats,
  searchMemory,
  runManagedAgent,
  synthesizeSpeech,
  type ManagedAgent,
} from './lib/api';
import { connectSource, disconnectSource, getSyncStatus, listConnectors, startServerOAuth, triggerSync } from './lib/connectors-api';
import { streamChat } from './lib/sse';
import { useSpeech } from './hooks/useSpeech';
import { useVoiceSession } from './hooks/useVoiceSession';
import type { VoiceState } from './lib/voice-client';
import { ApprovalBell } from './components/ApprovalBell';
import { SiriWave } from './components/ui/siri-wave';
import type { ChatMessage, ModelInfo, SavingsData } from './types';
import { SOURCE_CATALOG, type ConnectRequest, type ConnectorInfo } from './types/connectors';

type PageKey = 'chat' | 'dashboard' | 'agents' | 'memory' | 'data' | 'activity' | 'settings';
const VOICE_SILENCE_MS = 1200;

const PAGE_PATHS: Record<PageKey, string> = {
  chat: '/voice',
  dashboard: '/',
  agents: '/agents',
  memory: '/memory',
  data: '/data-sources',
  activity: '/logs',
  settings: '/settings',
};

const NAV_ITEMS: Array<{ id: PageKey; label: string; icon: LucideIcon; hint: string }> = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard, hint: '01' },
  { id: 'chat', label: 'Voz', icon: Mic2, hint: '02' },
  { id: 'agents', label: 'Automação', icon: Workflow, hint: '03' },
  { id: 'data', label: 'Conexões', icon: Database, hint: '04' },
  { id: 'activity', label: 'Histórico', icon: Activity, hint: '05' },
  { id: 'memory', label: 'Memória', icon: BrainCircuit, hint: '06' },
];

function pageFromPath(pathname: string): PageKey {
  if (pathname === '/dashboard') return 'dashboard';
  const entry = Object.entries(PAGE_PATHS).find(([, path]) => path === pathname);
  return (entry?.[0] as PageKey | undefined) || 'dashboard';
}

function formatNumber(value: number | undefined, digits = 0): string {
  if (value === undefined || Number.isNaN(value)) return '—';
  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function timeAgo(timestamp?: number | null): string {
  if (!timestamp) return 'never';
  const seconds = Math.max(1, Math.floor((Date.now() - timestamp) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
}

function statusTone(status: string): 'cyan' | 'green' | 'amber' | 'red' | 'muted' {
  if (status === 'running' || status === 'connected' || status === 'Connected' || status === 'completed') return 'green';
  if (status === 'paused' || status === 'needs_attention' || status === 'syncing') return 'amber';
  if (status === 'error' || status === 'failed' || status === 'Authentication Error') return 'red';
  if (status === 'idle' || status === 'ready') return 'cyan';
  return 'muted';
}

function connectorIsReady(connector: ConnectorInfo): boolean {
  return connector.health_status ? connector.health_status === 'Connected' : connector.connected;
}

function Brand() {
  return (
    <div className="nova-brand">
      <div className="nova-brand-mark" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>
      <div>
        <div className="nova-brand-name">NÚCLEO</div>
        <div className="nova-brand-sub">agente inteligente</div>
      </div>
    </div>
  );
}

function StatusDot({ tone = 'cyan', pulse = false }: { tone?: ReturnType<typeof statusTone>; pulse?: boolean }) {
  return <span className={`nova-status-dot nova-status-dot--${tone}${pulse ? ' is-pulsing' : ''}`} aria-hidden="true" />;
}

function IconButton({ label, children, onClick, className = '' }: { label: string; children: ReactNode; onClick?: () => void; className?: string }) {
  return (
    <button className={`nova-icon-button ${className}`} type="button" aria-label={label} title={label} onClick={onClick}>
      {children}
    </button>
  );
}

function Metric({ label, value, detail, icon: Icon, tone = 'cyan' }: { label: string; value: string; detail: string; icon: LucideIcon; tone?: string }) {
  return (
    <div className="nova-metric">
      <div className="nova-metric-top">
        <span className={`nova-icon-chip nova-icon-chip--${tone}`}><Icon size={15} /></span>
        <span className="nova-overline">{label}</span>
        <MoreHorizontal size={16} className="nova-muted-icon" />
      </div>
      <div className="nova-metric-value">{value}</div>
      <div className="nova-metric-detail">{detail}</div>
    </div>
  );
}

function SectionTitle({ eyebrow, title, detail, action }: { eyebrow: string; title: string; detail?: string; action?: ReactNode }) {
  return (
    <div className="nova-section-title">
      <div>
        <div className="nova-overline">{eyebrow}</div>
        <h1>{title}</h1>
        {detail && <p>{detail}</p>}
      </div>
      {action}
    </div>
  );
}

function Pill({ children, tone = 'muted' }: { children: ReactNode; tone?: ReturnType<typeof statusTone> }) {
  return <span className={`nova-pill nova-pill--${tone}`}><StatusDot tone={tone} />{children}</span>;
}

function EmptyState({ onStart }: { onStart: () => void }) {
  return (
    <div className="nova-empty-state">
      <div className="nova-constellation" aria-hidden="true"><span /><span /><span /><span /><i /><i /><i /></div>
      <div className="nova-empty-orb"><Sparkles size={26} /></div>
      <div className="nova-overline">private local intelligence</div>
      <h2>What do you want to make possible?</h2>
      <p>Ask a question, explore an idea, or give Jarvis a task. Your context stays close and your thinking stays yours.</p>
      <button className="nova-primary-button" type="button" onClick={onStart}><Sparkles size={16} /> Start a conversation <ArrowUpRight size={15} /></button>
      <div className="nova-empty-hints"><span><Zap size={13} /> local-first</span><span><ShieldCheck size={13} /> private by design</span><span><Mic size={13} /> voice ready</span></div>
    </div>
  );
}

function NovaMessage({ message, live }: { message: ChatMessage; live?: boolean }) {
  const isUser = message.role === 'user';
  return (
    <article className={`nova-message nova-message--${isUser ? 'user' : 'assistant'}`}>
      <div className="nova-message-meta">
        <span className={`nova-avatar nova-avatar--${isUser ? 'user' : 'jarvis'}`}>{isUser ? 'Y' : 'J'}</span>
        <span>{isUser ? 'You' : 'Jarvis'}</span>
        <span className="nova-message-time">{new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        {!isUser && live && <Pill tone="cyan">thinking</Pill>}
      </div>
      <div className="nova-message-body">
        {isUser ? message.content : message.content ? <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown> : <div className="nova-typing"><i /><i /><i /></div>}
      </div>
      {!isUser && message.toolCalls && message.toolCalls.length > 0 && (
        <div className="nova-tool-strip"><TerminalSquare size={14} /> {message.toolCalls.map((tool) => tool.tool).join(' · ')}</div>
      )}
    </article>
  );
}

function ChatView({ onNavigate, autoStartVoice = false }: { onNavigate: (page: PageKey) => void; autoStartVoice?: boolean }) {
  const messages = useAppStore((s) => s.messages);
  const activeId = useAppStore((s) => s.activeId);
  const selectedModel = useAppStore((s) => s.selectedModel);
  const models = useAppStore((s) => s.models);
  const streamState = useAppStore((s) => s.streamState);
  const deepResearch = useAppStore((s) => s.deepResearch);
  const setDeepResearch = useAppStore((s) => s.setDeepResearch);
  const createConversation = useAppStore((s) => s.createConversation);
  const addMessage = useAppStore((s) => s.addMessage);
  const updateLastAssistant = useAppStore((s) => s.updateLastAssistant);
  const setStreamState = useAppStore((s) => s.setStreamState);
  const resetStream = useAppStore((s) => s.resetStream);
  const [draft, setDraft] = useState('');
  const [error, setError] = useState('');
  const [voiceMode, setVoiceMode] = useState(false);
  const voiceModeRef = useRef(false);
  const listRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const autoVoiceStartedRef = useRef(false);
  const speech = useSpeech();
  const isStreaming = streamState.isStreaming && streamState.conversationId === activeId;

  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [messages, streamState.content]);

  const speakAndListen = useCallback(async (text: string) => {
    if (!voiceModeRef.current) return;

    const listenAgain = () => {
      if (!voiceModeRef.current) return;
      window.setTimeout(() => {
        if (voiceModeRef.current) void speech.startRecording({ autoStopSilenceMs: VOICE_SILENCE_MS });
      }, 350);
    };

    const speakWithBrowser = () => new Promise<boolean>((resolve) => {
      if (!('speechSynthesis' in window) || !('SpeechSynthesisUtterance' in window)) {
        resolve(false);
        return;
      }
      let settled = false;
      const finish = (ok: boolean) => {
        if (settled) return;
        settled = true;
        resolve(ok);
      };
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text || 'Pronto.');
      utterance.lang = 'pt-BR';
      utterance.rate = 1;
      utterance.pitch = 1;
      utterance.onend = () => {
        finish(true);
        listenAgain();
      };
      utterance.onerror = () => finish(false);
      window.speechSynthesis.speak(utterance);
      window.setTimeout(() => finish(false), 8000);
    });

    if (await speakWithBrowser()) return;

    try {
      const blob = await synthesizeSpeech(text || 'Pronto.');
      if (!voiceModeRef.current) return;
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      const cleanup = () => URL.revokeObjectURL(url);
      audio.onended = () => { cleanup(); listenAgain(); };
      audio.onerror = () => { cleanup(); listenAgain(); };
      await audio.play();
      return;
    } catch {
      // Browser speech is a useful fallback for Tauri/offline development or
      // when the selected server TTS provider is temporarily unavailable.
    }

    listenAgain();
  }, [speech.startRecording]);

  const send = useCallback(async (value?: string): Promise<string> => {
    const content = (value ?? draft).trim();
    if (!content || isStreaming) return '';
    const model = selectedModel || models[0]?.id;
    if (!model) {
      setError('Choose a model in the top bar before sending.');
      return '';
    }
    setError('');
    setDraft('');
    const conversationId = activeId || createConversation(model);
    addMessage(conversationId, { id: `${Date.now()}-user`, role: 'user', content, timestamp: Date.now() });
    const history = useAppStore.getState().messages.map((message) => ({ role: message.role, content: message.content }));
    addMessage(conversationId, { id: `${Date.now()}-assistant`, role: 'assistant', content: '', timestamp: Date.now() });
    const controller = new AbortController();
    abortRef.current = controller;
    setStreamState({ conversationId, isStreaming: true, phase: deepResearch ? 'Researching' : 'Generating', content: '', elapsedMs: 0, activeToolCalls: [] });
    let accumulated = '';
    try {
      for await (const event of streamChat({ model, messages: history, stream: true, temperature: 0.7, max_tokens: 4096 }, controller.signal)) {
        if (event.event === 'agent_turn_start') setStreamState({ phase: 'Agent thinking' });
        if (event.event === 'tool_call_start') setStreamState({ phase: 'Using a tool' });
        if (event.event === 'inference_start') setStreamState({ phase: 'Generating' });
        if (event.event === 'error') {
          let detail = 'O agente retornou um erro durante a execução.';
          try { detail = JSON.parse(event.data).error || detail; } catch { /* keep actionable fallback */ }
          throw new Error(detail);
        }
        try {
          const data = JSON.parse(event.data);
          const delta = data.choices?.[0]?.delta?.content || data.delta?.content || '';
          if (delta) {
            accumulated += delta;
            setStreamState({ content: accumulated, phase: '' });
            updateLastAssistant(conversationId, accumulated);
          }
        } catch {
          // Non-JSON event frames are status-only.
        }
      }
      if (!accumulated) {
        accumulated = 'I could not generate a response. Check the active model and try again.';
        updateLastAssistant(conversationId, accumulated);
      }
    } catch (caught) {
      if ((caught as Error).name !== 'AbortError') {
        const message = caught instanceof Error ? caught.message : 'The conversation could not be completed.';
        setError(message);
        updateLastAssistant(conversationId, accumulated || `Connection interrupted: ${message}`);
      }
    } finally {
      abortRef.current = null;
      resetStream();
    }
    return accumulated;
  }, [activeId, addMessage, createConversation, deepResearch, draft, isStreaming, models, resetStream, selectedModel, setStreamState, updateLastAssistant]);

  const toggleVoiceMode = async () => {
    if (voiceModeRef.current) {
      voiceModeRef.current = false;
      setVoiceMode(false);
      window.speechSynthesis?.cancel();
      if (speech.state === 'recording') {
        try { await speech.stopRecording(); } catch { /* already stopped */ }
      }
      return;
    }

    if (!speech.available) {
      setError('Voice is unavailable. Check the OpenAI speech backend and microphone permission.');
      return;
    }
    voiceModeRef.current = true;
    setVoiceMode(true);
    const started = await speech.startRecording({ autoStopSilenceMs: VOICE_SILENCE_MS });
    if (!started) {
      voiceModeRef.current = false;
      setVoiceMode(false);
    }
  };

  const handleMic = async () => {
    if (speech.state === 'recording') {
      try {
        const transcript = await speech.stopRecording();
        if (!transcript) {
          if (voiceModeRef.current) await speech.startRecording({ autoStopSilenceMs: VOICE_SILENCE_MS });
          return;
        }
        if (voiceModeRef.current) {
          const answer = await send(transcript);
          await speakAndListen(answer);
        } else {
          setDraft((current) => `${current}${current ? ' ' : ''}${transcript}`);
        }
      } catch { /* surfaced by hook */ }
    } else {
      // The microphone button is also a one-click entry into hands-free
      // conversation. Users should not need to discover a second toggle just
      // to make a spoken question receive a spoken answer.
      if (!voiceModeRef.current) {
        voiceModeRef.current = true;
        setVoiceMode(true);
      }
      const started = await speech.startRecording({ autoStopSilenceMs: VOICE_SILENCE_MS });
      if (!started) {
        voiceModeRef.current = false;
        setVoiceMode(false);
      }
    }
  };

  useEffect(() => {
    if (!autoStartVoice || autoVoiceStartedRef.current) return;
    autoVoiceStartedRef.current = true;
    void handleMic();
  }, [autoStartVoice]);

  const handleAutoStop = useCallback(async (transcript: string) => {
    if (!voiceModeRef.current) return;
    if (!transcript.trim()) {
      await speech.startRecording({ autoStopSilenceMs: VOICE_SILENCE_MS });
      return;
    }
    const answer = await send(transcript);
    await speakAndListen(answer);
  }, [send, speakAndListen, speech.startRecording]);

  useEffect(() => {
    speech.setAutoStopHandler(handleAutoStop);
    return () => speech.setAutoStopHandler(null);
  }, [handleAutoStop, speech.setAutoStopHandler]);

  useEffect(() => {
    if (speech.error) setError(speech.error);
  }, [speech.error]);

  useEffect(() => () => {
    voiceModeRef.current = false;
    window.speechSynthesis?.cancel();
    void speech.stopRecording();
  }, [speech.stopRecording]);

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void send();
    }
  };

  return (
    <div className="nova-chat-layout">
      <section className="nova-chat-column">
          <div className="nova-chat-toolbar">
          <div><span className="nova-overline">live workspace</span><span className="nova-toolbar-title">Conversation</span></div>
          <div className="nova-toolbar-actions">
            <button className={`nova-toggle ${deepResearch ? 'is-active' : ''}`} type="button" onClick={() => setDeepResearch(!deepResearch)}><Search size={14} /> Deep research <span>{deepResearch ? 'ON' : 'OFF'}</span></button>
            <button className={`nova-toggle ${voiceMode ? 'is-active nova-voice-toggle' : ''}`} type="button" onClick={() => void toggleVoiceMode()} aria-pressed={voiceMode} title={voiceMode ? 'Turn off continuous voice mode' : 'Turn on continuous voice mode'}><span className="nova-voice-toggle-icon">{voiceMode ? <Volume2 size={14} /> : <Mic2 size={14} />}</span> Voice mode <span>{voiceMode ? 'ON' : 'OFF'}</span></button>
            <IconButton label="New conversation" onClick={() => createConversation(selectedModel)}><Plus size={17} /></IconButton>
          </div>
        </div>
        <div className="nova-chat-scroll" ref={listRef}>
          {messages.length === 0 && !isStreaming ? <EmptyState onStart={() => document.getElementById('nova-composer')?.focus()} /> : <div className="nova-thread">{messages.map((message, index) => <NovaMessage key={message.id} message={message} live={isStreaming && index === messages.length - 1} />)}</div>}
        </div>
        <div className="nova-composer-wrap">
          {error && <div className="nova-inline-error"><Circle size={12} fill="currentColor" /> {error} <button type="button" onClick={() => setError('')}><X size={13} /></button></div>}
          <div className={`nova-composer ${speech.state === 'recording' ? 'is-recording' : ''}`}>
            <div className="nova-composer-signal"><span /><span /><span /><span /><span /></div>
            <textarea id="nova-composer" value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={handleKeyDown} placeholder={selectedModel ? 'Ask Jarvis anything…' : 'Choose a model to begin…'} rows={1} disabled={isStreaming} />
            <div className="nova-composer-actions">
              <button className={`nova-round-button ${speech.state === 'recording' ? 'is-live' : ''} ${voiceMode ? 'is-voice-mode' : ''}`} type="button" onClick={() => void handleMic()} disabled={!speech.available || isStreaming} aria-label={speech.state === 'recording' ? 'Stop listening' : 'Start listening'} title={speech.state === 'recording' ? 'Stop listening' : 'Start listening'}><Mic size={17} /></button>
              {isStreaming ? <button className="nova-send-button is-stop" type="button" onClick={() => { abortRef.current?.abort(); resetStream(); }} aria-label="Stop generation"><Square size={15} /></button> : <button className="nova-send-button" type="button" onClick={() => void send()} disabled={!draft.trim()} aria-label="Send message"><Send size={16} /></button>}
            </div>
          </div>
          <div className="nova-composer-note" aria-live="polite"><span><kbd>Enter</kbd> send</span><span><kbd>Shift</kbd> + <kbd>Enter</kbd> new line</span><span>{voiceMode ? speech.state === 'recording' ? `listening · ${speech.mode}` : speech.state === 'transcribing' ? 'transcribing…' : 'voice mode active' : speech.available ? `voice ready · ${speech.mode}` : 'voice unavailable'}</span><button className="nova-composer-voice-link" type="button" onClick={() => void toggleVoiceMode()}>{voiceMode ? <><VolumeX size={12} /> turn off voice mode</> : <><Volume2 size={12} /> turn on voice mode</>}</button></div>
        </div>
      </section>
      <aside className="nova-context-column">
        <div className="nova-context-card nova-context-card--hero"><div className="nova-chat-siri-preview"><SiriWave variant={voiceMode || isStreaming ? 'wave' : 'fluid-dots'} size={170} renderScale={0.6} className="nova-chat-siri-wave" aria-hidden="true" /></div><div className="nova-overline">session signal</div><strong>{isStreaming ? streamState.phase || 'Processing' : 'Ready when you are'}</strong><p>{selectedModel || 'No model selected'} · private route</p><div className="nova-signal-line"><span style={{ width: isStreaming ? '74%' : '42%' }} /></div></div>
        <div className="nova-context-card"><div className="nova-card-heading"><span className="nova-overline">shortcuts</span><MoreHorizontal size={16} /></div><button className="nova-context-action" type="button" onClick={() => onNavigate('agents')}><Bot size={15} /><span>Delegate to an agent</span><ArrowUpRight size={14} /></button><button className="nova-context-action" type="button" onClick={() => onNavigate('data')}><Database size={15} /><span>Connect a source</span><ArrowUpRight size={14} /></button><button className="nova-context-action" type="button" onClick={() => onNavigate('memory')}><BrainCircuit size={15} /><span>Search memory</span><ArrowUpRight size={14} /></button></div>
        <div className="nova-context-card nova-context-card--quote"><Sparkles size={17} /><p>“The best assistant feels less like a tool and more like a clear second mind.”</p><span>OpenJarvis principle 01</span></div>
      </aside>
    </div>
  );
}

function OrionPanel({ title, note, children, className = '' }: { title: string; note?: string; children: ReactNode; className?: string }) {
  return <section className={`orion-panel ${className}`}><div className="orion-panel-title"><b>{title}</b>{note && <span>{note}</span>}</div>{children}</section>;
}

// What the orb says in each state. The user should be able to tell, without
// reading, whether Jarvis is waiting, listening, thinking or talking.
const VOICE_COPY: Record<VoiceState, { title: string; hint: string }> = {
  idle: { title: 'Conectando', hint: 'Preparando o microfone…' },
  connecting: { title: 'Conectando', hint: 'Preparando o microfone…' },
  listening: { title: 'Estou ouvindo', hint: 'Fale naturalmente comigo.' },
  user_speaking: { title: 'Ouvindo você', hint: 'Pode continuar.' },
  thinking: { title: 'Pensando', hint: 'Pode me interromper quando quiser.' },
  tool: { title: 'Executando', hint: 'Usando uma ferramenta para você.' },
  speaking: { title: 'Respondendo', hint: 'Fale para me interromper.' },
  error: { title: 'Algo deu errado', hint: 'Tente novamente em instantes.' },
};

function OrionVoiceOverlay({ open, onClose, model }: { open: boolean; onClose: () => void; model?: string }) {
  const addMessage = useAppStore((store) => store.addMessage);
  const activeId = useAppStore((store) => store.activeId);
  const createConversation = useAppStore((store) => store.createConversation);

  // Voice turns belong in the conversation like any other, so the transcript
  // is still there after the overlay closes.
  const record = useCallback((role: 'user' | 'assistant', content: string) => {
    if (!content.trim()) return;
    const conversationId = useAppStore.getState().activeId || createConversation(model);
    addMessage(conversationId, { id: `${Date.now()}-${role}`, role, content, timestamp: Date.now() });
  }, [addMessage, createConversation, model]);

  const voice = useVoiceSession({
    model,
    onUserTurn: (text) => record('user', text),
    onAssistantTurn: (text) => record('assistant', text),
  });

  const { start, stop, active } = voice;
  useEffect(() => {
    if (open) void start();
    else if (active) void stop();
  }, [open, start, stop, active]);

  if (!open) return null;

  const copy = VOICE_COPY[voice.state] ?? VOICE_COPY.listening;
  const close = () => { void voice.stop(); onClose(); };
  // The orb breathes with the microphone while listening and holds steady
  // while Jarvis talks, so amplitude always means "your voice".
  const amplitude = voice.state === 'speaking' ? 0.35 : Math.min(1, voice.level * 6);

  return <div className="orion-voice-overlay" role="presentation" onMouseDown={(event) => { if (event.currentTarget === event.target) close(); }}><div className="orion-voice-window" role="dialog" aria-modal="true" aria-labelledby="orion-voice-title"><div className="orion-voice-head"><strong id="orion-voice-title">ORION Voice</strong><p>Conversa em tempo real</p><span className="orion-voice-live"><i /> {voice.active ? 'ao vivo' : 'pronto'}</span></div><button className="orion-close" type="button" onClick={close} aria-label="Fechar modo de voz"><X size={18} /></button><div className={`orion-siri-stage orion-siri-stage--${voice.state}`} style={{ ['--orion-voice-level' as string]: amplitude.toFixed(3) }}><div className="orion-siri-halo" /><SiriWave variant={voice.state === 'tool' ? 'fluid-dots' : 'wave'} size={420} renderScale={0.72} activity={Math.max(amplitude, voice.state === 'listening' ? 0.22 : 0.12)} className="orion-siri-wave" aria-label={`Visualização da voz: ${copy.title}`} /><div className="orion-siri-state-mark" aria-hidden="true">{voice.state === 'speaking' ? <Volume2 size={16} /> : voice.muted ? <VolumeX size={16} /> : <Mic size={16} />}</div></div><div className="orion-voice-state" aria-live="polite"><h2>{voice.muted ? 'Microfone mudo' : copy.title}</h2><p>{voice.error || (voice.muted ? 'Toque no microfone para voltar.' : copy.hint)}</p>{voice.transcript && <p className="orion-voice-transcript">“{voice.transcript}”</p>}{voice.reply && <p className="orion-voice-reply">{voice.reply}</p>}</div><div className="orion-voice-actions"><button type="button" onClick={voice.cancel} aria-label="Interromper resposta" title="Interromper resposta"><Square size={16} /></button><button className="orion-mainmic" type="button" onClick={voice.toggleMute} aria-label={voice.muted ? 'Reativar microfone' : 'Silenciar microfone'} aria-pressed={voice.muted} title={voice.muted ? 'Reativar microfone' : 'Silenciar microfone'}>{voice.muted ? <VolumeX size={19} /> : <Mic size={19} />}</button><button type="button" onClick={close} aria-label="Encerrar modo de voz" title="Encerrar modo de voz"><X size={19} /></button></div></div></div>;
}

function DashboardView({ savings, health, onNavigate, onOpenVoice }: { savings: SavingsData | null; health: boolean | null; onNavigate: (page: PageKey) => void; onOpenVoice: () => void }) {
  const [telemetry, setTelemetry] = useState<Record<string, unknown> | null>(null);
  const [energy, setEnergy] = useState<Record<string, unknown> | null>(null);
  const [agents, setAgents] = useState<ManagedAgent[]>([]);
  const [command, setCommand] = useState('');
  const [chatDraft, setChatDraft] = useState('');
  const [connectors, setConnectors] = useState<ConnectorInfo[]>([]);
  const [connectorError, setConnectorError] = useState('');
  const [selectedConnector, setSelectedConnector] = useState<ConnectorInfo | null>(null);
  const [orionMessages, setOrionMessages] = useState<Array<{ role: 'user' | 'assistant'; content: string }>>([
    { role: 'assistant', content: 'Olá. Estou online. Diga o que você quer fazer ou conecte um serviço para eu agir.' },
  ]);
  const [sending, setSending] = useState(false);
  const [orionError, setOrionError] = useState('');
  const selectedModel = useAppStore((state) => state.selectedModel);
  const models = useAppStore((state) => state.models);
  const logs = useAppStore((state) => state.logEntries);
  useEffect(() => {
    Promise.allSettled([fetchTelemetry(), fetchEnergy(), fetchManagedAgents(), listConnectors()]).then(([telemetryResult, energyResult, agentsResult, connectorsResult]) => {
      if (telemetryResult.status === 'fulfilled' && telemetryResult.value && typeof telemetryResult.value === 'object') setTelemetry(telemetryResult.value as Record<string, unknown>);
      if (energyResult.status === 'fulfilled' && energyResult.value && typeof energyResult.value === 'object') setEnergy(energyResult.value as Record<string, unknown>);
      if (agentsResult.status === 'fulfilled') setAgents(agentsResult.value);
      if (connectorsResult.status === 'fulfilled') {
        setConnectors(connectorsResult.value);
        setConnectorError('');
      } else {
        setConnectorError(connectorsResult.reason instanceof Error ? connectorsResult.reason.message : 'Não foi possível carregar as conexões.');
      }
    });
  }, []);
  const askOrion = useCallback(async (rawValue: string) => {
    const content = rawValue.trim();
    if (!content || sending) return;
    const model = selectedModel || models[0]?.id || 'gpt-4o-mini';
    setSending(true);
    setOrionError('');
    setCommand('');
    setChatDraft('');
    const history = [...orionMessages, { role: 'user' as const, content }].slice(-12);
    setOrionMessages((current) => [...current, { role: 'user', content }, { role: 'assistant', content: '' }]);
    let answer = '';
    try {
      for await (const event of streamChat({ model, messages: history, stream: true, temperature: 0.7, max_tokens: 1024 })) {
        if (event.event === 'error') {
          let detail = 'O agente retornou um erro durante a execução.';
          try { detail = JSON.parse(event.data).error || detail; } catch { /* keep actionable fallback */ }
          throw new Error(detail);
        }
        try {
          const data = JSON.parse(event.data);
          const delta = data.choices?.[0]?.delta?.content || data.delta?.content || '';
          if (delta) {
            answer += delta;
            setOrionMessages((current) => current.map((message, index) => index === current.length - 1 ? { ...message, content: answer } : message));
          }
        } catch {
          // Status-only SSE frames do not contain response JSON.
        }
      }
      if (!answer) answer = 'Não recebi uma resposta do agente. Tente novamente.';
      setOrionMessages((current) => current.map((message, index) => index === current.length - 1 ? { ...message, content: answer } : message));
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : 'Não foi possível falar com o agente.';
      setOrionError(message);
      setOrionMessages((current) => current.map((item, index) => index === current.length - 1 ? { ...item, content: `Erro ao responder: ${message}` } : item));
    } finally {
      setSending(false);
    }
  }, [models, orionMessages, selectedModel, sending]);
  const cpu = Math.round(Number(telemetry?.cpu_percent ?? telemetry?.cpu ?? 23)) || 23;
  const memory = Math.round(Number(telemetry?.memory_percent ?? telemetry?.memory ?? 65)) || 65;
  const disk = Math.round(Number(telemetry?.disk_percent ?? telemetry?.disk ?? 48)) || 48;
  const activeAgent = agents.find((agent) => agent.status === 'running') || agents[0];
  const recentLogs = [...logs].reverse().slice(0, 2);
  const requestCount = formatNumber(savings?.total_calls);
  const currentActivity = activeAgent?.current_activity || (health ? 'Fale comigo ou escolha uma ação' : 'Conectando ao runtime…');
  const connectedCount = connectors.filter(connectorIsReady).length;
  const spotifyConnector = connectors.find((connector) => connector.connector_id === 'spotify');
  const connectionCards: Array<{ id: string; label: string; icon: LucideIcon }> = [
    { id: 'spotify', label: 'Spotify', icon: Music2 },
    { id: 'gmail', label: 'E-mail', icon: Mail },
    { id: 'whatsapp', label: 'WhatsApp', icon: MessageCircle },
    { id: 'gcalendar', label: 'Agenda', icon: CalendarDays },
    { id: 'obsidian', label: 'Arquivos', icon: FolderOpen },
    { id: 'weather', label: 'Clima', icon: CloudSun },
    { id: 'notion', label: 'Notion', icon: FileText },
    { id: 'hackernews', label: 'Notícias', icon: Globe2 },
  ];
  const openConnector = (connector: ConnectorInfo | undefined) => {
    if (!connector) {
      onNavigate('data');
      return;
    }
    if (connectorIsReady(connector)) onNavigate('data');
    else setSelectedConnector(connector);
  };

  return <div className="orion-dashboard">
    <div className="orion-ambient" aria-hidden="true"><i className="orion-blob orion-blob-a" /><i className="orion-blob orion-blob-b" /><i className="orion-blob orion-blob-c" /></div>
    <header className="orion-top"><div><h1>ORION</h1><p>Sistema inteligente · Tudo conectado</p></div><form className="orion-command" onSubmit={(event) => { event.preventDefault(); void askOrion(command); }}><Sparkles size={15} /><input value={command} onChange={(event) => setCommand(event.target.value)} placeholder="O que você quer fazer?" aria-label="Comando para o ORION" disabled={sending} /><kbd>⌘ K</kbd></form></header>

    <section className="orion-core" aria-label="Núcleo de voz">
      <div className="orion-orbit orion-orbit-1"><i /></div><div className="orion-orbit orion-orbit-2"><i /></div><div className="orion-orbit orion-orbit-3"><i /></div>
      <button className="orion-orb-shell" type="button" onClick={onOpenVoice} aria-label="Abrir modo de voz"><span className="orion-orb-glow" /><span className="orion-orb" /></button>
      <div className="orion-core-copy"><strong>{health ? 'ORION está pronto' : 'ORION conectando…'}</strong><span>{currentActivity}</span></div>
    </section>

    <OrionPanel title="Spotify" note={spotifyConnector?.connected ? 'conectado' : 'não conectado'} className="orion-spotify"><div className="orion-album-row"><div className="orion-album"><Music2 size={24} /></div>{spotifyConnector?.connected ? <div className="orion-song"><b>Spotify conectado</b><p>O agente já pode usar esta integração.</p><button className="orion-inline-action" type="button" onClick={() => onNavigate('data')}>Gerenciar conexão <ArrowUpRight size={12} /></button></div> : <div className="orion-song"><b>Conecte o Spotify</b><p>Depois disso, o ORION poderá controlar sua música.</p><button className="orion-inline-action" type="button" onClick={() => openConnector(spotifyConnector)}>Conectar agora <ArrowUpRight size={12} /></button></div>}</div></OrionPanel>

    <OrionPanel title="Computador" note="tempo real" className="orion-computer"><div className="orion-metrics"><div><small>CPU</small><strong>{cpu}%</strong></div><div><small>RAM</small><strong>{memory}%</strong></div><div><small>GPU</small><strong>{disk}%</strong></div></div><div className="orion-computer-wave"><svg viewBox="0 0 300 60" preserveAspectRatio="none"><path d="M0 43 C25 46 32 19 54 31 S90 49 111 27 S149 40 169 18 S207 44 229 25 S270 35 300 13" fill="none" stroke="#3986ff" strokeWidth="3" strokeLinecap="round" /></svg></div><span className="orion-computer-foot">{requestCount} solicitações · <HardDrive size={11} /> {disk}% disco</span></OrionPanel>

    <OrionPanel title="Conexões" note={connectors.length ? `${connectedCount} ativas` : 'carregando…'} className="orion-connect"><div className="orion-apps">{connectionCards.map(({ id, label, icon: Icon }) => { const connector = connectors.find((item) => item.connector_id === id); const status = connector?.connected ? 'Ativo' : connector ? 'Conectar' : 'Indisponível'; return <button className={connector?.connected ? 'is-connected' : ''} type="button" key={id} onClick={() => openConnector(connector)} disabled={!connector && connectors.length > 0} aria-label={`${label}: ${status}`}><Icon /><span>{label}</span><em>{status}</em></button>; })}<button type="button" className="orion-app-add" onClick={() => onNavigate('data')}><Plus /><span>Adicionar</span><em>Ver tudo</em></button></div>{connectorError && <div className="orion-connect-error"><Circle size={12} fill="currentColor" /> {connectorError}</div>}</OrionPanel>

    <OrionPanel title="Conversa" note={sending ? 'ORION pensando…' : health ? 'ORION online' : 'offline'} className="orion-chat"><div className="orion-chat-scroll">{orionMessages.slice(-4).map((message, index) => <div className={`orion-message orion-message-${message.role === 'user' ? 'user' : 'ai'}`} key={`${message.role}-${index}-${message.content.slice(0, 8)}`}>{message.content || '…'}</div>)}</div><form className="orion-chat-input" onSubmit={(event) => { event.preventDefault(); void askOrion(chatDraft); }}><input value={chatDraft} onChange={(event) => setChatDraft(event.target.value)} placeholder="Fale ou digite…" aria-label="Mensagem para o ORION" disabled={sending} /><button type="button" onClick={onOpenVoice} aria-label="Falar com o ORION"><Mic size={14} /></button><button type="submit" aria-label="Enviar mensagem" disabled={sending || !chatDraft.trim()}><Send size={14} /></button></form>{orionError && <small className="orion-chat-error">{orionError}</small>}{recentLogs.length > 0 && <small className="orion-chat-log">{recentLogs[0].message}</small>}</OrionPanel>

    <div className="orion-quick"><button type="button" onClick={onOpenVoice}><Mic size={12} /> Modo voz</button><button type="button" onClick={() => openConnector(connectors.find((connector) => connector.connector_id === 'spotify'))}><Music2 size={12} /> Música</button><button type="button" onClick={() => void askOrion('Abra o Chrome.')}><FolderOpen size={12} /> Abrir app</button><button type="button" onClick={() => onNavigate('agents')}><Workflow size={12} /> Automação</button></div>
    <nav className="orion-dock" aria-label="Atalhos rápidos"><button type="button" onClick={() => onNavigate('dashboard')} aria-label="Dashboard"><LayoutDashboard size={18} /></button><button type="button" onClick={onOpenVoice} aria-label="Voz"><Mic2 size={18} /></button><button type="button" onClick={() => onNavigate('data')} aria-label="Conexões"><Database size={18} /></button><button type="button" onClick={() => onNavigate('activity')} aria-label="Histórico"><Activity size={18} /></button><button type="button" onClick={() => onNavigate('agents')} aria-label="Automação"><Workflow size={18} /></button><button type="button" onClick={() => onNavigate('settings')} aria-label="Configurações"><Settings size={18} /></button><button type="button" onClick={() => onNavigate('memory')} aria-label="Memória"><BrainCircuit size={18} /></button></nav>
    {selectedConnector && <ConnectorDialog connector={selectedConnector} onClose={() => setSelectedConnector(null)} onComplete={() => { setSelectedConnector(null); void listConnectors().then(setConnectors).catch(() => undefined); }} />}
  </div>;
}

function HealthRow({ label, value, tone }: { label: string; value: string; tone: ReturnType<typeof statusTone> }) { return <div className="nova-health-row"><span><StatusDot tone={tone} pulse={tone === 'green'} />{label}</span><strong>{value}</strong></div>; }

function AgentsView() {
  const [agents, setAgents] = useState<ManagedAgent[]>([]);
  const [loading, setLoading] = useState(true);
  const [runningId, setRunningId] = useState<string | null>(null);
  const load = useCallback(() => { setLoading(true); fetchManagedAgents().then(setAgents).catch(() => setAgents([])).finally(() => setLoading(false)); }, []);
  useEffect(() => { load(); }, [load]);
  const run = async (id: string) => { setRunningId(id); try { await runManagedAgent(id); load(); } finally { setRunningId(null); } };
  return <div className="nova-page"><SectionTitle eyebrow="agent studio" title="Delegate the busywork." detail="Small autonomous workers, visible and accountable." action={<button className="nova-secondary-button" type="button" onClick={load}><RefreshCw size={15} className={loading ? 'nova-spin' : ''} /> Refresh</button>} />{loading && agents.length === 0 ? <LoadingGrid /> : agents.length === 0 ? <div className="nova-panel nova-panel-empty-state"><Bot size={24} /><h2>No agents yet</h2><p>When managed agents are configured, their state and next action will appear here.</p></div> : <div className="nova-agent-grid">{agents.map((agent) => <div className="nova-agent-card" key={agent.id}><div className="nova-agent-card-top"><span className="nova-agent-avatar"><Bot size={20} /></span><Pill tone={statusTone(agent.status)}>{agent.status.replace('_', ' ')}</Pill><button className="nova-kebab" type="button"><MoreHorizontal size={17} /></button></div><h2>{agent.name}</h2><p>{agent.summary_memory || 'No summary recorded yet.'}</p><div className="nova-agent-meta"><span><Activity size={13} /> {formatNumber(agent.total_runs)} runs</span><span><Zap size={13} /> {formatNumber(agent.total_tokens)} tokens</span></div><div className="nova-agent-footer"><span>{agent.current_activity || `Updated ${timeAgo(agent.updated_at)}`}</span><button className="nova-small-action" type="button" disabled={runningId === agent.id} onClick={() => void run(agent.id)}>{runningId === agent.id ? <RefreshCw size={13} className="nova-spin" /> : <Play size={13} />} Run now</button></div></div>)}</div>}</div>;
}

function MemoryView() {
  const [stats, setStats] = useState<Record<string, unknown> | null>(null);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Array<Record<string, unknown>>>([]);
  const [searching, setSearching] = useState(false);
  useEffect(() => { getMemoryStats().then((value) => setStats(value as unknown as Record<string, unknown>)).catch(() => setStats(null)); }, []);
  const search = async () => { if (!query.trim()) return; setSearching(true); try { const response = await searchMemory(query, 8); setResults(response as unknown as Array<Record<string, unknown>>); } catch { setResults([]); } finally { setSearching(false); } };
  return <div className="nova-page"><SectionTitle eyebrow="long-term context" title="Memory, with intention." detail="Search the knowledge Jarvis has indexed and decide what deserves to stay close." /><div className="nova-memory-hero"><div className="nova-memory-orb"><BrainCircuit size={26} /></div><div><div className="nova-overline">semantic memory</div><h2>{formatNumber(Number(stats?.documents ?? stats?.chunks ?? stats?.items))} indexed fragments</h2><p>Your private context is available when it makes an answer more useful.</p></div><div className="nova-memory-stats"><span><strong>{formatNumber(Number(stats?.documents ?? 0))}</strong> documents</span><span><strong>{formatNumber(Number(stats?.vectors ?? stats?.embeddings ?? 0))}</strong> vectors</span></div></div><div className="nova-search-box"><Search size={18} /><input value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') void search(); }} placeholder="Search your memory…" /><button type="button" onClick={() => void search()} disabled={searching}>{searching ? <RefreshCw size={15} className="nova-spin" /> : <ArrowUpRight size={16} />}</button></div><div className="nova-memory-results">{results.length === 0 ? <div className="nova-panel nova-panel-empty-state"><Library size={23} /><h2>Nothing searched yet</h2><p>Try “my plans”, “recent meetings”, or anything you want to bring back.</p></div> : results.map((result, index) => <div className="nova-memory-result" key={index}><div className="nova-memory-result-icon"><FileText size={16} /></div><div><strong>{String(result.title || result.content || result.text || 'Memory fragment')}</strong><p>{String(result.snippet || result.content || result.text || 'Indexed context')}</p></div><span>{typeof result.score === 'number' ? `${Math.round(result.score * 100)}% match` : 'indexed'}</span></div>)}</div></div>;
}

function ConnectorDialog({ connector, onClose, onComplete }: { connector: ConnectorInfo; onClose: () => void; onComplete: () => void }) {
  const meta = SOURCE_CATALOG.find((item) => item.connector_id === connector.connector_id);
  const [values, setValues] = useState<ConnectRequest>({});
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  const submit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    // Reserve the OAuth popup during the click event. Opening it only after
    // the POST resolves makes Chromium/Opera treat it as an unsolicited
    // popup and silently block the real authorization screen.
    const oauthPopup = connector.auth_type === 'oauth' ? window.open('about:blank', '_blank', 'width=600,height=700') : null;
    setSaving(true);
    setError('');
    try {
      const response = await connectSource(connector.connector_id, values);
      if (response.status === 'oauth_required' && response.oauth_start) {
        if (!oauthPopup) throw new Error('O pop-up de autorização foi bloqueado. Permita pop-ups para concluir a conexão.');
        await startServerOAuth(connector.connector_id, response.oauth_start, oauthPopup);
      } else if (response.status === 'pending') {
        throw new Error(connector.auth_type === 'oauth'
          ? 'A conexão ficou pendente. Configure as credenciais OAuth desta integração e tente novamente.'
          : 'A conexão ficou pendente. Verifique a configuração desta fonte e tente novamente.');
      } else if (oauthPopup && !oauthPopup.closed) {
        oauthPopup.close();
      }
      onComplete();
      onClose();
    } catch (caught) {
      if (oauthPopup && !oauthPopup.closed) oauthPopup.close();
      setError(caught instanceof Error ? caught.message : 'Could not connect this source.');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="nova-dialog-backdrop" role="presentation" onMouseDown={(event) => { if (event.currentTarget === event.target) onClose(); }}>
      <section className="nova-dialog" role="dialog" aria-modal="true" aria-labelledby="connector-dialog-title">
        <div className="nova-dialog-heading">
          <div><div className="nova-overline">conectar fonte</div><h2 id="connector-dialog-title">{connector.display_name}</h2><p>{meta?.description || 'Dê acesso ao ORION a esta fonte.'}</p></div>
          <IconButton label="Fechar diálogo" onClick={onClose}><X size={17} /></IconButton>
        </div>
        {meta?.steps?.[0] && <div className="nova-dialog-tip"><FileText size={15} /><span>{meta.steps[0].label}</span></div>}
        <form onSubmit={submit}>
          {(meta?.inputFields || []).map((field) => (
            <label className="nova-dialog-field" key={field.name}><span>{field.name}</span><input type={field.type || 'text'} placeholder={field.placeholder} value={String(values[field.name as keyof ConnectRequest] || '')} onChange={(event) => setValues((current) => ({ ...current, [field.name]: event.target.value }))} required /></label>
          ))}
          {error && <div className="nova-inline-error"><Circle size={12} fill="currentColor" /> {error}</div>}
          <div className="nova-dialog-actions"><button className="nova-secondary-button" type="button" onClick={onClose}>Cancelar</button><button className="nova-primary-button" type="submit" disabled={saving}>{saving ? <RefreshCw size={14} className="nova-spin" /> : <Check size={14} />} {saving ? 'Conectando…' : 'Conectar fonte'}</button></div>
        </form>
      </section>
    </div>
  );
}

function DataView() {
  const [connectors, setConnectors] = useState<ConnectorInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [working, setWorking] = useState<string | null>(null);
  const [selectedConnector, setSelectedConnector] = useState<ConnectorInfo | null>(null);
  const [error, setError] = useState('');
  const load = useCallback(() => { setLoading(true); setError(''); listConnectors().then(setConnectors).catch((caught) => { setConnectors([]); setError(caught instanceof Error ? caught.message : 'Não foi possível carregar as conexões.'); }).finally(() => setLoading(false)); }, []);
  useEffect(() => { load(); }, [load]);
  const sync = async (id: string) => { setWorking(id); setError(''); try { await triggerSync(id); const deadline = Date.now() + 30000; let status = await getSyncStatus(id); while (status.state === 'syncing' && Date.now() < deadline) { await new Promise((resolve) => window.setTimeout(resolve, 750)); status = await getSyncStatus(id); } if (status.state === 'error') setError(status.error || 'A sincronização falhou.'); } catch (caught) { setError(caught instanceof Error ? caught.message : 'A sincronização falhou.'); } finally { setWorking(null); load(); } };
  const disconnect = async (id: string) => { setWorking(id); setError(''); try { await disconnectSource(id); } catch (caught) { setError(caught instanceof Error ? caught.message : 'Não foi possível desconectar a fonte.'); } finally { setWorking(null); load(); } };
  return <div className="nova-page"><SectionTitle eyebrow="connected context" title="Bring your world closer." detail="Connect only what you want Jarvis to understand." action={<button className="nova-secondary-button" type="button" onClick={load}><RefreshCw size={15} className={loading ? 'nova-spin' : ''} /> Refresh</button>} />{error && <div className="nova-inline-error nova-page-error"><Circle size={12} fill="currentColor" /> {error}</div>}<div className="nova-source-summary"><div><Database size={18} /><span><strong>{connectors.filter(connectorIsReady).length}</strong> verified sources</span></div><div><FileText size={18} /><span><strong>{formatNumber(connectors.reduce((sum, connector) => sum + (connector.chunks || 0), 0))}</strong> indexed fragments</span></div><button className="nova-primary-button" type="button" onClick={() => setSelectedConnector(connectors.find((connector) => !connector.configured && !connector.connected) || connectors[0] || null)}><Plus size={15} /> Add source</button></div>{loading && connectors.length === 0 ? <LoadingGrid /> : <div className="nova-source-grid">{connectors.map((connector) => { const health = connector.health_status || (connector.connected ? 'Connected' : 'Configuration Missing'); const configured = connector.configured ?? connector.connected; return <div className={`nova-source-card ${connectorIsReady(connector) ? 'is-connected' : ''}`} key={connector.connector_id}><div className="nova-source-card-top"><div className="nova-source-icon"><Database size={18} /></div><Pill tone={statusTone(health)}>{health}</Pill></div><h2>{connector.display_name}</h2><p>{connector.auth_type === 'oauth' ? 'OAuth connection' : 'Local connector'} · {formatNumber(connector.chunks)} fragments</p>{connector.health_error && <p className="nova-source-health-error">{connector.health_error}</p>}<div className="nova-source-actions">{configured ? <><button className="nova-small-action" type="button" disabled={working === connector.connector_id} onClick={() => void sync(connector.connector_id)}>{working === connector.connector_id ? <RefreshCw size={13} className="nova-spin" /> : <RefreshCw size={13} />} Sync</button><button className="nova-text-button danger" type="button" onClick={() => void disconnect(connector.connector_id)}>Disconnect</button></> : <button className="nova-small-action" type="button" onClick={() => setSelectedConnector(connector)}>Connect <ArrowUpRight size={13} /></button>}</div></div>; })}</div>}{selectedConnector && <ConnectorDialog connector={selectedConnector} onClose={() => setSelectedConnector(null)} onComplete={load} />}</div>;
}

function ActivityView() {
  const logs = useAppStore((s) => s.logEntries);
  return <div className="nova-page"><SectionTitle eyebrow="observability" title="See what Jarvis sees." detail="A transparent trail of models, tools and conversations." action={<Pill tone="green">recording</Pill>} /><div className="nova-panel nova-log-panel">{logs.length === 0 ? <div className="nova-panel-empty-state"><Activity size={23} /><h2>Quiet for now</h2><p>Activity will appear here when you start a conversation or run an agent.</p></div> : <div className="nova-log-list">{[...logs].reverse().map((log, index) => <div className="nova-log-row" key={`${log.timestamp}-${index}`}><span className={`nova-log-level nova-log-level--${log.level}`}><StatusDot tone={log.level === 'error' ? 'red' : log.level === 'warn' ? 'amber' : 'cyan'} />{log.level}</span><span className="nova-log-category">{log.category}</span><p>{log.message}</p><time>{timeAgo(log.timestamp)}</time></div>)}</div>}</div></div>;
}

function SettingsView() {
  const settings = useAppStore((s) => s.settings);
  const updateSettings = useAppStore((s) => s.updateSettings);
  const serverInfo = useAppStore((s) => s.serverInfo);
  const models = useAppStore((s) => s.models);
  return <div className="nova-page"><SectionTitle eyebrow="workspace control" title="Make it yours." detail="Tune the way Jarvis sounds, thinks and connects." /><div className="nova-settings-layout"><div className="nova-settings-nav"><span className="is-active">Appearance</span><span>Inference</span><span>Voice</span><span>Privacy</span></div><div className="nova-settings-content"><div className="nova-settings-group"><div className="nova-settings-heading"><div><div className="nova-overline">appearance</div><h2>A space that feels like yours</h2></div><Sparkles size={18} /></div><div className="nova-setting-row"><div><strong>Theme</strong><p>Choose how the command center looks.</p></div><div className="nova-segmented">{(['system', 'light', 'dark'] as const).map((theme) => <button type="button" key={theme} className={settings.theme === theme ? 'is-active' : ''} onClick={() => updateSettings({ theme })}>{theme}</button>)}</div></div><div className="nova-setting-row"><div><strong>Interface scale</strong><p>Adjust the density of the workspace.</p></div><select value={settings.fontSize} onChange={(event) => updateSettings({ fontSize: event.target.value as 'small' | 'default' | 'large' })}><option value="small">Compact</option><option value="default">Comfortable</option><option value="large">Spacious</option></select></div></div><div className="nova-settings-group"><div className="nova-settings-heading"><div><div className="nova-overline">inference</div><h2>Your local mind</h2></div><TerminalSquare size={18} /></div><div className="nova-setting-row"><div><strong>Default model</strong><p>{models.length} models available from the active server.</p></div><select value={settings.defaultModel || ''} onChange={(event) => updateSettings({ defaultModel: event.target.value })}><option value="">Use best available</option>{models.map((model) => <option value={model.id} key={model.id}>{model.id}</option>)}</select></div><div className="nova-setting-row"><div><strong>API endpoint</strong><p>Current connection: {serverInfo?.engine || 'not connected'}</p></div><span className="nova-setting-value">{settings.apiUrl || 'local runtime'}</span></div></div><div className="nova-settings-group"><div className="nova-settings-heading"><div><div className="nova-overline">voice</div><h2>Speak naturally</h2></div><Mic size={18} /></div><div className="nova-setting-row"><div><strong>Voice input</strong><p>Allow Jarvis to listen when you choose.</p></div><button type="button" className={`nova-switch ${settings.speechEnabled ? 'is-on' : ''}`} onClick={() => updateSettings({ speechEnabled: !settings.speechEnabled })}><span /></button></div></div></div></div></div>;
}

function LoadingGrid() { return <div className="nova-loading-grid">{[1, 2, 3].map((item) => <div className="nova-skeleton" key={item} />)}</div>; }

export default function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const [page, setPage] = useState<PageKey>(pageFromPath(location.pathname));
  const [mobileNav, setMobileNav] = useState(false);
  const [launchVoice, setLaunchVoice] = useState(false);
  const [voiceOverlay, setVoiceOverlay] = useState(false);
  const [health, setHealth] = useState<boolean | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [recommendedModel, setRecommendedModel] = useState('');
  const setStoreModels = useAppStore((s) => s.setModels);
  const setModelsLoading = useAppStore((s) => s.setModelsLoading);
  const selectedModel = useAppStore((s) => s.selectedModel);
  const storeModels = useAppStore((s) => s.models);
  const setSelectedModel = useAppStore((s) => s.setSelectedModel);
  const serverInfo = useAppStore((s) => s.serverInfo);
  const setServerInfo = useAppStore((s) => s.setServerInfo);
  const savings = useAppStore((s) => s.savings);
  const setSavings = useAppStore((s) => s.setSavings);
  const settings = useAppStore((s) => s.settings);
  const loadConversations = useAppStore((s) => s.loadConversations);

  useEffect(() => { loadConversations(); }, [loadConversations]);
  useEffect(() => {
    let alive = true;
    setModelsLoading(true);
    Promise.allSettled([fetchModels(), fetchServerInfo(), fetchSavings(), checkHealth(), fetchRecommendedModel()]).then(([modelsResult, serverResult, savingsResult, healthResult, recommendedResult]) => {
      if (!alive) return;
      if (modelsResult.status === 'fulfilled') {
        const recommendation = recommendedResult.status === 'fulfilled' ? recommendedResult.value.model : '';
        const serverModels = modelsResult.value;
        const mergedModels = recommendation && !serverModels.some((model) => model.id === recommendation)
          ? [{ id: recommendation, object: 'model', created: Date.now(), owned_by: 'openai' }, ...serverModels]
          : serverModels;
        setModels(mergedModels);
        setStoreModels(mergedModels);
        if (recommendation) {
          setRecommendedModel(recommendation);
          setSelectedModel(recommendation);
        }
      }
      if (serverResult.status === 'fulfilled') setServerInfo(serverResult.value);
      if (savingsResult.status === 'fulfilled') setSavings(savingsResult.value);
      if (healthResult.status === 'fulfilled') setHealth(healthResult.value);
      setModelsLoading(false);
    });
    return () => { alive = false; };
  }, [setModelsLoading, setSelectedModel, setServerInfo, setSavings, setStoreModels]);
  useEffect(() => { setPage(pageFromPath(location.pathname)); }, [location.pathname]);
  useEffect(() => { document.documentElement.dataset.novaTheme = settings.theme; }, [settings.theme]);

  const allModels = storeModels.length ? storeModels : models;
  const availableModels = recommendedModel
    ? allModels.filter((model) => model.id === recommendedModel || model.owned_by?.toLowerCase() === 'openai')
    : allModels;
  const go = (next: PageKey) => { setLaunchVoice(false); setVoiceOverlay(false); setPage(next); setMobileNav(false); navigate(PAGE_PATHS[next]); };
  // Keep voice attached to the ORION dashboard. The overlay is a dashboard
  // interaction, so opening it must not silently replace the new screen with
  // the legacy chat route underneath.
  const openVoice = () => { setLaunchVoice(false); setVoiceOverlay(true); setMobileNav(false); };
  const closeVoice = () => { setLaunchVoice(false); setVoiceOverlay(false); setPage('dashboard'); setMobileNav(false); navigate(PAGE_PATHS.dashboard); };
  useEffect(() => {
    const handleCommandShortcut = (event: globalThis.KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        go('chat');
        window.setTimeout(() => document.getElementById('nova-composer')?.focus(), 0);
      }
    };
    window.addEventListener('keydown', handleCommandShortcut);
    return () => window.removeEventListener('keydown', handleCommandShortcut);
  }, [location.pathname]);
  const pageTitle = NAV_ITEMS.find((item) => item.id === page)?.label || 'Conversation';
  const content = page === 'chat' ? <ChatView onNavigate={go} autoStartVoice={launchVoice} /> : page === 'dashboard' ? <DashboardView savings={savings} health={health} onNavigate={go} onOpenVoice={openVoice} /> : page === 'agents' ? <AgentsView /> : page === 'memory' ? <MemoryView /> : page === 'data' ? <DataView /> : page === 'activity' ? <ActivityView /> : <SettingsView />;

  return <><div className="nova-app nova-app--orion"><div className="nova-background" aria-hidden="true"><span /><span /><span /></div><header className="nova-topbar"><IconButton label="Open navigation" className="nova-mobile-menu" onClick={() => setMobileNav(!mobileNav)}><Menu size={19} /></IconButton><Brand /><div className="nova-topbar-center"><span className="nova-breadcrumb">workspace <ChevronDown size={13} /></span><span className="nova-breadcrumb-separator">/</span><strong>{pageTitle}</strong></div><div className="nova-topbar-right"><div className="nova-server-status"><StatusDot tone={health ? 'green' : health === false ? 'red' : 'amber'} pulse={health === true} /><span>{health ? 'online' : health === false ? 'offline' : 'connecting'}</span></div><div className="nova-model-select"><TerminalSquare size={14} /><select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)} aria-label="Choose model"><option value="">Choose model</option>{availableModels.map((model) => <option value={model.id} key={model.id}>{model.id}</option>)}</select><ChevronDown size={13} /></div><ApprovalBell /><IconButton label="Settings" onClick={() => go('settings')}><Settings size={17} /></IconButton><div className="nova-user-badge">G</div></div></header><div className="nova-shell"><aside className={`nova-rail ${mobileNav ? 'is-open' : ''}`}><div className="nova-rail-head"><Brand /><IconButton label="Close navigation" className="nova-mobile-menu" onClick={() => setMobileNav(false)}><X size={17} /></IconButton></div><div className="nova-rail-section"><span className="nova-rail-label">workspace</span>{NAV_ITEMS.map((item) => { const Icon = item.icon; return <button className={`nova-nav-item ${page === item.id ? 'is-active' : ''}`} type="button" key={item.id} onClick={() => go(item.id)}><span className="nova-nav-icon"><Icon size={17} /></span><span>{item.label}</span><em>{item.hint}</em></button>; })}</div><div className="nova-rail-section nova-rail-bottom"><span className="nova-rail-label">system</span><div className="nova-rail-health"><StatusDot tone={health ? 'green' : 'amber'} pulse={health === true} /><div><strong>{health ? 'Runtime ready' : 'Starting runtime'}</strong><span>{serverInfo?.engine || 'local route'}</span></div></div><button className="nova-nav-item" type="button" onClick={() => go('settings')}><span className="nova-nav-icon"><Settings size={17} /></span><span>Settings</span></button><div className="nova-rail-footer"><span>OJ / 01</span><span>v1.0</span></div></div></aside>{mobileNav && <button className="nova-mobile-scrim" type="button" aria-label="Close navigation" onClick={() => setMobileNav(false)} />}</div><main className="nova-main"><div className="nova-page-transition" key={page}>{content}</div></main></div><OrionVoiceOverlay open={voiceOverlay} onClose={closeVoice} model={selectedModel} /></>;
}
