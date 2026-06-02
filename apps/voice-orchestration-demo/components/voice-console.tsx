"use client";

import {
  Activity,
  Bot,
  Cable,
  CircleStop,
  Headphones,
  Mic,
  Play,
  RadioTower,
  RotateCcw,
  Sparkles,
  Wifi,
  Zap,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { createPipecatClient } from "../lib/pipecat-live";
import {
  AgentName,
  AgentState,
  VoiceMode,
  initialState,
  mockEvents,
  orchestrationReducer,
} from "../lib/orchestration-state";

type RuntimeClient = Awaited<ReturnType<typeof createPipecatClient>>;

const connectEndpoint =
  process.env.NEXT_PUBLIC_CONNECT_ENDPOINT ?? "http://localhost:7860/connect";

const statusCopy = {
  idle: "Standby",
  connecting: "Linking",
  listening: "Listening",
  thinking: "Thinking",
  speaking: "Speaking",
  error: "Fault",
};

function AgentPanel({ agent }: { agent: AgentState }) {
  const phaseClass = `agent-card agent-card--${agent.phase}`;
  const confidence = Math.round(agent.confidence * 100);

  return (
    <section className={phaseClass} aria-label={agent.title}>
      <div className="agent-card__header">
        <div>
          <p className="agent-card__eyebrow">{agent.name}</p>
          <h2>{agent.title}</h2>
        </div>
        <span className="agent-card__status">{agent.phase.replace("_", " ")}</span>
      </div>
      <div className="agent-card__body">
        <p className="agent-card__task">{agent.task}</p>
        <p className="agent-card__feedback">{agent.feedback}</p>
      </div>
      <div className="confidence" aria-label={`${agent.title} confidence ${confidence}%`}>
        <span>Confidence</span>
        <div className="confidence__track">
          <div style={{ width: `${confidence}%` }} />
        </div>
        <strong>{confidence}%</strong>
      </div>
    </section>
  );
}

function VoiceCore({
  status,
  label,
}: {
  status: keyof typeof statusCopy;
  label: string;
}) {
  return (
    <section className={`voice-core voice-core--${status}`} aria-label="Voice status">
      <div className="voice-core__ring voice-core__ring--outer" />
      <div className="voice-core__ring voice-core__ring--middle" />
      <div className="voice-core__center">
        <Bot size={40} strokeWidth={1.5} />
        <p>{statusCopy[status]}</p>
      </div>
      <div className="waveform" aria-hidden="true">
        {Array.from({ length: 28 }).map((_, index) => (
          <span key={index} style={{ animationDelay: `${index * 42}ms` }} />
        ))}
      </div>
      <p className="voice-core__label">{label}</p>
    </section>
  );
}

function Transcript({
  transcript,
}: {
  transcript: Array<{ id: string; role: string; text: string; final: boolean }>;
}) {
  return (
    <section className="transcript" aria-label="Conversation transcript">
      <div className="section-title">
        <Headphones size={16} />
        <span>Transcript</span>
      </div>
      <div className="transcript__list">
        {transcript.length === 0 ? (
          <p className="empty-state">No voice turns yet.</p>
        ) : (
          transcript.map((item) => (
            <article key={item.id} className={`transcript__item transcript__item--${item.role}`}>
              <span>{item.role}</span>
              <p>{item.text}</p>
            </article>
          ))
        )}
      </div>
    </section>
  );
}

function Timeline({
  items,
}: {
  items: Array<{ id: string; agent?: AgentName; label: string; detail: string }>;
}) {
  return (
    <section className="timeline" aria-label="Agent activity timeline">
      <div className="section-title">
        <Activity size={16} />
        <span>Agent Activity</span>
      </div>
      <div className="timeline__list">
        {items.length === 0 ? (
          <p className="empty-state">Specialist activity will appear here.</p>
        ) : (
          items.map((item) => (
            <article key={item.id} className="timeline__item">
              <span className={item.agent ? `timeline__dot timeline__dot--${item.agent}` : "timeline__dot"} />
              <div>
                <strong>{item.label}</strong>
                <p>{item.detail}</p>
              </div>
            </article>
          ))
        )}
      </div>
    </section>
  );
}

function ModeSwitch({
  mode,
  onModeChange,
}: {
  mode: VoiceMode;
  onModeChange: (mode: VoiceMode) => void;
}) {
  return (
    <fieldset className="mode-switch" aria-label="Demo mode">
      <button
        type="button"
        className={mode === "mock" ? "is-active" : ""}
        onClick={() => onModeChange("mock")}
      >
        Mock
      </button>
      <button
        type="button"
        className={mode === "live" ? "is-active" : ""}
        onClick={() => onModeChange("live")}
      >
        Live
      </button>
    </fieldset>
  );
}

export function VoiceConsole() {
  const [state, dispatch] = useReducer(orchestrationReducer, initialState);
  const [isRunningMock, setIsRunningMock] = useState(false);
  const clientRef = useRef<RuntimeClient | null>(null);

  const runMock = useCallback(() => {
    dispatch({ type: "reset-demo" });
    dispatch({ type: "set-voice-status", status: "listening" });
    setIsRunningMock(true);
    mockEvents.forEach((event, index) => {
      window.setTimeout(() => {
        dispatch(event);
        if (index === mockEvents.length - 1) {
          window.setTimeout(() => {
            dispatch({ type: "set-voice-status", status: "listening" });
            setIsRunningMock(false);
          }, 1400);
        }
      }, 650 + index * 820);
    });
  }, []);

  const connectLive = useCallback(async () => {
    dispatch({ type: "set-mode", mode: "live" });
    dispatch({ type: "set-voice-status", status: "connecting" });
    try {
      const client = await createPipecatClient(dispatch);
      clientRef.current = client;
      await client.startBotAndConnect({ endpoint: connectEndpoint });
    } catch (error) {
      dispatch({
        type: "error",
        message: error instanceof Error ? error.message : "Failed to connect",
      });
    }
  }, []);

  const disconnectClient = useCallback(async () => {
    await clientRef.current?.disconnect();
    clientRef.current = null;
  }, []);

  const disconnectLive = useCallback(async () => {
    await disconnectClient();
    dispatch({ type: "set-voice-status", status: "idle" });
    dispatch({ type: "set-transport", state: "disconnected" });
  }, [disconnectClient]);

  useEffect(() => {
    return () => {
      void disconnectClient();
    };
  }, [disconnectClient]);

  const health = useMemo(
    () => [
      { label: "Daily", value: state.mode === "mock" ? "mock" : state.transportState, icon: Wifi },
      { label: "Deepgram", value: state.mode === "mock" ? "simulated" : "server", icon: RadioTower },
      { label: "GLM", value: state.mode === "mock" ? "simulated" : "server", icon: Sparkles },
      { label: "Cartesia", value: state.mode === "mock" ? "simulated" : "server", icon: Zap },
    ],
    [state.mode, state.transportState],
  );

  const setMode = useCallback(
    (mode: VoiceMode) => {
      if (state.mode === "live" && mode === "mock") {
        void disconnectLive();
      }
      dispatch({ type: "set-mode", mode });
    },
    [disconnectLive, state.mode],
  );

  return (
    <main className="console-shell">
      <div className="grid-plane" aria-hidden="true" />
      <header className="top-rail">
        <div className="brand-lockup">
          <div className="brand-lockup__mark">
            <Sparkles size={18} />
          </div>
          <div>
            <p>Acme Corp</p>
            <h1>Orchestrator Console</h1>
          </div>
        </div>
        <div className="health-strip">
          {health.map((item) => {
            const Icon = item.icon;
            return (
              <span key={item.label}>
                <Icon size={14} />
                {item.label}: <strong>{item.value}</strong>
              </span>
            );
          })}
        </div>
        <ModeSwitch mode={state.mode} onModeChange={setMode} />
      </header>

      <section className="control-band">
        <div>
          <p className="control-band__eyebrow">Primary Agent</p>
          <h2>Single voice, specialist reasoning underneath.</h2>
        </div>
        <div className="control-band__actions">
          {state.mode === "mock" ? (
            <button
              type="button"
              className="command-button command-button--primary"
              onClick={runMock}
              disabled={isRunningMock}
            >
              <Play size={17} />
              {isRunningMock ? "Running" : "Run Mock"}
            </button>
          ) : (
            <>
              <button type="button" className="command-button command-button--primary" onClick={connectLive}>
                <Mic size={17} />
                Connect
              </button>
              <button type="button" className="command-button" onClick={disconnectLive}>
                <CircleStop size={17} />
                Disconnect
              </button>
            </>
          )}
          <button type="button" className="command-button" onClick={() => dispatch({ type: "reset-demo" })}>
            <RotateCcw size={17} />
            Reset
          </button>
        </div>
      </section>

      <section className="orchestration-grid">
        <AgentPanel agent={state.agents.sales} />
        <VoiceCore status={state.voiceStatus} label={state.jobLabel} />
        <AgentPanel agent={state.agents.support} />
      </section>

      <section className="lower-grid">
        <Transcript transcript={state.transcript} />
        <Timeline items={state.timeline} />
      </section>

      {state.error ? (
        <aside className="fault-panel" role="alert">
          <Cable size={16} />
          {state.error}
        </aside>
      ) : null}
    </main>
  );
}
