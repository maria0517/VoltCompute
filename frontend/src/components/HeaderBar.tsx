import React from "react";
import { formatDateTime, formatMonth, formatShortDateTime, nf } from "../lib/format";
import { TimelineInfo } from "../types/telemetry";

export type ViewId = "console" | "map";

interface HeaderBarProps {
  timeline: TimelineInfo;
  liveMae: number;
  activeView: ViewId;
  autoplay: boolean;
  connected: boolean;
  onSelectView: (view: ViewId) => void;
  onStepBack: () => void;
  onNextHour: () => void;
  onTogglePlay: () => void;
  onSliderChange: (step: number) => void;
}

export const HeaderBar: React.FC<HeaderBarProps> = ({
  timeline,
  liveMae,
  activeView,
  autoplay,
  connected,
  onSelectView,
  onStepBack,
  onNextHour,
  onTogglePlay,
  onSliderChange,
}) => (
  <>
    <header className="app-header">
      <div className="brand">
        <div className="brand-mark">V</div>
        <div>
          <div className="brand-name">VoltCompute</div>
          <div className="brand-sub">PPC Renewables · Dispecerat surplus solar</div>
        </div>
      </div>

      <nav className="tabs" aria-label="Secțiuni">
        <button
          type="button"
          className={`tab ${activeView === "console" ? "active" : ""}`}
          onClick={() => onSelectView("console")}
        >
          Panou de control
        </button>
        <button
          type="button"
          className={`tab ${activeView === "map" ? "active" : ""}`}
          onClick={() => onSelectView("map")}
        >
          Harta rețelei
        </button>
      </nav>

      <div className="header-meta">
        <span className="chip">
          Perioadă: <strong>{formatMonth(timeline.month_label)}</strong>
        </span>
        <span className="chip">
          Eroare model (24h): <strong className="num">±{nf(liveMae)} MW</strong>
        </span>
        <span className="chip">
          <span className={`chip-dot ${connected ? "" : "off"}`} />
          {connected ? "Conectat la server" : "Conexiune întreruptă"}
        </span>
      </div>
    </header>

    <div className="transport">
      <button type="button" className="btn" onClick={onStepBack}>
        ◀ Ora anterioară
      </button>
      <button type="button" className="btn btn-primary" onClick={onNextHour}>
        Ora următoare ▶
      </button>
      <button type="button" className="btn" onClick={onTogglePlay}>
        {autoplay ? "⏸ Oprește redarea" : "▶ Redare automată"}
      </button>

      <div className="timeline">
        <div className="timeline-top">
          <span className="timeline-now">{formatDateTime(timeline.current_timestamp)}</span>
          <span className="timeline-range">
            {formatShortDateTime(timeline.slider_start_label)} →{" "}
            {formatShortDateTime(timeline.slider_end_label)}
          </span>
        </div>
        <input
          type="range"
          className="range"
          aria-label="Selectează ora"
          min={0}
          max={Math.max(0, timeline.hours_in_month - 1)}
          value={timeline.current_step}
          onChange={(e) => onSliderChange(Number(e.target.value))}
        />
      </div>
    </div>
  </>
);
