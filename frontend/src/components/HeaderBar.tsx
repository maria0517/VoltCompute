import React from "react";
import { Map, Activity, Play, Square, SkipForward, ChevronLeft } from "lucide-react";
import { TimelineInfo } from "../types/telemetry";

interface HeaderBarProps {
  timeline: TimelineInfo;
  liveMae: number;
  activeView: "console" | "map";
  autoplay: boolean;
  onSelectView: (view: "console" | "map") => void;
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
  onSelectView,
  onStepBack,
  onNextHour,
  onTogglePlay,
  onSliderChange,
}) => {
  return (
    <>
      <div className="vc-header-row">
        <div>
          <div className="vc-title-wrap">
            <span className="vc-live-dot" />
            <span className="vc-title">
              {activeView === "console"
                ? "VOLTCOMPUTE // PPC GRID DISPATCH"
                : "VOLTCOMPUTE // SCADA NATIONAL POWER FLOW MAP"}
            </span>
          </div>
          <div className="vc-sub" style={{ marginTop: 4 }}>
            LUNA CURENTĂ: {timeline.month_label} (Windows API) · {timeline.hours_in_month}h în lună
            (Luna #{timeline.ring_cycle + 1}) · MAE Live (24h): ±{liveMae.toFixed(2)} MW
          </div>
        </div>

        <div className="vc-view-tabs">
          <button
            type="button"
            className={`vc-btn ${activeView === "console" ? "vc-btn-primary" : ""}`}
            onClick={() => onSelectView("console")}
          >
            <Activity size={16} />
            Consola Principală
          </button>
          <button
            type="button"
            className={`vc-btn ${activeView === "map" ? "vc-btn-cyan" : ""}`}
            onClick={() => onSelectView("map")}
          >
            <Map size={16} />
            Harta Rețea SCADA (400 kV)
          </button>
        </div>
      </div>

      <div className="vc-timeline-bar">
        <button type="button" className="vc-btn" onClick={onStepBack}>
          <ChevronLeft size={16} />
          Pas Înapoi
        </button>

        <button type="button" className="vc-btn vc-btn-primary" onClick={onNextHour}>
          <SkipForward size={16} />
          Simulează Ora Următoare
        </button>

        <button
          type="button"
          className={`vc-btn ${autoplay ? "vc-btn-danger" : ""}`}
          onClick={onTogglePlay}
        >
          {autoplay ? <Square size={15} /> : <Play size={15} />}
          {autoplay ? "Stop Live Feed" : "Play Live Feed"}
        </button>

        <div className="vc-slider-box">
          <div className="vc-slider-top">
            <span style={{ color: "#8E95A5" }}>{timeline.slider_start_label}</span>
            <span className="vc-slider-current">{timeline.current_timestamp}</span>
            <span style={{ color: "#8E95A5" }}>{timeline.slider_end_label}</span>
          </div>
          <input
            type="range"
            className="vc-range-input"
            min={0}
            max={Math.max(0, timeline.hours_in_month - 1)}
            value={timeline.current_step}
            onChange={(e) => onSliderChange(Number(e.target.value))}
          />
        </div>
      </div>
    </>
  );
};
