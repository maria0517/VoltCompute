import React from "react";
import { Tone } from "./StatusBadge";

interface MetricCardProps {
  label: string;
  value: string;
  unit?: string;
  note?: string;
  tone?: Tone;
  meter?: { value: number; max: number; tone?: Tone };
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  unit,
  note,
  tone = "neutral",
  meter,
}) => {
  const pct = meter ? Math.min(100, Math.max(0, (meter.value / meter.max) * 100)) : 0;
  return (
    <div className="kpi">
      <div className="kpi-label">{label}</div>
      <div className={`kpi-value tone-${tone}`}>
        {value}
        {unit && <span className="kpi-unit">{unit}</span>}
      </div>
      {meter && (
        <div className="meter" aria-hidden>
          <div className={`meter-fill meter-${meter.tone ?? "info"}`} style={{ width: `${pct}%` }} />
        </div>
      )}
      {note && <div className="kpi-note">{note}</div>}
    </div>
  );
};
