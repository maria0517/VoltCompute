import React, { useState } from "react";
import { nf } from "../lib/format";
import {
  ChartPoint,
  DashboardKPIs,
  ExplainabilityInfo,
  TimelineInfo,
} from "../types/telemetry";
import { Card } from "./ui/Card";
import { MetricCard } from "./ui/MetricCard";

interface PpcDispatchPanelProps {
  timeline: TimelineInfo;
  kpis: DashboardKPIs;
  explainability: ExplainabilityInfo;
  chartPoints: ChartPoint[];
  yAxisMax: number;
}

const FEATURE_LABELS: Record<string, string> = {
  solar_irradiance_wm2: "Radiație solară",
  clear_sky_proxy_mw: "Producție teoretică (cer senin)",
  ambient_temp_celsius: "Temperatură ambientală",
  day_ahead_price_eur: "Preț piață (PZU)",
  hour_sin: "Ciclu zilnic (sin)",
  hour_cos: "Ciclu zilnic (cos)",
  thermal_derating_factor: "Pierderi termice panouri",
  net_load_factor_mw: "Consum local",
};

const W = 960;
const H = 300;
const PAD = { l: 44, r: 16, t: 14, b: 28 };

export const PpcDispatchPanel: React.FC<PpcDispatchPanelProps> = ({
  timeline,
  kpis,
  explainability,
  chartPoints,
  yAxisMax,
}) => {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const plotW = W - PAD.l - PAD.r;
  const plotH = H - PAD.t - PAD.b;
  const maxVal = Math.max(16, yAxisMax);
  const n = Math.max(1, chartPoints.length - 1);

  const xOf = (i: number) => PAD.l + (i / n) * plotW;
  const yOf = (mw: number) => PAD.t + plotH - (Math.min(maxVal, Math.max(0, mw)) / maxVal) * plotH;

  const limit = kpis.grid_limit_mw;
  const limitY = yOf(limit);
  const nowIdx = chartPoints.findIndex((p) => p.is_current);

  const forecastPath = chartPoints
    .map((p, i) => `${i === 0 ? "M" : "L"}${xOf(i).toFixed(1)},${yOf(p.predicted_injection_mw).toFixed(1)}`)
    .join(" ");

  const measured = chartPoints
    .map((p, i) => (p.real_injection_mw !== null ? { i, v: p.real_injection_mw } : null))
    .filter((x): x is { i: number; v: number } => x !== null);
  const measuredPath = measured
    .map((m, k) => `${k === 0 ? "M" : "L"}${xOf(m.i).toFixed(1)},${yOf(m.v).toFixed(1)}`)
    .join(" ");

  const ticks = [0, 4, 8, 12, 16].filter((t) => t <= maxVal && Math.abs(t - limit) > 1.2);

  const hovered = hoverIdx !== null ? chartPoints[hoverIdx] : chartPoints[nowIdx] ?? chartPoints[0];

  const shap = Object.entries(explainability.top_shap_features).slice(0, 5);
  const maxShap = Math.max(0.1, ...shap.map(([, v]) => Math.abs(v)));

  const overLimit = kpis.actual_injection_mw > limit;
  const surplusTone = kpis.actual_surplus_mw > 0.1 ? "warning" : "neutral";

  const checks = [
    {
      ok: kpis.surplus_mw > explainability.min_surplus_threshold_mw,
      text: (
        <>
          Surplus prognozat <strong>{nf(kpis.surplus_mw)} MW</strong> (necesar peste{" "}
          {nf(explainability.min_surplus_threshold_mw, 1)} MW)
        </>
      ),
    },
    {
      ok: kpis.day_ahead_price_eur < explainability.max_pzu_price_eur,
      text: (
        <>
          Preț piață <strong>{nf(kpis.day_ahead_price_eur)} €/MWh</strong> (necesar sub{" "}
          {nf(explainability.max_pzu_price_eur, 1)})
        </>
      ),
    },
    {
      ok: kpis.confidence >= explainability.confidence_threshold,
      text: (
        <>
          Încredere model <strong>{nf(kpis.confidence * 100, 0)}%</strong> (necesar minim{" "}
          {nf(explainability.confidence_threshold * 100, 0)}%)
        </>
      ),
    },
    {
      ok: explainability.anomalies.length === 0,
      text:
        explainability.anomalies.length === 0 ? (
          <>Senzori în parametri normali</>
        ) : (
          <>
            Valori neobișnuite la senzori: <strong>{explainability.anomalies.join(", ")}</strong>
          </>
        ),
    },
  ];

  return (
    <div className="stack">
      <div className="kpi-grid">
        <MetricCard
          label="Injecție în linie"
          value={nf(kpis.actual_injection_mw)}
          unit="MW"
          tone={overLimit ? "danger" : "neutral"}
          meter={{
            value: kpis.actual_injection_mw,
            max: limit,
            tone: overLimit ? "danger" : kpis.actual_injection_mw > limit * 0.85 ? "warning" : "info",
          }}
          note={
            kpis.is_dispatched
              ? `Redusă de la ${nf(kpis.raw_injection_mw)} MW prin dispecerizare`
              : `Limită linie ${nf(limit, 0)} MW · prognoză ${nf(kpis.predicted_injection_mw)} MW`
          }
        />
        <MetricCard
          label="Surplus peste limită"
          value={`+${nf(kpis.actual_surplus_mw)}`}
          unit="MW"
          tone={surplusTone}
          note={`Radiație ${nf(kpis.solar_irradiance_wm2, 0)} W/m² · ${nf(kpis.ambient_temp_c, 1)} °C`}
        />
        <MetricCard
          label="Preț piață (PZU)"
          value={nf(kpis.day_ahead_price_eur)}
          unit="€/MWh"
          note={`Arbitraj posibil sub ${nf(explainability.max_pzu_price_eur, 0)} €/MWh`}
        />
        <MetricCard
          label="Venit cumulat PPC"
          value={nf(kpis.cumulative_profit_eur)}
          unit="€"
          tone={kpis.cumulative_profit_eur > 0 ? "success" : "neutral"}
          note={`${nf(kpis.cumulative_mwh)} MWh vânduți · ora curentă +${nf(kpis.revenue_eur)} €`}
        />
      </div>

      <Card
        title="Injecție în rețea"
        subtitle="Ultimele 24 de ore măsurate și prognoza modelului pentru următoarele 2 ore"
      >
        {hovered && (
          <div className="chart-readout">
            <span>
              Ora <strong>{hovered.short_time}</strong>
            </span>
            <span>
              Măsurat:{" "}
              <strong>
                {hovered.real_injection_mw !== null ? `${nf(hovered.real_injection_mw)} MW` : "—"}
              </strong>
            </span>
            <span>
              Prognoză: <strong>{nf(hovered.predicted_injection_mw)} MW</strong>
            </span>
            {hovered.delivered_mw > 0.01 && (
              <span>
                Livrat către data centre: <strong>{nf(hovered.delivered_mw)} MW</strong>
              </span>
            )}
          </div>
        )}

        <svg
          className="chart-svg"
          viewBox={`0 0 ${W} ${H}`}
          onMouseLeave={() => setHoverIdx(null)}
          role="img"
          aria-label="Grafic injecție în rețea"
        >
          {ticks.map((t) => (
            <g key={t}>
              <line x1={PAD.l} x2={W - PAD.r} y1={yOf(t)} y2={yOf(t)} stroke="#222c3a" strokeWidth={1} />
              <text x={PAD.l - 8} y={yOf(t) + 4} textAnchor="end" fill="#70809a" fontSize="11">
                {t}
              </text>
            </g>
          ))}

          {/* Limita liniei */}
          <line
            x1={PAD.l}
            x2={W - PAD.r}
            y1={limitY}
            y2={limitY}
            stroke="#ea5a5f"
            strokeWidth={1.4}
            strokeDasharray="6 4"
          />
          <text x={PAD.l - 8} y={limitY + 4} textAnchor="end" fill="#ea5a5f" fontSize="11" fontWeight="600">
            {nf(limit, 0)}
          </text>
          <text x={W - PAD.r} y={limitY - 6} textAnchor="end" fill="#ea5a5f" fontSize="11">
            Limită linie
          </text>

          {/* Zona de depășire */}
          {chartPoints.map((pt, i) => {
            if (i === 0) return null;
            const prev = chartPoints[i - 1];
            const v1 = pt.after_dispatch_mw ?? pt.real_injection_mw ?? pt.predicted_injection_mw;
            const v0 = prev.after_dispatch_mw ?? prev.real_injection_mw ?? prev.predicted_injection_mw;
            if (v1 <= limit && v0 <= limit) return null;
            const saved = pt.delivered_mw > 0.01;
            return (
              <polygon
                key={`ovl-${i}`}
                points={`${xOf(i - 1)},${limitY} ${xOf(i - 1)},${yOf(Math.max(limit, v0))} ${xOf(i)},${yOf(Math.max(limit, v1))} ${xOf(i)},${limitY}`}
                fill={saved ? "rgba(47,178,125,0.2)" : "rgba(234,90,95,0.22)"}
              />
            );
          })}

          {/* Marcaj „acum” */}
          {nowIdx >= 0 && (
            <g>
              <line
                x1={xOf(nowIdx)}
                x2={xOf(nowIdx)}
                y1={PAD.t}
                y2={H - PAD.b}
                stroke="#70809a"
                strokeDasharray="3 3"
              />
              <text x={xOf(nowIdx)} y={PAD.t - 3} textAnchor="middle" fill="#a3afc0" fontSize="11" fontWeight="600">
                Acum
              </text>
            </g>
          )}

          <path d={forecastPath} fill="none" stroke="#4c8dff" strokeWidth={2} strokeDasharray="5 4" />
          <path d={measuredPath} fill="none" stroke="#e8ecf2" strokeWidth={2.2} />

          {chartPoints.map((pt, i) => {
            const cx = xOf(i);
            const cy = yOf(pt.real_injection_mw ?? pt.predicted_injection_mw);
            return (
              <g key={pt.v_idx}>
                {pt.after_dispatch_mw !== null && (
                  <>
                    <line x1={cx} x2={cx} y1={cy} y2={yOf(pt.after_dispatch_mw)} stroke="#2fb27d" strokeWidth={2} />
                    <circle cx={cx} cy={yOf(pt.after_dispatch_mw)} r={4} fill="#2fb27d" />
                  </>
                )}
                <circle
                  cx={cx}
                  cy={cy}
                  r={pt.is_current ? 4.5 : 2.5}
                  fill={pt.is_future ? "#4c8dff" : "#e8ecf2"}
                />
                {i % 4 === 0 && (
                  <text x={cx} y={H - 8} textAnchor="middle" fill="#70809a" fontSize="11">
                    {pt.short_time}
                  </text>
                )}
                <rect
                  x={cx - plotW / n / 2}
                  y={PAD.t}
                  width={plotW / n}
                  height={plotH}
                  fill="transparent"
                  onMouseEnter={() => setHoverIdx(i)}
                />
              </g>
            );
          })}
        </svg>

        <div className="legend">
          <span className="legend-item" style={{ color: "#e8ecf2" }}>
            <span className="swatch" /> Injecție măsurată
          </span>
          <span className="legend-item" style={{ color: "#4c8dff" }}>
            <span className="swatch dashed" /> Prognoză model
          </span>
          <span className="legend-item" style={{ color: "#ea5a5f" }}>
            <span className="swatch dashed" /> Limită linie
          </span>
          <span className="legend-item" style={{ color: "#2fb27d" }}>
            <span className="swatch box" /> Injecție după dispecerizare
          </span>
        </div>
      </Card>

      <div className="grid-2">
        <Card title="Recomandarea sistemului" subtitle={`Situația la ora ${timeline.current_hour_str}`}>
          <p className="reasoning">{explainability.copilot_reasoning}</p>
          <ul className="checks">
            {checks.map((c, idx) => (
              <li key={idx} className={`check ${c.ok ? "ok" : "fail"}`}>
                <span className="check-mark">{c.ok ? "✓" : "✕"}</span>
                <span>{c.text}</span>
              </li>
            ))}
          </ul>
          <p className="hint">O ofertă se generează doar când toate condițiile sunt îndeplinite.</p>
        </Card>

        <Card
          title="De ce prezice modelul acest surplus"
          subtitle="Contribuția fiecărui factor la prognoză (TreeSHAP)"
        >
          <div className="shap">
            {shap.map(([feature, value]) => {
              const pct = Math.min(50, (Math.abs(value) / maxShap) * 50);
              const positive = value >= 0;
              return (
                <div key={feature}>
                  <div className="shap-head">
                    <span className="shap-name">{FEATURE_LABELS[feature] ?? feature}</span>
                    <span className="shap-val">
                      {positive ? "+" : "−"}
                      {nf(Math.abs(value))} MW
                    </span>
                  </div>
                  <div className="shap-track">
                    <span className="shap-mid" />
                    <span
                      className={`shap-bar ${positive ? "shap-pos" : "shap-neg"}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
          <p className="hint">Albastru: crește surplusul prognozat · Gri: îl scade.</p>
        </Card>
      </div>
    </div>
  );
};
