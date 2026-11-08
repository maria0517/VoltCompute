import React, { useState } from "react";
import {
  ChartPoint,
  DashboardKPIs,
  ExplainabilityInfo,
  TimelineInfo,
} from "../types/telemetry";

interface PpcDispatchPanelProps {
  timeline: TimelineInfo;
  kpis: DashboardKPIs;
  explainability: ExplainabilityInfo;
  chartPoints: ChartPoint[];
  yAxisMax: number;
}

const FEATURE_LABELS: Record<string, string> = {
  solar_irradiance_wm2: "Iradianță Solară (W/m²)",
  clear_sky_proxy_mw: "Ancoră Fizică Panouri (MW)",
  ambient_temp_celsius: "Temperatură Ambiantă (°C)",
  day_ahead_price_eur: "Preț PZU (EUR/MWh)",
  hour_sin: "Ciclu Diurn (Sin Oră)",
  hour_cos: "Ciclu Diurn (Cos Oră)",
  thermal_derating_factor: "Derating Termic Panouri",
  net_load_factor_mw: "Sarcină Locală Rețea (MW)",
};

export const PpcDispatchPanel: React.FC<PpcDispatchPanelProps> = ({
  timeline,
  kpis,
  explainability,
  chartPoints,
  yAxisMax,
}) => {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const width = 920;
  const height = 340;
  const padLeft = 48;
  const padRight = 24;
  const padTop = 24;
  const padBottom = 38;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;

  const maxVal = Math.max(16, yAxisMax);
  const nPts = Math.max(1, chartPoints.length - 1);

  const xOf = (i: number) => padLeft + (i / nPts) * plotW;
  const yOf = (mw: number) => padTop + plotH - (Math.min(maxVal, Math.max(0, mw)) / maxVal) * plotH;

  const limitY = yOf(kpis.grid_limit_mw);
  const currentIdx = chartPoints.findIndex((p) => p.is_current);

  // Polylines
  const predPath = chartPoints
    .map((p, i) => `${i === 0 ? "M" : "L"}${xOf(i).toFixed(1)},${yOf(p.predicted_injection_mw).toFixed(1)}`)
    .join(" ");

  const realPoints = chartPoints
    .map((p, i) => (p.real_injection_mw !== null ? { i, val: p.real_injection_mw } : null))
    .filter((x): x is { i: number; val: number } => x !== null);

  const realPath = realPoints
    .map((pt, idx) => `${idx === 0 ? "M" : "L"}${xOf(pt.i).toFixed(1)},${yOf(pt.val).toFixed(1)}`)
    .join(" ");

  const shapEntries = Object.entries(explainability.top_shap_features).slice(0, 5);
  const maxShap = Math.max(0.1, ...shapEntries.map(([, v]) => Math.abs(v)));

  const hoveredPoint = hoverIdx !== null ? chartPoints[hoverIdx] : chartPoints[currentIdx] ?? chartPoints[0];

  return (
    <div>
      <div className="vc-pane-head vc-pane-ppc">
        INTERFAȚĂ PPC // DISPECERAT PARC SOLAR &amp; CONTROL CONGESTIE
      </div>
      <div className="vc-pane-sub">
        Monitorizare injecție în linia de {kpis.grid_limit_mw.toFixed(0)} MW, predicție XGBoost (T+2h) și
        venit din valorificarea surplusului
      </div>

      {/* 4 KPI Cards */}
      <div className="vc-kpi-grid">
        <div className="vc-metric-card">
          <div className="vc-metric-label">Curent în Linie ({timeline.current_hour_str})</div>
          <div className="vc-metric-value">{kpis.actual_injection_mw.toFixed(2)} MW</div>
          <div
            className="vc-metric-delta"
            style={{
              color: kpis.is_dispatched
                ? "#00FF9D"
                : kpis.actual_injection_mw > kpis.grid_limit_mw
                ? "#FF7B7B"
                : "#8E95A5",
            }}
          >
            {kpis.is_dispatched
              ? `↓ Redus de la ${kpis.raw_injection_mw.toFixed(2)} MW`
              : `Predicție AI: ${kpis.predicted_injection_mw.toFixed(2)} MW`}
          </div>
        </div>

        <div className="vc-metric-card">
          <div className="vc-metric-label">Surplus Peste {kpis.grid_limit_mw.toFixed(0)} MW</div>
          <div className="vc-metric-value">+{kpis.actual_surplus_mw.toFixed(2)} MW</div>
          <div
            className="vc-metric-delta"
            style={{ color: kpis.actual_surplus_mw > 0.1 ? "#F59E0B" : "#00FF9D" }}
          >
            Confidență AI: {(kpis.confidence * 100).toFixed(0)}% · PZU {kpis.day_ahead_price_eur.toFixed(1)} €
          </div>
        </div>

        <div className="vc-metric-card">
          <div className="vc-metric-label">Profit Cumulat PPC</div>
          <div className="vc-metric-value" style={{ color: "#00FF9D" }}>
            +{kpis.cumulative_profit_eur.toLocaleString("ro-RO", { minimumFractionDigits: 2 })} €
          </div>
          <div className="vc-metric-delta" style={{ color: "#38BDF8" }}>
            Ora curentă: +{kpis.revenue_eur.toFixed(2)} € ({kpis.delivered_mw.toFixed(2)} MWh)
          </div>
        </div>

        <div className="vc-metric-card">
          <div className="vc-metric-label">Echivalent AI Compute Verde</div>
          <div className="vc-metric-value" style={{ color: "#38BDF8" }}>
            {kpis.ai_tokens_millions.toFixed(1)}M Tokens
          </div>
          <div className="vc-metric-delta" style={{ color: "#00FF9D" }}>
            🌱 {kpis.co2_saved_tons.toFixed(2)}t CO₂ evitate · {kpis.solar_irradiance_wm2.toFixed(0)} W/m²
          </div>
        </div>
      </div>

      {/* SCADA 24h + 2h Chart */}
      <div className="vc-card">
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 8,
            fontFamily: "var(--font-mono)",
            fontSize: "0.78rem",
          }}
        >
          <span style={{ color: "#FFFFFF", fontWeight: 700 }}>
            CURBA DE SARCINĂ SCADA (ULTIMELE 24H + PREDICȚIE AI T+2H)
          </span>
          {hoveredPoint && (
            <span style={{ color: "#38BDF8" }}>
              [{hoveredPoint.timestamp}] Real:{" "}
              {hoveredPoint.real_injection_mw !== null
                ? `${hoveredPoint.real_injection_mw.toFixed(2)} MW`
                : "Viitor"}{" "}
              · AI: {hoveredPoint.predicted_injection_mw.toFixed(2)} MW
              {hoveredPoint.delivered_mw > 0.01 ? ` · Livrat DC: -${hoveredPoint.delivered_mw.toFixed(2)} MW` : ""}
            </span>
          )}
        </div>

        <svg
          viewBox={`0 0 ${width} ${height}`}
          style={{ width: "100%", height: "auto", display: "block", overflow: "visible" }}
          onMouseLeave={() => setHoverIdx(null)}
        >
          {/* Horizontal grid lines */}
          {[0, 4, 8, 10, 12, 16].map((val) => {
            const y = yOf(val);
            const isLimit = val === 10;
            return (
              <g key={val}>
                <line
                  x1={padLeft}
                  y1={y}
                  x2={width - padRight}
                  y2={y}
                  stroke={isLimit ? "#FF4B4B" : "rgba(255,255,255,0.07)"}
                  strokeWidth={isLimit ? 1.8 : 1}
                  strokeDasharray={isLimit ? "6 4" : undefined}
                />
                <text
                  x={padLeft - 8}
                  y={y + 4}
                  textAnchor="end"
                  fill={isLimit ? "#FF7B7B" : "#8E95A5"}
                  fontSize="10"
                  fontFamily="JetBrains Mono, monospace"
                >
                  {val} MW
                </text>
              </g>
            );
          })}

          {/* Highlight area above 10 MW */}
          {chartPoints.map((pt, i) => {
            if (i === 0) return null;
            const prev = chartPoints[i - 1];
            const v1 = pt.after_dispatch_mw ?? pt.real_injection_mw ?? pt.predicted_injection_mw;
            const v0 = prev.after_dispatch_mw ?? prev.real_injection_mw ?? prev.predicted_injection_mw;
            if (v1 <= kpis.grid_limit_mw && v0 <= kpis.grid_limit_mw) return null;
            const x0 = xOf(i - 1);
            const x1 = xOf(i);
            const y0 = yOf(Math.max(kpis.grid_limit_mw, v0));
            const y1 = yOf(Math.max(kpis.grid_limit_mw, v1));
            const isSaved = pt.delivered_mw > 0.01;
            return (
              <polygon
                key={`fill-${i}`}
                points={`${x0},${limitY} ${x0},${y0} ${x1},${y1} ${x1},${limitY}`}
                fill={isSaved ? "rgba(0, 255, 157, 0.22)" : "rgba(255, 75, 75, 0.25)"}
              />
            );
          })}

          {/* Vertical marker NOW (T) */}
          {currentIdx >= 0 && (
            <g>
              <line
                x1={xOf(currentIdx)}
                y1={padTop}
                x2={xOf(currentIdx)}
                y2={height - padBottom}
                stroke="#00FF9D"
                strokeWidth={1.5}
                strokeDasharray="3 3"
              />
              <text
                x={xOf(currentIdx)}
                y={padTop - 7}
                textAnchor="middle"
                fill="#00FF9D"
                fontSize="10"
                fontWeight="700"
                fontFamily="JetBrains Mono, monospace"
              >
                ACUM ({timeline.current_hour_str})
              </text>
            </g>
          )}

          {/* AI Predicted Injection Line */}
          <path
            d={predPath}
            fill="none"
            stroke="#38BDF8"
            strokeWidth={2.2}
            strokeDasharray="5 3"
          />

          {/* Real Injection Line */}
          <path d={realPath} fill="none" stroke="#FFFFFF" strokeWidth={2.4} />

          {/* Points & Post-Dispatch Green Markers */}
          {chartPoints.map((pt, i) => {
            const cx = xOf(i);
            const cy = yOf(pt.real_injection_mw ?? pt.predicted_injection_mw);
            return (
              <g key={pt.v_idx}>
                {pt.after_dispatch_mw !== null && (
                  <>
                    <line
                      x1={cx}
                      y1={cy}
                      x2={cx}
                      y2={yOf(pt.after_dispatch_mw)}
                      stroke="#00FF9D"
                      strokeWidth={2.2}
                    />
                    <circle
                      cx={cx}
                      cy={yOf(pt.after_dispatch_mw)}
                      r={5}
                      fill="#00FF9D"
                      stroke="#0c0d12"
                      strokeWidth={1.5}
                    />
                  </>
                )}
                <circle
                  cx={cx}
                  cy={cy}
                  r={pt.is_current ? 5.5 : pt.is_future ? 4 : 3}
                  fill={pt.is_future ? "#38BDF8" : pt.is_current ? "#00FF9D" : "#FFFFFF"}
                />
                {i % 4 === 0 && (
                  <text
                    x={cx}
                    y={height - 12}
                    textAnchor="middle"
                    fill="#8E95A5"
                    fontSize="10"
                    fontFamily="JetBrains Mono, monospace"
                  >
                    {pt.short_time}
                  </text>
                )}
                <rect
                  x={cx - plotW / nPts / 2}
                  y={padTop}
                  width={plotW / nPts}
                  height={plotH}
                  fill="transparent"
                  style={{ cursor: "crosshair" }}
                  onMouseEnter={() => setHoverIdx(i)}
                />
              </g>
            );
          })}
        </svg>

        <div
          style={{
            display: "flex",
            gap: 18,
            marginTop: 8,
            fontFamily: "var(--font-mono)",
            fontSize: "0.73rem",
            color: "#8E95A5",
            flexWrap: "wrap",
          }}
        >
          <span>⚪ Linia Albă: Injecție Reală Măsurată (MW)</span>
          <span style={{ color: "#38BDF8" }}>🔵 Linia Punctată: Predicție AI XGBoost (T+2h)</span>
          <span style={{ color: "#FF7B7B" }}>🔴 Plafon Linie PPC ({kpis.grid_limit_mw.toFixed(0)} MW)</span>
          <span style={{ color: "#00FF9D" }}>🟢 Puncte Verzi: Curent Redus după Alocarea către Data Centre</span>
        </div>
      </div>

      {/* AI Copilot Reasoning + TreeSHAP Explainability */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 12,
        }}
      >
        <div className="vc-card" style={{ marginBottom: 0 }}>
          <div className="vc-modal-head">🤖 VOLTCOMPUTE COPILOT // JUSTIFICARE DECIZIE</div>
          <div className="vc-modal-body" style={{ fontSize: "0.83rem" }}>
            {explainability.copilot_reasoning}
          </div>
          <div
            style={{
              marginTop: 10,
              paddingTop: 8,
              borderTop: "1px solid rgba(255,255,255,0.07)",
              fontFamily: "var(--font-mono)",
              fontSize: "0.73rem",
              color: "#8E95A5",
            }}
          >
            Reguli active: Surplus &gt; {explainability.min_surplus_threshold_mw} MW · PZU &lt;{" "}
            {explainability.max_pzu_price_eur} €/MWh · Confidență ≥{" "}
            {(explainability.confidence_threshold * 100).toFixed(0)}%
          </div>
        </div>

        <div className="vc-card" style={{ marginBottom: 0 }}>
          <div className="vc-modal-head">📊 EXPLICABILITATE MODEL // TREESHAP EXACT (MW)</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {shapEntries.map(([feat, val]) => {
              const pct = Math.min(100, (Math.abs(val) / maxShap) * 100);
              const isPos = val >= 0;
              return (
                <div key={feat} style={{ fontFamily: "var(--font-mono)", fontSize: "0.73rem" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
                    <span style={{ color: "#D1D5DB" }}>{FEATURE_LABELS[feat] ?? feat}</span>
                    <span style={{ color: isPos ? "#00FF9D" : "#38BDF8", fontWeight: 700 }}>
                      {isPos ? `+${val.toFixed(2)}` : val.toFixed(2)} MW
                    </span>
                  </div>
                  <div className="vc-bar-bg" style={{ margin: 0, height: 5 }}>
                    <div
                      style={{
                        width: `${pct}%`,
                        height: "100%",
                        background: isPos ? "#00FF9D" : "#38BDF8",
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
