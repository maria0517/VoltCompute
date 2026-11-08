import React, { useState } from "react";
import {
  DashboardKPIs,
  DataCenterCard,
  MapTopology,
  TimelineInfo,
} from "../types/telemetry";

interface ScadaMapPanelProps {
  timeline: TimelineInfo;
  kpis: DashboardKPIs;
  datacenters: DataCenterCard[];
  topology: MapTopology;
}

// Conturul geografic simplificat al României (lat, lon) pentru fundalul SCADA
const ROMANIA_BORDER: [number, number][] = [
  [47.95, 22.88],
  [48.22, 26.65],
  [47.25, 27.62],
  [46.15, 28.12],
  [45.45, 28.22],
  [45.2, 29.68],
  [43.78, 28.58],
  [43.68, 25.35],
  [43.82, 22.92],
  [44.68, 22.15],
  [45.75, 20.28],
  [46.62, 21.18],
  [47.95, 22.88],
];

export const ScadaMapPanel: React.FC<ScadaMapPanelProps> = ({
  timeline,
  kpis,
  datacenters,
  topology,
}) => {
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const width = 920;
  const height = 540;

  // Proiecție Mercator locală pe România (lon: 20.0 -> 29.9, lat: 43.4 -> 48.4)
  const lonMin = 20.0;
  const lonMax = 29.9;
  const latMin = 43.4;
  const latMax = 48.4;

  const project = (lat: number, lon: number): [number, number] => {
    const x = 45 + ((lon - lonMin) / (lonMax - lonMin)) * (width - 90);
    const y = height - 40 - ((lat - latMin) / (latMax - latMin)) * (height - 80);
    return [x, y];
  };

  const toSvgPath = (coords: [number, number][]) =>
    coords
      .map(([lat, lon], idx) => {
        const [x, y] = project(lat, lon);
        return `${idx === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");

  const [solarX, solarY] = project(topology.solar_park.lat, topology.solar_park.lon);
  const selectedDc = datacenters.find((d) => d.id === selectedId) ?? datacenters[0];

  return (
    <div>
      <div className="vc-pane-head vc-pane-ppc">
        HARTA NAȚIONALĂ SCADA 400 kV // FLUX ENERGETIC PPC → DATA CENTRES
      </div>
      <div className="vc-pane-sub">
        Topologie magistrală 400 kV Transelectrica și rutare live a surplusului solar către
        clusterele AI/HPC
      </div>

      {/* Top KPI Summary Bar */}
      <div className="vc-kpi-grid">
        <div className="vc-metric-card">
          <div className="vc-metric-label">Producție Parc Solar PPC</div>
          <div className="vc-metric-value">{kpis.raw_injection_mw.toFixed(2)} MW</div>
          <div className="vc-metric-delta" style={{ color: "#8E95A5" }}>
            Muntenia Sud · {kpis.solar_irradiance_wm2.toFixed(0)} W/m²
          </div>
        </div>

        <div className="vc-metric-card">
          <div className="vc-metric-label">Curent în Magistrala PPC</div>
          <div className="vc-metric-value">{kpis.actual_injection_mw.toFixed(2)} MW</div>
          <div
            className="vc-metric-delta"
            style={{
              color:
                kpis.actual_injection_mw > kpis.grid_limit_mw ? "#FF7B7B" : "#00FF9D",
            }}
          >
            Plafon termic linie: {kpis.grid_limit_mw.toFixed(0)} MW
          </div>
        </div>

        <div className="vc-metric-card">
          <div className="vc-metric-label">Flux Rutat spre Data Centre</div>
          <div className="vc-metric-value" style={{ color: "#00FF9D" }}>
            {kpis.delivered_mw.toFixed(2)} MW
          </div>
          <div className="vc-metric-delta" style={{ color: "#38BDF8" }}>
            Tarif fix garantat: 30 €/MWh
          </div>
        </div>

        <div className="vc-metric-card">
          <div className="vc-metric-label">Echivalent AI Compute</div>
          <div className="vc-metric-value" style={{ color: "#38BDF8" }}>
            {kpis.ai_tokens_millions.toFixed(1)}M Tokens
          </div>
          <div className="vc-metric-delta" style={{ color: "#00FF9D" }}>
            🌱 {kpis.co2_saved_tons.toFixed(2)}t CO₂ evitate
          </div>
        </div>
      </div>

      {/* Geospatial SCADA Map Canvas */}
      <div className="vc-card" style={{ padding: 12 }}>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          style={{
            width: "100%",
            height: "auto",
            background: "radial-gradient(circle at 50% 50%, #131824 0%, #0a0c10 100%)",
            borderRadius: 10,
            border: "1px solid rgba(255,255,255,0.08)",
            display: "block",
          }}
        >
          {/* Coordinate Grid */}
          {[44, 45, 46, 47, 48].map((lat) => {
            const [, y] = project(lat, 25);
            return (
              <g key={`lat-${lat}`}>
                <line
                  x1={0}
                  y1={y}
                  x2={width}
                  y2={y}
                  stroke="rgba(56, 189, 248, 0.06)"
                  strokeWidth={1}
                />
                <text
                  x={12}
                  y={y - 4}
                  fill="rgba(142,149,165,0.45)"
                  fontSize="9"
                  fontFamily="JetBrains Mono, monospace"
                >
                  {lat}°N
                </text>
              </g>
            );
          })}
          {[21, 23, 25, 27, 29].map((lon) => {
            const [x] = project(45, lon);
            return (
              <g key={`lon-${lon}`}>
                <line
                  x1={x}
                  y1={0}
                  x2={x}
                  y2={height}
                  stroke="rgba(56, 189, 248, 0.06)"
                  strokeWidth={1}
                />
                <text
                  x={x + 4}
                  y={height - 10}
                  fill="rgba(142,149,165,0.45)"
                  fontSize="9"
                  fontFamily="JetBrains Mono, monospace"
                >
                  {lon}°E
                </text>
              </g>
            );
          })}

          {/* Romania Border Contour */}
          <path
            d={toSvgPath(ROMANIA_BORDER)}
            fill="rgba(30, 41, 59, 0.28)"
            stroke="rgba(148, 163, 184, 0.25)"
            strokeWidth={1.5}
          />

          {/* 400 kV National Backbone Ring */}
          <path
            d={toSvgPath(topology.backbone_400kv)}
            fill="none"
            stroke="rgba(148, 163, 184, 0.28)"
            strokeWidth={2}
            strokeDasharray="4 4"
          />

          {/* Dedicated Power Corridors from PPC Solar Park to each Data Center */}
          {datacenters.map((dc) => {
            const pathD = toSvgPath(dc.waypoints);
            const isAccepted = dc.status === "ACCEPTED" || dc.allocated_mwh > 0.01;
            const isPending = dc.status === "OFFER_PENDING";
            const isRejected =
              dc.status === "REJECTED_THERMAL_SAFETY" || dc.status === "REFUSED";

            const strokeColor = isAccepted
              ? "#00FF9D"
              : isPending
              ? "#F59E0B"
              : isRejected
              ? "rgba(255, 75, 75, 0.45)"
              : "rgba(56, 189, 248, 0.28)";

            return (
              <g key={`corridor-${dc.id}`}>
                {/* Base corridor */}
                <path
                  d={pathD}
                  fill="none"
                  stroke={strokeColor}
                  strokeWidth={isAccepted ? 5 : isPending ? 3.5 : 2}
                  opacity={isAccepted ? 0.3 : 0.7}
                />
                {/* Animated power flow overlay when active or pending */}
                {(isAccepted || isPending) && (
                  <path
                    d={pathD}
                    fill="none"
                    stroke={isAccepted ? "#00FF9D" : "#F59E0B"}
                    strokeWidth={3}
                    className="vc-power-flow-active"
                  />
                )}
              </g>
            );
          })}

          {/* PPC Solar Park Node (Muntenia Sud) */}
          <g transform={`translate(${solarX}, ${solarY})`}>
            <circle r={18} fill="rgba(250, 204, 21, 0.18)" />
            <circle
              r={10}
              fill="#FACC15"
              stroke="#0c0d12"
              strokeWidth={2}
            />
            <text
              x={0}
              y={28}
              textAnchor="middle"
              fill="#FACC15"
              fontSize="11"
              fontWeight="700"
              fontFamily="JetBrains Mono, monospace"
            >
              ☀️ {topology.solar_park.short}
            </text>
            <text
              x={0}
              y={42}
              textAnchor="middle"
              fill="#E5E7EB"
              fontSize="10"
              fontFamily="JetBrains Mono, monospace"
            >
              Injecție: {kpis.raw_injection_mw.toFixed(2)} MW
            </text>
          </g>

          {/* Data Center Nodes */}
          {datacenters.map((dc) => {
            const [x, y] = project(dc.lat, dc.lon);
            const isAccepted = dc.status === "ACCEPTED" || dc.allocated_mwh > 0.01;
            const isPending = dc.status === "OFFER_PENDING";
            const isBlocked = dc.status === "REJECTED_THERMAL_SAFETY";
            const nodeColor = isAccepted
              ? "#00FF9D"
              : isPending
              ? "#F59E0B"
              : isBlocked
              ? "#FF4B4B"
              : dc.color;

            return (
              <g
                key={dc.id}
                transform={`translate(${x}, ${y})`}
                style={{ cursor: "pointer" }}
                onClick={() => setSelectedId(dc.id)}
              >
                <circle
                  r={isAccepted || isPending ? 20 : 14}
                  fill={nodeColor}
                  opacity={0.18}
                />
                <circle
                  r={9}
                  fill={nodeColor}
                  stroke="#0c0d12"
                  strokeWidth={2}
                />
                <text
                  x={0}
                  y={-15}
                  textAnchor="middle"
                  fill="#FFFFFF"
                  fontSize="11"
                  fontWeight="700"
                  fontFamily="JetBrains Mono, monospace"
                >
                  #{dc.rank} {dc.short} ({(dc.match_score * 100).toFixed(0)}%)
                </text>
                <text
                  x={0}
                  y={23}
                  textAnchor="middle"
                  fill={nodeColor}
                  fontSize="10"
                  fontWeight="600"
                  fontFamily="JetBrains Mono, monospace"
                >
                  {isAccepted
                    ? `🟢 +${dc.allocated_mwh.toFixed(2)} MW [COMPUTING]`
                    : isPending
                    ? `🟡 OFERTĂ ACTIVĂ`
                    : `${dc.temp_c.toFixed(1)}°C · Lib: ${dc.free_capacity_mw.toFixed(1)} MW`}
                </text>
              </g>
            );
          })}
        </svg>

        {selectedDc && (
          <div
            style={{
              marginTop: 10,
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              fontFamily: "var(--font-mono)",
              fontSize: "0.76rem",
              color: "#D1D5DB",
              flexWrap: "wrap",
              gap: 8,
            }}
          >
            <span>
              📍 <strong>{selectedDc.name}</strong> ({selectedDc.city}) · Job:{" "}
              <span style={{ color: "#38BDF8" }}>{selectedDc.workload}</span>
            </span>
            <span>
              HVAC: <strong>{selectedDc.temp_c.toFixed(1)}°C</strong> /{" "}
              {selectedDc.thermal_limit_c.toFixed(1)}°C · Capacitate liberă:{" "}
              <strong style={{ color: "#00FF9D" }}>
                {selectedDc.free_capacity_mw.toFixed(2)} MW
              </strong>
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
