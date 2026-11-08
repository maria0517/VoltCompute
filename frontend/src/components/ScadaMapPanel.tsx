import React, { useState } from "react";
import { nf } from "../lib/format";
import { DashboardKPIs, DataCenterCard, MapTopology, TimelineInfo } from "../types/telemetry";
import { STATUS_META } from "./DataCenterPanel";
import { Card } from "./ui/Card";
import { MetricCard } from "./ui/MetricCard";
import { StatusBadge } from "./ui/StatusBadge";

interface ScadaMapPanelProps {
  timeline: TimelineInfo;
  kpis: DashboardKPIs;
  datacenters: DataCenterCard[];
  topology: MapTopology;
}

const ROMANIA_BORDER: [number, number][] = [
  [47.95, 22.88], [48.22, 26.65], [47.25, 27.62], [46.15, 28.12], [45.45, 28.22],
  [45.2, 29.68], [43.78, 28.58], [43.68, 25.35], [43.82, 22.92], [44.68, 22.15],
  [45.75, 20.28], [46.62, 21.18], [47.95, 22.88],
];

const W = 960;
const H = 500;
const LON = { min: 20.0, max: 29.9 };
const LAT = { min: 43.4, max: 48.4 };

const project = (lat: number, lon: number): [number, number] => [
  40 + ((lon - LON.min) / (LON.max - LON.min)) * (W - 80),
  H - 34 - ((lat - LAT.min) / (LAT.max - LAT.min)) * (H - 68),
];

const toPath = (coords: [number, number][]) =>
  coords
    .map(([lat, lon], i) => {
      const [x, y] = project(lat, lon);
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

export const ScadaMapPanel: React.FC<ScadaMapPanelProps> = ({ kpis, datacenters, topology }) => {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected = datacenters.find((d) => d.id === selectedId) ?? datacenters[0];
  const [sx, sy] = project(topology.solar_park.lat, topology.solar_park.lon);
  const overLimit = kpis.actual_injection_mw > kpis.grid_limit_mw;

  return (
    <div className="stack">
      <div className="kpi-grid">
        <MetricCard
          label="Producție parc solar"
          value={nf(kpis.raw_injection_mw)}
          unit="MW"
          note={`Muntenia Sud · ${nf(kpis.solar_irradiance_wm2, 0)} W/m²`}
        />
        <MetricCard
          label="Curent în linia de 400 kV"
          value={nf(kpis.actual_injection_mw)}
          unit="MW"
          tone={overLimit ? "danger" : "neutral"}
          meter={{ value: kpis.actual_injection_mw, max: kpis.grid_limit_mw, tone: overLimit ? "danger" : "info" }}
          note={`Limită linie ${nf(kpis.grid_limit_mw, 0)} MW`}
        />
        <MetricCard
          label="Energie rutată către data centre"
          value={nf(kpis.delivered_mw)}
          unit="MW"
          tone={kpis.delivered_mw > 0.01 ? "success" : "neutral"}
          note="Tarif fix 30 €/MWh"
        />
        <MetricCard
          label="CO₂ evitat (cumulat)"
          value={nf(kpis.co2_saved_tons)}
          unit="tone"
          note={`≈ ${nf(kpis.ai_tokens_millions, 1)} mil. tokeni AI alimentați cu energie verde`}
        />
      </div>

      <Card
        title="Harta rețelei de transport"
        subtitle="Magistrala de 400 kV și rutele către data centre. Click pe un nod pentru detalii."
      >
        <div className="map-wrap">
          <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", display: "block" }}>
            <path d={toPath(ROMANIA_BORDER)} fill="#121923" stroke="#2a3649" strokeWidth={1.2} />
            <path
              d={toPath(topology.backbone_400kv)}
              fill="none"
              stroke="#34425a"
              strokeWidth={1.5}
              strokeDasharray="4 5"
            />

            {datacenters.map((dc) => {
              const d = toPath(dc.waypoints);
              const active = dc.status === "ACCEPTED" || dc.allocated_mwh > 0.01;
              const pending = dc.status === "OFFER_PENDING";
              const blocked = dc.status === "REJECTED_THERMAL_SAFETY" || dc.status === "REFUSED";
              const color = active ? "#2fb27d" : pending ? "#e3a634" : blocked ? "#6e3236" : "#34425a";
              return (
                <g key={`route-${dc.id}`}>
                  <path d={d} fill="none" stroke={color} strokeWidth={active ? 3 : 1.8} />
                  {(active || pending) && (
                    <path
                      d={d}
                      fill="none"
                      stroke={active ? "#6fd4a8" : "#f0c566"}
                      strokeWidth={2}
                      className="flow-line"
                    />
                  )}
                </g>
              );
            })}

            <g transform={`translate(${sx}, ${sy})`}>
              <circle r={11} fill="#151b24" stroke="#e3a634" strokeWidth={2.5} />
              <circle r={4} fill="#e3a634" />
              <text y={28} textAnchor="middle" fill="#e8ecf2" fontSize="12" fontWeight="600">
                {topology.solar_park.short}
              </text>
              <text y={43} textAnchor="middle" fill="#a3afc0" fontSize="11">
                {nf(kpis.raw_injection_mw)} MW produși
              </text>
            </g>

            {datacenters.map((dc) => {
              const [x, y] = project(dc.lat, dc.lon);
              const active = dc.status === "ACCEPTED" || dc.allocated_mwh > 0.01;
              const pending = dc.status === "OFFER_PENDING";
              const blocked = dc.status === "REJECTED_THERMAL_SAFETY";
              const color = active ? "#2fb27d" : pending ? "#e3a634" : blocked ? "#ea5a5f" : "#70809a";
              const isSel = dc.id === selected?.id;
              return (
                <g
                  key={dc.id}
                  transform={`translate(${x}, ${y})`}
                  style={{ cursor: "pointer" }}
                  onClick={() => setSelectedId(dc.id)}
                >
                  {isSel && <circle r={16} fill="none" stroke="#4c8dff" strokeWidth={1.5} />}
                  <circle r={9} fill="#151b24" stroke={color} strokeWidth={2.5} />
                  <circle r={3.5} fill={color} />
                  <text y={-17} textAnchor="middle" fill="#e8ecf2" fontSize="12" fontWeight="600">
                    {dc.short}
                  </text>
                  <text y={26} textAnchor="middle" fill={color} fontSize="11">
                    {active
                      ? `+${nf(dc.allocated_mwh)} MWh alocați`
                      : pending
                      ? "Ofertă în așteptare"
                      : `${nf(dc.temp_c, 1)} °C · ${nf(dc.free_capacity_mw, 1)} MW liberi`}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        <div className="legend">
          <span className="legend-item" style={{ color: "#2fb27d" }}>
            <span className="swatch" /> Energie rutată
          </span>
          <span className="legend-item" style={{ color: "#e3a634" }}>
            <span className="swatch" /> Ofertă în așteptare
          </span>
          <span className="legend-item" style={{ color: "#6e3236" }}>
            <span className="swatch" /> Refuzat / blocat termic
          </span>
          <span className="legend-item" style={{ color: "#34425a" }}>
            <span className="swatch dashed" /> Magistrala 400 kV
          </span>
        </div>

        {selected && (
          <div className="map-detail">
            <span>
              <strong>{selected.name}</strong> · {selected.city}
            </span>
            <span>
              Sarcină: <strong>{selected.workload}</strong>
            </span>
            <span>
              Temperatură: <strong>{nf(selected.temp_c, 1)} °C</strong> (max.{" "}
              {nf(selected.thermal_limit_c, 1)} °C)
            </span>
            <span>
              Capacitate liberă: <strong>{nf(selected.free_capacity_mw)} MW</strong>
            </span>
            <StatusBadge
              tone={(STATUS_META[selected.status] ?? STATUS_META.STANDBY).tone}
              label={(STATUS_META[selected.status] ?? STATUS_META.STANDBY).label}
            />
          </div>
        )}
      </Card>
    </div>
  );
};
