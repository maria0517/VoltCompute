import React from "react";
import { CheckCircle2, XCircle, RotateCcw } from "lucide-react";
import { DataCenterCard, NodeStatus, OfferState } from "../types/telemetry";

interface DataCenterPanelProps {
  datacenters: DataCenterCard[];
  offerState: OfferState;
  currentHourStr: string;
  busy: boolean;
  onAccept: () => void;
  onRefuse: () => void;
  onReset: () => void;
}

const STATUS_BADGE: Record<NodeStatus, { label: string; bg: string; color: string }> = {
  ACCEPTED: {
    label: "COMPUTING · ALOCAT",
    bg: "rgba(0, 255, 157, 0.16)",
    color: "#00FF9D",
  },
  OFFER_PENDING: {
    label: "OFERTĂ ACTIVĂ",
    bg: "rgba(245, 158, 11, 0.18)",
    color: "#F59E0B",
  },
  REJECTED_THERMAL_SAFETY: {
    label: "BLOCAT HVAC > 26°C",
    bg: "rgba(255, 75, 75, 0.18)",
    color: "#FF7B7B",
  },
  REFUSED: {
    label: "OFERTĂ REFUZATĂ",
    bg: "rgba(255, 75, 75, 0.14)",
    color: "#FF7B7B",
  },
  UNCERTAIN: {
    label: "ANOMALIE SENZOR >3σ",
    bg: "rgba(245, 158, 11, 0.16)",
    color: "#F59E0B",
  },
  ESCALATE_TO_HUMAN: {
    label: "ESCALADAT DISPECER",
    bg: "rgba(245, 158, 11, 0.16)",
    color: "#F59E0B",
  },
  IN_QUEUE: {
    label: "ÎN COADĂ (ELIGIBIL)",
    bg: "rgba(56, 189, 248, 0.14)",
    color: "#38BDF8",
  },
  STANDBY: {
    label: "STANDBY",
    bg: "rgba(255, 255, 255, 0.08)",
    color: "#8E95A5",
  },
};

export const DataCenterPanel: React.FC<DataCenterPanelProps> = ({
  datacenters,
  offerState,
  currentHourStr,
  busy,
  onAccept,
  onRefuse,
  onReset,
}) => {
  const { offer, offers_exhausted, offer_steps, fixed_price_eur, reference_tariff_eur } = offerState;
  const totalContracted = datacenters.reduce((acc, d) => acc + d.allocated_mwh, 0);

  return (
    <div>
      <div className="vc-pane-head vc-pane-dc">
        INTERFAȚĂ DATA CENTRE // POOL ROUTING &amp; TELEMETRIE HVAC
      </div>
      <div className="vc-pane-sub">
        Clasament determinist Match Score (0.4·VolumFit + 0.4·RatăAcceptare − 0.2·PenalizareTermică)
      </div>

      {/* Active Offer Modal */}
      {offer ? (
        <div
          className="vc-modal"
          style={{ borderColor: "rgba(56, 189, 248, 0.55)" }}
        >
          <div className="vc-modal-head">
            ⚡ OFERTĂ CURENTĂ ({offerState.mode}) · TARIF FIX GARANTAT {fixed_price_eur.toFixed(0)} €/MWh
          </div>
          <div className="vc-modal-title">
            Către: <span style={{ color: offer.node_color }}>{offer.node_name}</span> (Match Score:{" "}
            {(offer.match_score * 100).toFixed(0)}%)
          </div>
          <div className="vc-modal-body">
            Volum propus: <strong style={{ color: "#00FF9D" }}>{offer.total_mwh.toFixed(2)} MWh</strong>{" "}
            · Economie cumpărător:{" "}
            <strong style={{ color: "#38BDF8" }}>
              {offer.buyer_savings_eur.toLocaleString("ro-RO", { minimumFractionDigits: 2 })} €
            </strong>{" "}
            (față de {reference_tariff_eur.toFixed(0)} €/MWh retail).
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 10,
              marginTop: 12,
            }}
          >
            <button
              type="button"
              className="vc-btn vc-btn-primary"
              disabled={busy}
              onClick={onAccept}
            >
              <CheckCircle2 size={16} />
              Aprobă Tranzacția
            </button>

            <button
              type="button"
              className="vc-btn vc-btn-danger"
              disabled={busy}
              onClick={onRefuse}
            >
              <XCircle size={16} />
              Refuză — Următorul DC
            </button>
          </div>

          {totalContracted > 0.01 && (
            <button
              type="button"
              className="vc-btn"
              style={{ width: "100%", marginTop: 8 }}
              disabled={busy}
              onClick={onReset}
            >
              <RotateCcw size={14} />
              Resetează ofertele din fereastră
            </button>
          )}
        </div>
      ) : offer_steps.length > 0 && offers_exhausted ? (
        <div
          className="vc-modal"
          style={{ borderColor: "rgba(255, 75, 75, 0.55)" }}
        >
          <div className="vc-modal-head" style={{ color: "#FF7B7B" }}>
            🚫 ALL_DCS_EXHAUSTED // POOL EPUIZAT
          </div>
          <div className="vc-modal-title">Niciun Data Center Eligibil Rămas</div>
          <div className="vc-modal-body">
            Toate nodurile eligibile au refuzat oferta la {fixed_price_eur.toFixed(0)} €/MWh (sau sunt
            blocate termic HVAC &gt; 26°C). Se solicită intervenția dispeceratului PPC.
          </div>
          <button
            type="button"
            className="vc-btn"
            style={{ width: "100%", marginTop: 10 }}
            disabled={busy}
            onClick={onReset}
          >
            <RotateCcw size={15} />
            Resetează ofertele și reia de la primul clasat
          </button>
        </div>
      ) : totalContracted > 0.01 ? (
        <div
          className="vc-modal"
          style={{ borderColor: "rgba(0, 255, 157, 0.55)" }}
        >
          <div className="vc-modal-head" style={{ color: "#00FF9D" }}>
            ✅ SURPLUS ALOCAT INTEGRAL
          </div>
          <div className="vc-modal-title">
            {totalContracted.toFixed(2)} MWh Contractați la {fixed_price_eur.toFixed(0)} €/MWh
          </div>
          <div className="vc-modal-body">
            Venit PPC:{" "}
            <strong style={{ color: "#00FF9D" }}>
              {(totalContracted * fixed_price_eur).toFixed(2)} €
            </strong>{" "}
            · Economie Data Centre:{" "}
            <strong style={{ color: "#38BDF8" }}>
              {(totalContracted * (reference_tariff_eur - fixed_price_eur)).toFixed(2)} €
            </strong>
            .
          </div>
          <button
            type="button"
            className="vc-btn"
            style={{ width: "100%", marginTop: 10 }}
            disabled={busy}
            onClick={onReset}
          >
            <RotateCcw size={15} />
            Resetează ofertele
          </button>
        </div>
      ) : (
        <div className="vc-modal">
          <div className="vc-modal-head">⚡ FĂRĂ OFERTĂ LA ORA {currentHourStr}</div>
          <div className="vc-modal-body">
            Nu sunt îndeplinite condițiile simultane de arbitraj (surplus &gt; 5 MW și PZU &lt; 25
            €/MWh) nici la {currentHourStr}, nici în următoarele 2 ore. Clusterele rulează în regim
            standard.
          </div>
        </div>
      )}

      {/* 3 Data Center Cards ordered by Match Score */}
      {datacenters.map((dc) => {
        const badge = STATUS_BADGE[dc.status] ?? STATUS_BADGE.STANDBY;
        const tempPct = Math.min(100, Math.max(0, (dc.temp_c / 30.0) * 100));
        const isHot = dc.temp_c > dc.thermal_limit_c;
        return (
          <div
            key={dc.id}
            className="vc-dc-card"
            style={{
              borderColor:
                dc.status === "ACCEPTED"
                  ? "rgba(0, 255, 157, 0.5)"
                  : dc.status === "OFFER_PENDING"
                  ? "rgba(245, 158, 11, 0.55)"
                  : undefined,
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: 6,
              }}
            >
              <div>
                <span style={{ color: dc.color, fontWeight: 700, fontSize: "0.88rem" }}>
                  #{dc.rank} {dc.name}
                </span>
                <div style={{ fontSize: "0.7rem", color: "#8E95A5" }}>
                  {dc.city} · {dc.workload}
                </div>
              </div>
              <span
                className="vc-badge"
                style={{ background: badge.bg, color: badge.color }}
              >
                {badge.label}
              </span>
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(3, 1fr)",
                gap: 8,
                fontSize: "0.74rem",
                marginTop: 8,
              }}
            >
              <div>
                <div style={{ color: "#8E95A5" }}>Match Score</div>
                <div style={{ color: "#FFFFFF", fontWeight: 700 }}>
                  {(dc.match_score * 100).toFixed(0)}%
                </div>
              </div>
              <div>
                <div style={{ color: "#8E95A5" }}>Cap. Liberă</div>
                <div style={{ color: "#38BDF8", fontWeight: 700 }}>
                  {dc.free_capacity_mw.toFixed(2)} MW
                </div>
              </div>
              <div>
                <div style={{ color: "#8E95A5" }}>Rată Acceptare</div>
                <div style={{ color: "#00FF9D", fontWeight: 700 }}>
                  {(dc.acceptance_rate * 100).toFixed(0)}% ({dc.offers_accepted}/{dc.offers_sent})
                </div>
              </div>
            </div>

            {/* Thermal Bar */}
            <div style={{ marginTop: 8, fontSize: "0.72rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "#8E95A5" }}>Temperatură Servere (HVAC)</span>
                <span style={{ color: isHot ? "#FF7B7B" : "#D1D5DB", fontWeight: 700 }}>
                  {dc.temp_c.toFixed(1)}°C / plafon {dc.thermal_limit_c.toFixed(1)}°C
                </span>
              </div>
              <div className="vc-bar-bg">
                <div
                  style={{
                    width: `${tempPct}%`,
                    height: "100%",
                    background: isHot
                      ? "#FF4B4B"
                      : dc.temp_c > dc.thermal_limit_c - 1.2
                      ? "#F59E0B"
                      : "#00FF9D",
                  }}
                />
              </div>
            </div>

            {dc.allocated_mwh > 0.01 && (
              <div
                style={{
                  marginTop: 4,
                  fontSize: "0.73rem",
                  color: "#00FF9D",
                  fontWeight: 600,
                }}
              >
                ⚡ Energie Verde Alocată: +{dc.allocated_mwh.toFixed(2)} MWh [COMPUTING ACTIVE]
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};
