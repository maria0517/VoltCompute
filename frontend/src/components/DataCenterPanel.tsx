import React from "react";
import { nf } from "../lib/format";
import { DataCenterCard, NodeStatus, OfferState } from "../types/telemetry";
import { Card } from "./ui/Card";
import { StatusBadge, Tone } from "./ui/StatusBadge";

interface DataCenterPanelProps {
  datacenters: DataCenterCard[];
  offerState: OfferState;
  currentHourStr: string;
  busy: boolean;
  onAccept: () => void;
  onRefuse: () => void;
  onReset: () => void;
}

export const STATUS_META: Record<NodeStatus, { label: string; tone: Tone }> = {
  ACCEPTED: { label: "Alimentat cu energie verde", tone: "success" },
  OFFER_PENDING: { label: "Ofertă în așteptare", tone: "warning" },
  REJECTED_THERMAL_SAFETY: { label: "Blocat · temperatură ridicată", tone: "danger" },
  REFUSED: { label: "A refuzat oferta", tone: "danger" },
  UNCERTAIN: { label: "Date senzori neclare", tone: "warning" },
  ESCALATE_TO_HUMAN: { label: "Necesită decizie operator", tone: "warning" },
  IN_QUEUE: { label: "Eligibil", tone: "info" },
  STANDBY: { label: "În așteptare", tone: "neutral" },
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
    <div className="stack">
      <Card
        title="Ofertă de dispecerizare"
        subtitle={`Preț fix garantat: ${nf(fixed_price_eur, 0)} €/MWh`}
      >
        {offer ? (
          <>
            <div style={{ marginBottom: 10 }}>
              <div style={{ fontWeight: 650, fontSize: 15 }}>{offer.node_name}</div>
              <div className="card-sub">Fereastră de ofertare: {offerState.mode.toLowerCase()}</div>
            </div>
            <div className="rows">
              <div className="row">
                <span className="row-label">Volum propus</span>
                <span className="row-value">{nf(offer.total_mwh)} MWh</span>
              </div>
              <div className="row">
                <span className="row-label">Scor de potrivire</span>
                <span className="row-value">{nf(offer.match_score * 100, 0)}%</span>
              </div>
              <div className="row">
                <span className="row-label">Venit PPC</span>
                <span className="row-value tone-success">+{nf(offer.ppc_revenue_eur)} €</span>
              </div>
              <div className="row">
                <span className="row-label">
                  Economie data center (vs. {nf(reference_tariff_eur, 0)} €/MWh)
                </span>
                <span className="row-value">{nf(offer.buyer_savings_eur)} €</span>
              </div>
            </div>
            <div className="offer-actions">
              <button type="button" className="btn btn-primary btn-block" disabled={busy} onClick={onAccept}>
                Aprobă oferta
              </button>
              <button type="button" className="btn btn-danger btn-block" disabled={busy} onClick={onRefuse}>
                Refuză · trimite la următorul data center
              </button>
              {totalContracted > 0.01 && (
                <button type="button" className="btn btn-block" disabled={busy} onClick={onReset}>
                  Resetează alocările
                </button>
              )}
            </div>
          </>
        ) : offer_steps.length > 0 && offers_exhausted ? (
          <>
            <p className="callout callout-danger">
              Toate data centerele eligibile au refuzat oferta sau sunt blocate termic. Este
              necesară intervenția dispecerului PPC.
            </p>
            <div className="offer-actions">
              <button type="button" className="btn btn-block" disabled={busy} onClick={onReset}>
                Resetează și reia ofertarea
              </button>
            </div>
          </>
        ) : totalContracted > 0.01 ? (
          <>
            <div className="rows">
              <div className="row">
                <span className="row-label">Stare</span>
                <span className="row-value tone-success">Surplus alocat integral</span>
              </div>
              <div className="row">
                <span className="row-label">Volum contractat</span>
                <span className="row-value">{nf(totalContracted)} MWh</span>
              </div>
              <div className="row">
                <span className="row-label">Venit PPC</span>
                <span className="row-value tone-success">
                  +{nf(totalContracted * fixed_price_eur)} €
                </span>
              </div>
            </div>
            <div className="offer-actions">
              <button type="button" className="btn btn-block" disabled={busy} onClick={onReset}>
                Resetează alocările
              </button>
            </div>
          </>
        ) : (
          <p className="callout">
            Nicio ofertă activă la ora {currentHourStr}. O ofertă apare când surplusul depășește
            5 MW, prețul pieței este sub 25 €/MWh și datele senzorilor sunt valide.
          </p>
        )}
      </Card>

      <Card
        title="Centre de date"
        subtitle="Ordonate după scorul de potrivire · limită termică HVAC 26 °C"
      >
        <div className="dc-list">
          {datacenters.map((dc) => {
            const meta = STATUS_META[dc.status] ?? STATUS_META.STANDBY;
            const scale = 32;
            const fill = Math.min(100, Math.max(0, (dc.temp_c / scale) * 100));
            const limitPos = Math.min(100, (dc.thermal_limit_c / scale) * 100);
            const hot = dc.temp_c > dc.thermal_limit_c;
            const warm = !hot && dc.temp_c > dc.thermal_limit_c - 1;
            const cls =
              dc.status === "ACCEPTED"
                ? "is-active"
                : dc.status === "OFFER_PENDING"
                ? "is-pending"
                : dc.status === "REJECTED_THERMAL_SAFETY"
                ? "is-blocked"
                : "";

            return (
              <article key={dc.id} className={`dc-item ${cls}`}>
                <div className="dc-top">
                  <div>
                    <div className="dc-rank">Locul {dc.rank}</div>
                    <div className="dc-name">{dc.name}</div>
                    <div className="dc-city">{dc.city}</div>
                  </div>
                  <StatusBadge tone={meta.tone} label={meta.label} />
                </div>

                <div className="dc-stats">
                  <div>
                    <div className="dc-stat-label">Scor potrivire</div>
                    <div className="dc-stat-value">{nf(dc.match_score * 100, 0)}%</div>
                  </div>
                  <div>
                    <div className="dc-stat-label">Capacitate liberă</div>
                    <div className="dc-stat-value">{nf(dc.free_capacity_mw)} MW</div>
                  </div>
                  <div>
                    <div className="dc-stat-label">Alocat</div>
                    <div className={`dc-stat-value ${dc.allocated_mwh > 0.01 ? "tone-success" : ""}`}>
                      {nf(dc.allocated_mwh)} MWh
                    </div>
                  </div>
                </div>

                <div className="temp">
                  <div className="temp-head">
                    <span>Temperatură servere</span>
                    <span className={hot ? "tone-danger" : ""}>
                      <strong className="num">{nf(dc.temp_c, 1)} °C</strong> / max.{" "}
                      {nf(dc.thermal_limit_c, 1)} °C
                    </span>
                  </div>
                  <div className="temp-track">
                    <div
                      className={`temp-fill ${hot ? "hot" : warm ? "warn" : ""}`}
                      style={{ width: `${fill}%` }}
                    />
                    <span className="temp-limit" style={{ left: `${limitPos}%` }} />
                  </div>
                </div>
              </article>
            );
          })}
        </div>
      </Card>
    </div>
  );
};
