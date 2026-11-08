import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  acceptOffer,
  fetchDashboardState,
  refuseOffer,
  resetOffers,
} from "./api/client";
import { DataCenterPanel } from "./components/DataCenterPanel";
import { HeaderBar, ViewId } from "./components/HeaderBar";
import { PpcDispatchPanel } from "./components/PpcDispatchPanel";
import { ScadaMapPanel } from "./components/ScadaMapPanel";
import { StatusBanner } from "./components/StatusBanner";
import { DashboardStateResponse } from "./types/telemetry";

export const App: React.FC = () => {
  const [state, setState] = useState<DashboardStateResponse | null>(null);
  const [activeView, setActiveView] = useState<ViewId>("console");
  const [autoplay, setAutoplay] = useState<boolean>(false);
  const [busy, setBusy] = useState<boolean>(false);
  const [toast, setToast] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const lastAutopauseOfferRef = useRef<string | null>(null);

  const showToast = useCallback((msg: string) => {
    setToast(msg);
    window.setTimeout(() => setToast((prev) => (prev === msg ? null : prev)), 4000);
  }, []);

  const loadStep = useCallback(
    async (step?: number, cycle: number = 0, fromAutoplay: boolean = false) => {
      try {
        const data = await fetchDashboardState(step, cycle);
        setState(data);
        setError(null);

        if (fromAutoplay && data.offer_state.offer) {
          const offerKey = `${data.offer_state.offer.node_id}@${data.offer_state.offer.steps[0]}`;
          if (offerKey !== lastAutopauseOfferRef.current) {
            lastAutopauseOfferRef.current = offerKey;
            setAutoplay(false);
            showToast(
              `Redarea s-a oprit: ${data.offer_state.offer.node_short} are o ofertă care așteaptă decizia dumneavoastră.`
            );
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Eroare de conexiune la server");
        setAutoplay(false);
      }
    },
    [showToast]
  );

  useEffect(() => {
    void loadStep();
  }, [loadStep]);

  // Redare automată: o oră la fiecare secundă, continuu de la o lună la alta
  useEffect(() => {
    if (!autoplay || !state) return;
    const timer = window.setTimeout(() => {
      const nxt = state.timeline.current_step + 1;
      if (nxt >= state.timeline.hours_in_month) {
        void loadStep(0, state.timeline.ring_cycle + 1, true);
      } else {
        void loadStep(nxt, state.timeline.ring_cycle, true);
      }
    }, 1000);
    return () => window.clearTimeout(timer);
  }, [autoplay, state, loadStep]);

  const handleStepBack = () => {
    if (!state) return;
    setAutoplay(false);
    const prv = state.timeline.current_step - 1;
    if (prv < 0) {
      // serverul limitează pasul la ultima oră a lunii precedente
      void loadStep(9999, state.timeline.ring_cycle - 1);
    } else {
      void loadStep(prv, state.timeline.ring_cycle);
    }
  };

  const handleNextHour = () => {
    if (!state) return;
    setAutoplay(false);
    const nxt = state.timeline.current_step + 1;
    if (nxt >= state.timeline.hours_in_month) {
      void loadStep(0, state.timeline.ring_cycle + 1);
    } else {
      void loadStep(nxt, state.timeline.ring_cycle);
    }
  };

  const handleSliderChange = (newStep: number) => {
    if (!state) return;
    setAutoplay(false);
    void loadStep(newStep, state.timeline.ring_cycle);
  };

  const runAction = async (
    action: () => Promise<DashboardStateResponse>,
    fallbackError: string
  ) => {
    setBusy(true);
    try {
      const updated = await action();
      setState(updated);
      if (updated.toast) showToast(updated.toast);
    } catch (err) {
      showToast(err instanceof Error ? err.message : fallbackError);
    } finally {
      setBusy(false);
    }
  };

  const handleAccept = () => {
    const offer = state?.offer_state.offer;
    if (!state || !offer) return;
    void runAction(
      () => acceptOffer(offer.node_id, offer.per_step, state.timeline.current_step, state.timeline.ring_cycle),
      "Aprobarea ofertei a eșuat"
    );
  };

  const handleRefuse = () => {
    const offer = state?.offer_state.offer;
    if (!state || !offer) return;
    void runAction(
      () => refuseOffer(offer.node_id, offer.steps, state.timeline.current_step, state.timeline.ring_cycle),
      "Refuzarea ofertei a eșuat"
    );
  };

  const handleReset = () => {
    if (!state) return;
    const steps =
      state.offer_state.offer_steps.length > 0
        ? state.offer_state.offer_steps
        : state.offer_state.table_steps;
    void runAction(
      () => resetOffers(steps, state.timeline.current_step, state.timeline.ring_cycle),
      "Resetarea alocărilor a eșuat"
    );
  };

  if (!state) {
    return (
      <div className="app-shell">
        <div className="state-screen">
          {error ? (
            <>
              <strong style={{ color: "var(--danger)" }}>Nu mă pot conecta la server.</strong>
              <p style={{ marginTop: 8 }}>{error}</p>
              <p className="hint">Verificați că backend-ul rulează (uvicorn server:app).</p>
            </>
          ) : (
            "Se încarcă datele de telemetrie și modelul de prognoză…"
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <HeaderBar
        timeline={state.timeline}
        liveMae={state.kpis.live_window_mae}
        activeView={activeView}
        autoplay={autoplay}
        connected={error === null}
        onSelectView={setActiveView}
        onStepBack={handleStepBack}
        onNextHour={handleNextHour}
        onTogglePlay={() => setAutoplay((prev) => !prev)}
        onSliderChange={handleSliderChange}
      />

      <StatusBanner banner={state.banner} />

      <div className="workspace">
        {activeView === "console" ? (
          <PpcDispatchPanel
            timeline={state.timeline}
            kpis={state.kpis}
            explainability={state.explainability}
            chartPoints={state.chart_points}
            yAxisMax={state.y_axis_max}
          />
        ) : (
          <ScadaMapPanel
            timeline={state.timeline}
            kpis={state.kpis}
            datacenters={state.datacenters}
            topology={state.map_topology}
          />
        )}

        <DataCenterPanel
          datacenters={state.datacenters}
          offerState={state.offer_state}
          currentHourStr={state.timeline.current_hour_str}
          busy={busy}
          onAccept={handleAccept}
          onRefuse={handleRefuse}
          onReset={handleReset}
        />
      </div>

      {toast && <div className="toast" role="status">{toast}</div>}
    </div>
  );
};
