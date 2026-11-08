import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  acceptOffer,
  fetchDashboardState,
  refuseOffer,
  resetOffers,
} from "./api/client";
import { DataCenterPanel } from "./components/DataCenterPanel";
import { HeaderBar } from "./components/HeaderBar";
import { PpcDispatchPanel } from "./components/PpcDispatchPanel";
import { ScadaMapPanel } from "./components/ScadaMapPanel";
import { DashboardStateResponse } from "./types/telemetry";

export const App: React.FC = () => {
  const [state, setState] = useState<DashboardStateResponse | null>(null);
  const [activeView, setActiveView] = useState<"console" | "map">("console");
  const [autoplay, setAutoplay] = useState<boolean>(false);
  const [busy, setBusy] = useState<boolean>(false);
  const [toast, setToast] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const lastAutopauseOfferRef = useRef<string | null>(null);

  const showToast = useCallback((msg: string) => {
    setToast(msg);
    window.setTimeout(() => {
      setToast((prev) => (prev === msg ? null : prev));
    }, 3600);
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
              `⏸ Live Feed oprit: oferta către ${data.offer_state.offer.node_short} așteaptă Aprobă / Refuză.`
            );
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Eroare conexiune API");
        setAutoplay(false);
      }
    },
    [showToast]
  );

  useEffect(() => {
    void loadStep();
  }, [loadStep]);

  // Autoplay loop (1s interval, continuous monthly ring buffer)
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
      const prevCycle = state.timeline.ring_cycle - 1;
      void loadStep(9999, prevCycle);
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

  const handleAccept = async () => {
    if (!state || !state.offer_state.offer) return;
    setBusy(true);
    try {
      const updated = await acceptOffer(
        state.offer_state.offer.node_id,
        state.offer_state.offer.per_step,
        state.timeline.current_step,
        state.timeline.ring_cycle
      );
      setState(updated);
      if (updated.toast) showToast(`✅ ${updated.toast}`);
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Eroare la aprobare");
    } finally {
      setBusy(false);
    }
  };

  const handleRefuse = async () => {
    if (!state || !state.offer_state.offer) return;
    setBusy(true);
    try {
      const updated = await refuseOffer(
        state.offer_state.offer.node_id,
        state.offer_state.offer.steps,
        state.timeline.current_step,
        state.timeline.ring_cycle
      );
      setState(updated);
      if (updated.toast) showToast(`⛔ ${updated.toast}`);
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Eroare la refuz");
    } finally {
      setBusy(false);
    }
  };

  const handleReset = async () => {
    if (!state) return;
    const stepsToReset =
      state.offer_state.offer_steps.length > 0
        ? state.offer_state.offer_steps
        : state.offer_state.table_steps;
    setBusy(true);
    try {
      const updated = await resetOffers(
        stepsToReset,
        state.timeline.current_step,
        state.timeline.ring_cycle
      );
      setState(updated);
      if (updated.toast) showToast(`↩️ ${updated.toast}`);
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Eroare la resetare");
    } finally {
      setBusy(false);
    }
  };

  if (error && !state) {
    return (
      <div className="vc-container">
        <div className="vc-alert vc-alert-red">
          ⚠️ Eroare conectare la backend-ul FastAPI: {error}. Asigură-te că serverul rulează pe{" "}
          <code>uvicorn server:app --port 8000</code>.
        </div>
      </div>
    );
  }

  if (!state) {
    return (
      <div className="vc-container">
        <div className="vc-sub" style={{ paddingTop: 24 }}>
          ⚡ Se încarcă telemetria SCADA și modelul XGBoost...
        </div>
      </div>
    );
  }

  const alertClass =
    state.banner.level === "red"
      ? "vc-alert vc-alert-red"
      : state.banner.level === "amber"
      ? "vc-alert vc-alert-amber"
      : "vc-alert vc-alert-green";

  return (
    <div className="vc-container">
      <HeaderBar
        timeline={state.timeline}
        liveMae={state.kpis.live_window_mae}
        activeView={activeView}
        autoplay={autoplay}
        onSelectView={setActiveView}
        onStepBack={handleStepBack}
        onNextHour={handleNextHour}
        onTogglePlay={() => setAutoplay((prev) => !prev)}
        onSliderChange={handleSliderChange}
      />

      <div className={alertClass}>{state.banner.text}</div>

      <div className="vc-main-grid">
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

      {toast && <div className="vc-toast">{toast}</div>}
    </div>
  );
};
