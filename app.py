"""VOLTCOMPUTE // PPC GRID DISPATCH — SCADA Live Terminal & Intraday AI Agent Console."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agent import VoltComputeAgent
from schemas import AgentDecision, DispatchAction
from simulator import VoltComputeSimulator, load_scenario_status
from train_pipeline import load_hourly_history

st.set_page_config(
    page_title="VOLTCOMPUTE // PPC GRID DISPATCH",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

FIXED_PRICE_EUR = 30.0
LOOKAHEAD_HOURS = 4
HISTORY_HOURS = 24
REFERENCE_TARIFF_EUR = VoltComputeSimulator.STANDARD_RETAIL_PRICE_EUR
AUTOPLAY_SECONDS = 1.0

NODE_META: Dict[str, Dict[str, Any]] = {
    "DC_AI_BUCURESTI": {
        "name": "RoEduNet - UPB AI Cluster (București)",
        "short": "RoEduNet UPB",
        "daemon": "ROEDUNET UPB SLURM DAEMON",
        "queue": 45,
        "color": "#00FF9D",
    },
    "DC_CLOUD_CLUJ": {
        "name": "UTCN High-Performance Node (Cluj)",
        "short": "UTCN HPC Cluj",
        "daemon": "UTCN CLUJ KUBERNETES DAEMON",
        "queue": 28,
        "color": "#A78BFA",
    },
    "DC_EDGE_TIMISOARA": {
        "name": "UVT HPC Center (Timișoara)",
        "short": "UVT HPC Timișoara",
        "daemon": "UVT TIMIȘOARA SLURM DAEMON",
        "queue": 20,
        "color": "#38BDF8",
    },
}

SCENARIOS: Dict[str, str] = {
    "CRITICAL_SOLAR_SUNDAY": "☀️ Amiază Însorită (Vârf Solar)",
    "DATACENTER_THERMAL_RISK": "🔥 Avarie Chiller (Risc Termic)",
    "UNCERTAIN_WEATHER_SHOCK": "⛈️ Front Instabil (Confidență < 75%)",
    "REAL_CSV": "📊 Prosumatori CSV (Scară Nativă)",
}

# =====================================================================
# STYLE (Exact SCADA Terminal Aesthetic from main/AI-urea)
# =====================================================================
st.markdown(
    """
<style>
#MainMenu, footer, header {visibility: hidden;}
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background-color: #0c0d12 !important;
    color: #E5E7EB;
}
.block-container {padding-top: 1.1rem; padding-bottom: 1.5rem; max-width: 100%;}
[data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace !important;
}
[data-testid="stMetric"] {
    background: #12141b;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] {color: #8E95A5;}
.vc-title {
    font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace;
    font-size: 1.25rem; font-weight: 700; letter-spacing: 0.03em; color: #FFFFFF;
}
.vc-live {
    display: inline-block; width: 9px; height: 9px; border-radius: 50%;
    background: #00FF9D; margin-right: 8px; animation: vcpulse 1.8s infinite;
}
@keyframes vcpulse {
    0% {box-shadow: 0 0 0 0 rgba(0,255,157,0.7);}
    70% {box-shadow: 0 0 0 8px rgba(0,255,157,0);}
    100% {box-shadow: 0 0 0 0 rgba(0,255,157,0);}
}
.vc-sub {font-family: ui-monospace, Menlo, monospace; font-size: 0.78rem; color: #8E95A5;}
.vc-badge-pill {
    display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 0.70rem;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    background: rgba(56,189,248,0.10); border: 1px solid rgba(56,189,248,0.30);
    color: #38BDF8; margin-left: 6px;
}
.vc-alert {
    border-radius: 10px; padding: 10px 16px; margin: 8px 0 12px 0;
    font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace; font-size: 0.85rem;
}
.vc-alert-red {background: rgba(255,75,75,0.12); border: 1px solid rgba(255,75,75,0.5); color: #FF7B7B;}
.vc-alert-amber {background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.5); color: #FBBF24;}
.vc-alert-green {background: rgba(0,255,157,0.07); border: 1px solid rgba(0,255,157,0.3); color: #00FF9D;}
.vc-modal {
    background: #14161e; border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px; padding: 16px; margin-bottom: 10px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.55);
}
.vc-modal-head {
    font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace;
    font-size: 0.76rem; color: #38BDF8; letter-spacing: 0.06em; margin-bottom: 6px;
}
.vc-modal-title {font-size: 1.0rem; font-weight: 700; color: #FFFFFF; margin-bottom: 6px;}
.vc-modal-body {font-size: 0.85rem; color: #D1D5DB; line-height: 1.45;}
.vc-mono {font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace;}
.vc-ticker {
    display: flex; justify-content: space-between; align-items: baseline;
    font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace; margin: 2px 0 4px 0;
}
.vc-ticker-price {font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em;}
.vc-ticker-meta {font-size: 0.78rem; color: #8E95A5;}
.vc-badge-live {
    display: inline-block; padding: 2px 9px; border-radius: 6px; font-size: 0.72rem;
    font-weight: 700; letter-spacing: 0.08em; margin-right: 10px;
    background: rgba(255,75,75,0.16); color: #FF4B4B; border: 1px solid rgba(255,75,75,0.5);
    animation: vcpulse 1.4s infinite;
}
.vc-badge-paused {
    display: inline-block; padding: 2px 9px; border-radius: 6px; font-size: 0.72rem;
    font-weight: 700; letter-spacing: 0.08em; margin-right: 10px;
    background: rgba(142,149,165,0.14); color: #8E95A5; border: 1px solid rgba(142,149,165,0.35);
}
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# DATA, MODEL & SESSION STATE
# =====================================================================
@st.cache_data(show_spinner=False)
def read_history() -> pd.DataFrame:
    return load_hourly_history()


def demo_status(step: int) -> Dict[str, Any]:
    """Explicit fail-closed synthetic fallback when CSV files are missing."""
    ts = pd.Timestamp("2026-05-01") + pd.Timedelta(hours=step)
    grid = dict(
        timestamp=ts.isoformat(),
        hour=ts.hour,
        solar_irradiance_wm2=700.0,
        ambient_temp_celsius=22.0,
        day_ahead_price_eur=0.0,
        grid_capacity_limit_mw=0.35,
        prosumer_battery_soc_pct=85.0,
        weather_stability_index=0.58,
    )
    centers = [
        dict(
            dc_id="DC_AI_BUCURESTI",
            capacitate_libera_mw=6.0,
            temp_servere_c=21.0,
            limita_termica_c=26.0,
            pret_contract_eur_mwh=110.0,
        )
    ]
    return dict(
        grid=grid,
        base_grid=deepcopy(grid),
        data_centers=centers,
        confidence_override=None,
        data_source="DEMO SYNTHETIC — CSV indisponibil",
    )


for name, default in (
    ("agent", None),
    ("transactions", {}),
    ("decision_index", {}),
    ("operator_log", []),
    ("scenario", "CRITICAL_SOLAR_SUNDAY"),
    ("current_step", 35),
    ("autoplay", False),
    ("pending_advance", False),
):
    if name not in st.session_state:
        st.session_state[name] = default

if st.session_state.agent is None:
    st.session_state.agent = VoltComputeAgent()
agent: VoltComputeAgent = st.session_state.agent

MIN_STEP = 24
MAX_STEP = 719

if st.session_state.pending_advance:
    st.session_state.pending_advance = False
    st.session_state.current_step = int(min(MAX_STEP, int(st.session_state.current_step) + 1))

try:
    history = read_history()
    real_data = True
    feed_source = "data/1_macro + 2_micro (100 prosumatori) + 3_data_centers.csv"
except (OSError, ValueError, KeyError):
    real_data = False
    feed_source = "synthetic fallback feed"
    times = pd.date_range("2026-05-01", periods=720, freq="h")
    history = pd.DataFrame(
        {
            "timestamp": times,
            "hour": times.hour,
            "radiatie_solara_wm2": np.maximum(0.0, 850.0 * np.sin((times.hour - 6) * np.pi / 13.0)),
            "temp_c": 22.0 + 6.0 * np.sin((times.hour - 5) * np.pi / 15.0),
            "pret_pzu_eur_mwh": np.full(720, 45.0),
            "Surplus_National_MW": np.maximum(0.0, 0.4 * np.sin((times.hour - 6) * np.pi / 12.0)),
            "avg_battery_soc_pct": np.full(720, 80.0),
            "total_prosumer_load_mw": np.full(720, 0.12),
        }
    )

model_source = "XGBoost t+2h (model.joblib + TreeSHAP)" if agent.model_artifact is not None else "Fallback fizic"


# =====================================================================
# HEADER + SCENARIO BAR + TIMELINE CONTROLS
# =====================================================================
st.markdown(
    f"""
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-wrap:wrap;">
  <div class="vc-title">
    <span class="vc-live"></span>VOLTCOMPUTE // PPC GRID DISPATCH
    <span class="vc-badge-pill">INTRADAY t+2h</span>
    <span class="vc-badge-pill">EU AI ACT · HITL</span>
    <span class="vc-badge-pill">PII &amp; PROMPT GUARD</span>
  </div>
  <div class="vc-sub">FEED: {feed_source} · MODEL: {model_source} · 720h</div>
</div>
""",
    unsafe_allow_html=True,
)


def select_scenario(name: str) -> None:
    st.session_state.autoplay = False
    st.session_state.scenario = name
    st.session_state.current_step = 35


sc_cols = st.columns(4, gap="small")
for col, (sc_name, sc_label) in zip(sc_cols, SCENARIOS.items()):
    is_active = st.session_state.scenario == sc_name
    col.button(
        f"● {sc_label}" if is_active else sc_label,
        key=f"scenario_{sc_name}",
        on_click=select_scenario,
        args=(sc_name,),
        type="primary" if is_active else "secondary",
        width="stretch",
    )


def _step_back() -> None:
    st.session_state.autoplay = False
    st.session_state.current_step = max(MIN_STEP, int(st.session_state.current_step) - 1)


def _step_forward() -> None:
    st.session_state.autoplay = False
    st.session_state.current_step = min(MAX_STEP, int(st.session_state.current_step) + 1)


def _toggle_play() -> None:
    st.session_state.autoplay = not st.session_state.autoplay


bar_back, bar_fwd, bar_play, bar_slider = st.columns([1, 1, 1.7, 5], gap="small")
bar_back.button("◀ Pas Înapoi", key="nav_step_back", on_click=_step_back, width="stretch")
bar_fwd.button("▶ Pas Înainte", key="nav_step_fwd", on_click=_step_forward, width="stretch")
bar_play.button(
    "⏹ Stop Live Feed" if st.session_state.autoplay else "▶▶ Play Live Feed (Auto-step 1s)",
    key="nav_toggle_play",
    on_click=_toggle_play,
    type="primary",
    width="stretch",
)
with bar_slider:
    st.select_slider(
        "Timeline",
        options=list(range(MIN_STEP, MAX_STEP + 1)),
        key="current_step",
        format_func=lambda i: pd.Timestamp(history.iloc[i]["timestamp"]).strftime("%Y-%m-%d %H:%M"),
        label_visibility="collapsed",
    )


# =====================================================================
# TRANSACTIONAL STATE AT STEP T
# =====================================================================
step = int(st.session_state.current_step)
scenario = str(st.session_state.scenario)

if real_data:
    status = load_scenario_status(scenario, step)
else:
    status = demo_status(step)

grid = status["base_grid"]
centers = status["data_centers"]
tx_key = (scenario, step, status["data_source"])

if tx_key not in st.session_state.transactions:
    decision = agent.run_cycle(
        status["grid"],
        data_centers=centers,
        confidence_override=status["confidence_override"],
    )
    st.session_state.transactions[tx_key] = dict(
        decision=decision,
        forecast=decision.model_copy(deep=True),
        injection=decision.demo_injected_surplus_mw,
        result=None,
        grid=deepcopy(status["grid"]),
        centers=deepcopy(centers),
    )
    st.session_state.decision_index[decision.decision_id] = tx_key

tx = st.session_state.transactions[tx_key]
decision: AgentDecision = tx["decision"]
forecast: AgentDecision = tx["forecast"]

executed_reduction = round(decision.executed_absorbed_mw + decision.executed_curtailment_mw, 3)
remaining_excess = (
    float(tx["result"]["remaining_grid_excess_mw"])
    if tx["result"] is not None
    else float(decision.grid_excess_mw)
)
saved_curtailment = min(
    max(0.0, decision.grid_excess_mw - decision.executed_curtailment_mw),
    decision.executed_absorbed_mw,
)


def transact(
    key: Tuple[str, int, str],
    decision_id: str,
    center_id: str,
    action: str,
    fallback_revision: int | None = None,
) -> None:
    """Executes an operator HITL action (Approve / Dutch Auction / Reject / Approve Fallback)."""
    record = st.session_state.transactions[key]
    if st.session_state.decision_index.get(decision_id) != key:
        return
    try:
        result = st.session_state.agent.act(
            record["decision"],
            dispatch_action=DispatchAction(
                decision_id=decision_id,
                center_id=center_id,
                action=action,  # type: ignore[arg-type]
                fallback_revision=fallback_revision,
            ),
        )
    except ValueError:
        st.session_state.action_error = (
            "Acțiune blocată de controalele de siguranță (verificați oferta activă și temperatura serverelor)."
        )
        return
    record["decision"] = AgentDecision.model_validate(result["decision"])
    record["result"] = result
    st.session_state.operator_log.append(
        dict(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            scenario=key[0],
            step=key[1],
            decision_id=decision_id,
            center_id=center_id,
            action=action,
            effective_mw=result["effective_mw"],
            executed_mw=result["executed_mw"],
            fallback_revision=fallback_revision,
            effective_absorbed_mw=result["effective_absorbed_mw"],
            effective_curtailment_mw=result["effective_curtailment_mw"],
            executed_absorbed_mw=result["executed_absorbed_mw"],
            remaining_grid_excess_mw=result["remaining_grid_excess_mw"],
            execution_state=result["execution_state"],
            webhook_sent=result["webhook_sent"],
        )
    )


# =====================================================================
# SERIES HELPERS FOR SCADA LIVE CHART
# =====================================================================
is_park_scale = scenario != "REAL_CSV"
visual_limit_mw = 10.0 if is_park_scale else float(grid.get("grid_capacity_limit_mw", 0.35))


def step_net_injection_mw(idx: int) -> float:
    """Computes net grid injection at step `idx` on the active scenario scale."""
    row = history.iloc[idx]
    if not is_park_scale:
        return round(float(row["Surplus_National_MW"]), 3)
    irr = float(row["radiatie_solara_wm2"])
    temp = float(row["temp_c"])
    h = int(row["hour"])
    derating = 1.0 - max(0.0, (temp - 25.0) * 0.004)
    gross = (irr / 1000.0) * 17.8 * derating
    local_load = 1.4 + 0.5 * float(np.sin(np.pi * h / 12.0))
    base_net = max(0.0, gross - local_load)
    # În fereastra solară 09..16 aliniem vârful cu surplusul scenariului (+8 MW + export CSV)
    if 9 <= h <= 16 and irr > 200:
        return round(max(base_net, 10.0 + float(row["Surplus_National_MW"]) + 8.0 * (irr / 850.0)), 3)
    return round(base_net, 3)


# =====================================================================
# ALERT BANNER
# =====================================================================
now_dt = pd.Timestamp(grid["timestamp"]).to_pydatetime()
target_dt = now_dt + timedelta(hours=2)
active_pzu_eur = float(tx["grid"].get("demo_market_price_eur_mwh") or grid["day_ahead_price_eur"])

if decision.status == "ESCALATE_TO_HUMAN":
    alert_html = (
        f'<div class="vc-alert vc-alert-amber">⚠️ STATUS: UNCERTAIN — Preluat de Operator Uman · '
        f'Confidența modelului ({decision.confidence_score:.0%}) este sub pragul de siguranță (75%) pe fond de '
        f'instabilitate atmosferică. Dispecerizarea automată este blocată (0.00 MW).</div>'
    )
elif decision.status == "DATA_ERROR":
    alert_html = (
        '<div class="vc-alert vc-alert-red">🛑 DATA_ERROR: Telemetrie coruptă sau model indisponibil — '
        'alocare automată blocată în mod fail-closed (0.00 MW).</div>'
    )
elif executed_reduction > 0 and remaining_excess <= 1e-6:
    alert_html = (
        f'<div class="vc-alert vc-alert-green">✅ REȚEA SALVATĂ ÎN SIMULARE: {decision.executed_absorbed_mw:.2f} MW '
        f'absorbiți (Data Centers + EV/BESS) pentru fereastra t+2h ({target_dt.strftime("%H:%M")}) — '
        f'excesul peste limita liniei a fost eliminat complet!</div>'
    )
elif decision.surplus_predicted_mw > 0.05:
    alert_html = (
        f'<div class="vc-alert vc-alert-red">⚠️ ALERTĂ CURTAILMENT (t+2h → {target_dt.strftime("%H:%M")}): '
        f'Surplus prognozat de {decision.surplus_predicted_mw:.2f} MW (Baterii prosumatori la '
        f'{float(grid["prosumer_battery_soc_pct"]):.1f}% SoC · PZU {active_pzu_eur:.1f} €/MWh)</div>'
    )
else:
    alert_html = (
        f'<div class="vc-alert vc-alert-green">● GRID NOMINAL: Nicio depășire critică a limitei liniei '
        f'prognozată în următoarele {LOOKAHEAD_HOURS} ore.</div>'
    )

col_main, col_modal = st.columns([0.72, 0.28], gap="large")

# =====================================================================
# MAIN COLUMN (SCADA CHART + ARBITRAGE TABLE + XAI/FALLBACK)
# =====================================================================
with col_main:
    st.markdown(alert_html, unsafe_allow_html=True)

    accepted_offers = [o for o in decision.dc_offers if o.status == "ACCEPTED"]
    planned_revenue_eur = sum(o.offered_mw * o.offered_price_eur for o in decision.dc_offers if o.offered_mw > 0)
    executed_revenue_eur = sum(o.offered_mw * o.offered_price_eur for o in accepted_offers)
    displayed_revenue = executed_revenue_eur if executed_revenue_eur > 0 else planned_revenue_eur
    revenue_label = "executat" if executed_revenue_eur > 0 else "planificat Flash Deal"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Iradiere & Vreme (t)",
        f"{float(grid['solar_irradiance_wm2']):.0f} W/m²",
        f"{now_dt.strftime('%H:%M')} · {float(grid['ambient_temp_celsius']):.1f}°C · SoC {float(grid['prosumer_battery_soc_pct']):.0f}%",
    )
    m2.metric(
        "Capacitate Rețea",
        f"{visual_limit_mw:.2f} MW" if not is_park_scale else f"{visual_limit_mw:.0f} MW",
        f"exces rămas {remaining_excess:.2f} MW",
        delta_color="inverse",
    )
    m3.metric(
        "Surplus Alocat (t+2h)",
        f"{decision.allocated_mw:.2f} MW",
        f"executat {decision.executed_absorbed_mw:.2f} MW · conf {decision.confidence_score:.0%}",
    )
    m4.metric(
        f"Venit PPC ({revenue_label})",
        f"{displayed_revenue:,.2f} €",
        f"PZU {active_pzu_eur:.1f} €/MWh",
        delta_color="off",
    )

    # ---------------- Live ticker chart ----------------
    start_i = max(MIN_STEP, step - HISTORY_HOURS)
    end_i = min(MAX_STEP, step + LOOKAHEAD_HOURS)
    window = list(range(start_i, end_i + 1))
    xs = [pd.Timestamp(history.iloc[i]["timestamp"]).to_pydatetime() for i in window]
    limit_line = [visual_limit_mw for _ in window]

    inj_hist = [step_net_injection_mw(i) if i <= step else None for i in window]
    inj_fcst = [step_net_injection_mw(i) if i >= step else None for i in window]

    # Curba de surplus prezis de model pe fereastra t..t+4
    fc_map: Dict[int, float] = {0: float(decision.surplus_predicted_mw)}
    for pt in forecast.forecast_window:
        fc_map[pt.prediction_horizon_hours] = round(
            float(pt.predicted_export_mw) + (float(tx["injection"]) if is_park_scale else 0.0), 3
        )
    sur_pred = [fc_map.get(i - step) if i >= step else None for i in window]

    # Curba după Arbitraj (când operatorul aprobă oferte DC sau Fallback)
    inj_after_arb = [
        max(0.0, (step_net_injection_mw(i) - executed_reduction))
        if (i >= step and executed_reduction > 0)
        else None
        for i in window
    ]

    # Bare stivuite pe noduri Data Center
    node_bars: Dict[str, List[float]] = {dc_id: [] for dc_id in NODE_META}
    offer_by_id = {o.dc_id: o for o in decision.dc_offers}
    for i in window:
        for dc_id in NODE_META:
            if i == step and dc_id in offer_by_id:
                node_bars[dc_id].append(round(float(offer_by_id[dc_id].offered_mw), 3))
            elif i > step and dc_id in offer_by_id and decision.allocated_mw > 0:
                share = float(offer_by_id[dc_id].offered_mw) / max(decision.allocated_mw, 1e-6)
                s_h = fc_map.get(i - step, 0.0)
                node_bars[dc_id].append(round(s_h * share, 3))
            else:
                node_bars[dc_id].append(0.0)

    live_value = step_net_injection_mw(step)
    prev_value = step_net_injection_mw(step - 1) if step > MIN_STEP else live_value
    tick_delta = live_value - prev_value
    tick_color = "#00FF9D" if tick_delta >= 0 else "#FF4B4B"
    tick_arrow = "▲" if tick_delta >= 0 else "▼"
    status_badge = (
        '<span class="vc-badge-live">● LIVE</span>'
        if st.session_state.autoplay
        else '<span class="vc-badge-paused">❚❚ PAUSED</span>'
    )
    st.markdown(
        f"""
<div class="vc-ticker">
  <div>{status_badge}<span class="vc-ticker-meta">PPC SOLAR INJECTION · NET MW ({SCENARIOS[scenario]})</span></div>
  <div>
    <span class="vc-ticker-price" style="color:{tick_color};">{live_value:.2f} MW</span>
    <span class="vc-ticker-meta" style="color:{tick_color};margin-left:8px;">{tick_arrow} {abs(tick_delta):.2f} vs T-1h · {now_dt.strftime('%d %b %H:%M')}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    y_axis_max = max(max(step_net_injection_mw(i) for i in window), visual_limit_mw) * 1.18 + 0.25

    fig = go.Figure()
    for dc_id, meta in NODE_META.items():
        fig.add_trace(
            go.Bar(
                x=xs,
                y=node_bars[dc_id],
                name=f"Alocat: {meta['short']}",
                marker_color=meta["color"],
                opacity=0.82,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=inj_hist,
            mode="lines",
            name="Producție solară netă (istoric ≤ T)",
            line=dict(color="#00FF9D", width=2.5, shape="spline", smoothing=0.6),
            fill="tozeroy",
            fillcolor="rgba(0,255,157,0.07)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=inj_fcst,
            mode="lines",
            name="Injecție prognozată fără arbitraj (T..T+4)",
            line=dict(color="#38BDF8", width=2, dash="dash", shape="spline", smoothing=0.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=sur_pred,
            mode="lines+markers",
            name="Surplus prezis de model XGBoost (t+1..t+4)",
            line=dict(color="#F59E0B", width=2, dash="dot"),
            marker=dict(size=6),
        )
    )
    if executed_reduction > 0:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=inj_after_arb,
                mode="lines+markers",
                name="Injecție netă după Arbitraj (Rețea Salvată)",
                line=dict(color="#10B981", width=3),
                marker=dict(size=8, color="#00FF9D", symbol="diamond"),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=limit_line,
            mode="lines",
            name="Limită linie PPC",
            line=dict(color="#FF4B4B", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[now_dt],
            y=[live_value],
            mode="markers",
            name="Tick curent",
            marker=dict(size=13, color=tick_color, line=dict(color="#FFFFFF", width=2)),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_hline(
        y=live_value,
        line_width=1,
        line_dash="dot",
        line_color=tick_color,
        opacity=0.6,
        annotation_text=f" {live_value:.2f} MW ",
        annotation_position="right",
        annotation_font=dict(color="#0c0d12", size=11),
        annotation_bgcolor=tick_color,
    )
    fig.add_vrect(
        x0=now_dt,
        x1=now_dt + timedelta(hours=LOOKAHEAD_HOURS),
        fillcolor="rgba(142,149,165,0.12)",
        line_width=0,
        annotation_text="PROGNOZĂ ML (t+1..t+4h)",
        annotation_position="top left",
        annotation_font=dict(color="#F59E0B", size=11),
    )
    fig.add_shape(
        type="line",
        x0=now_dt,
        x1=now_dt,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="#FF4B4B", width=2, dash="dot"),
    )
    fig.add_annotation(
        x=now_dt,
        y=1,
        yref="paper",
        text=f"T = {now_dt.strftime('%H:%M')}",
        showarrow=False,
        yanchor="bottom",
        font=dict(color="#FF7B7B", size=11),
    )
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#8E95A5", size=11),
        height=420,
        margin=dict(l=10, r=60, t=40, b=10),
        hovermode="x unified",
        uirevision="voltcompute-live",
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            range=[now_dt - timedelta(hours=HISTORY_HOURS), now_dt + timedelta(hours=LOOKAHEAD_HOURS)],
            showspikes=True,
            spikecolor="rgba(255,255,255,0.35)",
            spikethickness=1,
            spikedash="dot",
        ),
        yaxis=dict(
            title="MW",
            side="right",
            range=[0, y_axis_max],
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.1)",
            showspikes=True,
            spikecolor="rgba(255,255,255,0.35)",
            spikethickness=1,
            spikedash="dot",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0, font=dict(size=10)),
    )
    st.plotly_chart(fig, width="stretch", key="scada_live_chart")

    # ---------------- Arbitrage table ----------------
    st.markdown(
        '<div class="vc-sub" style="margin:4px 0;">ARBITRAJ MULTI-NOD · DECIZIE AGENT LA ORA T → LIVRARE T+2H</div>',
        unsafe_allow_html=True,
    )
    table_rows = []
    for offer in decision.dc_offers:
        meta = NODE_META.get(
            offer.dc_id,
            {"name": offer.dc_name, "queue": 20},
        )
        q_total = int(meta["queue"])
        jobs_unlocked = (
            max(1, int(round(q_total * offer.offered_mw / max(offer.available_capacity_mw, 0.1))))
            if offer.offered_mw > 0
            else 0
        )
        table_rows.append(
            {
                "Nod": f"{meta['name']} [{offer.dc_id}]",
                "Temp Răcire (°C)": offer.current_temp_c,
                "Cutoff (°C)": offer.thermal_limit_c - 1.5,
                "Capacitate (MW)": offer.available_capacity_mw,
                "Status Agent": offer.status,
                "Ofertă (MW)": offer.offered_mw,
                "Preț Ofertă (€/MWh)": offer.offered_price_eur if offer.offered_mw > 0 or offer.status == "REJECTED_BY_OPERATOR" else 0.0,
                "Contract (€/MWh)": offer.contract_price_eur,
                "Joburi Deblocate": f"{jobs_unlocked}/{q_total}",
            }
        )
    table_df = pd.DataFrame(table_rows)

    def _status_color(value: str) -> str:
        palette = {
            "OFFER_GENERATED": "color:#00FF9D;font-weight:600",
            "ACCEPTED": "color:#10B981;font-weight:700",
            "REJECTED_THERMAL_SAFETY": "color:#FF4B4B;font-weight:600",
            "REJECTED_BY_OPERATOR": "color:#FF7B7B;font-weight:600",
            "ESCALATE_TO_HUMAN": "color:#F59E0B;font-weight:600",
        }
        return palette.get(value, "color:#8E95A5")

    st.dataframe(
        table_df.style.map(_status_color, subset=["Status Agent"]).format(
            {
                "Temp Răcire (°C)": "{:.1f}",
                "Cutoff (°C)": "{:.1f}",
                "Capacitate (MW)": "{:.2f}",
                "Ofertă (MW)": "{:.2f}",
                "Preț Ofertă (€/MWh)": "{:.1f}",
                "Contract (€/MWh)": "{:.1f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    # ---------------- Expanders: Fallback Cascade & XAI TreeSHAP ----------------
    exp_col1, exp_col2 = st.columns(2, gap="medium")
    with exp_col1:
        fb = decision.fallback
        fallback_pending = (
            max(0.0, fb.ppc_blue_ev_mw - decision.executed_ppc_blue_ev_mw)
            + max(0.0, fb.bess_storage_mw - decision.executed_bess_storage_mw)
            + max(0.0, fb.curtailment_mw - decision.executed_curtailment_mw)
            if fb
            else 0.0
        )
        with st.expander(
            f"🔋 Cascadă Fallback PPC (EV / BESS / Curtailment) · De aprobat: {fallback_pending:.2f} MW",
            expanded=fallback_pending > 0.01 or decision.executed_ppc_blue_ev_mw > 0,
        ):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Treaptă Fallback": label,
                            "Planificat (MW)": getattr(fb, field, 0.0),
                            "Executat (MW)": executed_val,
                            "În Așteptare (MW)": max(0.0, getattr(fb, field, 0.0) - executed_val),
                        }
                        for label, field, executed_val in (
                            ("Treapta 1 · Stații EV PPC Blue (≤3.5 MW)", "ppc_blue_ev_mw", decision.executed_ppc_blue_ev_mw),
                            ("Treapta 2 · Baterii Industriale BESS (≤4.0 MW)", "bess_storage_mw", decision.executed_bess_storage_mw),
                            ("Treapta 3 · Curtailment Asistat SCADA", "curtailment_mw", decision.executed_curtailment_mw),
                        )
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            st.button(
                "✅ Aprobă Fallback EV / BESS / Curtailment (Simulare HITL)",
                key=f"{decision.decision_id}_PPC_FALLBACK_r{decision.fallback_plan_revision}_APPROVED",
                type="primary",
                width="stretch",
                disabled=(
                    fallback_pending <= 1e-9
                    or decision.confidence_score < 0.75
                    or decision.status == "DATA_ERROR"
                ),
                on_click=transact,
                args=(
                    tx_key,
                    decision.decision_id,
                    "PPC_FALLBACK",
                    "APPROVED",
                    decision.fallback_plan_revision,
                ),
            )

    with exp_col2:
        with st.expander("🧠 Explicabilitate XAI (TreeSHAP) & Metrici XGBoost vs. Baseline", expanded=True):
            if forecast.top_shap_features:
                s_names = list(forecast.top_shap_features.keys())
                s_vals = [forecast.top_shap_features[k] for k in s_names]
                s_pcts = [forecast.shap_percentages.get(k, 0.0) for k in s_names]
                shap_fig = go.Figure(
                    go.Bar(
                        y=s_names,
                        x=s_pcts,
                        orientation="h",
                        text=[f"{p:.1f}% ({v:+.4f} MW)" for p, v in zip(s_pcts, s_vals)],
                        textposition="auto",
                        marker_color=["#00FF9D" if v >= 0 else "#38BDF8" for v in s_vals],
                    )
                )
                shap_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="JetBrains Mono, monospace", color="#8E95A5", size=10),
                    height=155,
                    margin=dict(l=5, r=10, t=5, b=5),
                    xaxis=dict(title="Pondere |SHAP| Top 3 (%)", range=[0, 110], gridcolor="rgba(255,255,255,0.05)"),
                )
                st.plotly_chart(shap_fig, width="stretch", key="xai_shap_chart")
            metrics = (agent.model_artifact or {}).get("metrics", {})
            m_rows = [
                {
                    "Model (t+2h)": label,
                    "MAE (MW)": round(metrics[m_key]["mae"], 5),
                    "RMSE (MW)": round(metrics[m_key]["rmse"], 5),
                    "R²": round(metrics[m_key]["r2"], 4),
                }
                for m_key, label in (
                    ("xgboost", "XGBoost Regressor (VoltCompute)"),
                    ("baseline_seasonal", "Baseline Sezonier 24h y(t-22)"),
                    ("baseline_persistence", "Baseline Persistență y(t)"),
                )
                if m_key in metrics
            ]
            if m_rows:
                st.dataframe(pd.DataFrame(m_rows), hide_index=True, width="stretch")


# =====================================================================
# SECONDARY COLUMN: DATACENTER CLIENT VIEW (SLURM / K8S MODALS)
# =====================================================================
with col_modal:
    st.markdown(
        '<div class="vc-sub" style="margin-bottom:8px;">DATACENTER CLIENT VIEW · SLURM / K8S DISPATCH</div>',
        unsafe_allow_html=True,
    )
    if "action_error" in st.session_state:
        st.error(st.session_state.pop("action_error"))

    if tx["result"] is not None:
        res = tx["result"]
        st.markdown(
            f"""
<div class="vc-alert vc-alert-green" style="padding:8px 12px;margin-bottom:10px;">
  ⚡ <b>{res['execution_state']}</b> · Absorbție: <b>{ decision.executed_absorbed_mw:.2f} MW</b> ·
  CO₂ evitat: <b>{(decision.executed_mw + decision.executed_ppc_blue_ev_mw) * 0.28:.2f} t</b>
</div>
""",
            unsafe_allow_html=True,
        )

    for offer in decision.dc_offers:
        meta = NODE_META.get(
            offer.dc_id,
            {
                "name": offer.dc_name,
                "short": offer.dc_id,
                "daemon": f"{offer.dc_id} SLURM DAEMON",
                "queue": 20,
                "color": "#00FF9D",
            },
        )
        q_total = int(meta["queue"])
        jobs_cnt = (
            max(1, int(round(q_total * offer.offered_mw / max(offer.available_capacity_mw, 0.1))))
            if offer.offered_mw > 0
            else 0
        )
        savings_eur = offer.offered_mw * (offer.contract_price_eur - offer.offered_price_eur)

        if offer.status in ("OFFER_GENERATED", "ACCEPTED"):
            badge_txt = "✅ ACCEPTED & RUNNING" if offer.status == "ACCEPTED" else f"🔔 ACTIVE FLASH DEAL · RUNDĂ #{offer.auction_round}"
            st.markdown(
                f"""
<div class="vc-modal" style="border-left: 3px solid {meta['color']};">
  <div class="vc-modal-head">{meta['daemon']} · {badge_txt}</div>
  <div class="vc-modal-title">{meta['name']}</div>
  <div class="vc-modal-body">
    PPC Dispatcher oferă <span class="vc-mono" style="color:#00FF9D;font-weight:700;">{offer.offered_mw:.2f} MW</span>
    pentru fereastra <span class="vc-mono">{target_dt.strftime('%H:%M')}–{(target_dt + timedelta(hours=1)).strftime('%H:%M')}</span> la
    <span class="vc-mono" style="color:#00FF9D;font-weight:700;">{offer.offered_price_eur:.0f} €/MWh</span>
    (discount <span class="vc-mono">{offer.discount_pct:.0f}%</span> vs <span class="vc-mono">{offer.contract_price_eur:.0f} €/MWh</span> contract ·
    răcire <span class="vc-mono">{offer.current_temp_c:.1f}°C</span>).
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
            k1, k2 = st.columns(2)
            k1.metric("Economie estimată (1h)", f"-{savings_eur:,.2f} €")
            k2.metric("Joburi AI/HPC pornite", f"{jobs_cnt}/{q_total}")
        else:
            status_reason = {
                "REJECTED_THERMAL_SAFETY": f"🔥 PROTECȚIE TERMICĂ: Servere la {offer.current_temp_c:.1f}°C (peste pragul de siguranță {offer.thermal_limit_c - 1.5:.1f}°C). Transfer blocat.",
                "ESCALATE_TO_HUMAN": f"⚠️ INCERTITUDINE METEO: Confidență ML ({decision.confidence_score:.0%}) sub 75% — în așteptarea dispecerului uman.",
                "REJECTED_BY_OPERATOR": "❌ OFERTĂ REFUZATĂ: Megawații au fost rerutați automat către următoarele noduri / cascada PPC Blue EV & BESS.",
                "STANDBY": "● STANDBY: Capacitate disponibilă în rezervă pentru rerutare.",
            }.get(offer.status, offer.status)
            st.markdown(
                f"""
<div class="vc-modal" style="opacity:0.85;">
  <div class="vc-modal-head">{meta['daemon']} · {offer.status}</div>
  <div class="vc-modal-title">{meta['name']}</div>
  <div class="vc-modal-body">{status_reason}</div>
</div>
""",
                unsafe_allow_html=True,
            )

        btn_disabled = (
            offer.status != "OFFER_GENERATED"
            or decision.confidence_score < 0.75
            or decision.status == "DATA_ERROR"
        )
        b_acc, b_neg, b_ref = st.columns([1.3, 1.1, 0.9], gap="small")
        b_acc.button(
            "Acceptă & Scalează",
            key=f"{decision.decision_id}_{offer.dc_id}_APPROVED",
            type="primary",
            width="stretch",
            disabled=btn_disabled,
            on_click=transact,
            args=(tx_key, decision.decision_id, offer.dc_id, "APPROVED"),
        )
        b_neg.button(
            "−10 €/MWh",
            key=f"{decision.decision_id}_{offer.dc_id}_DUTCH_AUCTION_STEP",
            type="secondary",
            width="stretch",
            disabled=btn_disabled or offer.offered_price_eur <= 10.0,
            on_click=transact,
            args=(tx_key, decision.decision_id, offer.dc_id, "DUTCH_AUCTION_STEP"),
        )
        b_ref.button(
            "Refuză",
            key=f"{decision.decision_id}_{offer.dc_id}_REJECTED",
            type="secondary",
            width="stretch",
            disabled=btn_disabled,
            on_click=transact,
            args=(tx_key, decision.decision_id, offer.dc_id, "REJECTED"),
        )


# =====================================================================
# AUTO-STEP LIVE FEED
# =====================================================================
if st.session_state.autoplay:
    time.sleep(AUTOPLAY_SECONDS)
    if st.session_state.current_step < MAX_STEP:
        st.session_state.pending_advance = True
    else:
        st.session_state.autoplay = False
    st.rerun()
