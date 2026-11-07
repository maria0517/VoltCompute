import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor

from agent import VoltComputeAgent
from schemas import GridTelemetry
from simulator import VoltComputeSimulator
from train import (
    DATA_PATH,
    ENGINEERED_FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_engineered_features,
    generate_synthetic_dataset,
)

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

DATACENTERS: List[Dict[str, Any]] = [
    {
        "id": "ROEDUNET-UPB-BUC",
        "name": "RoEduNet - UPB AI Cluster (București)",
        "short": "RoEduNet UPB",
        "capacity_mw": 6.0,
        "cooling_c": 20.5,
        "queue": 45,
        "color": "#00FF9D",
    },
    {
        "id": "UVT-HPC-TIM",
        "name": "UVT HPC Center (Timișoara)",
        "short": "UVT HPC",
        "capacity_mw": 3.5,
        "cooling_c": 22.0,
        "queue": 20,
        "color": "#38BDF8",
    },
    {
        "id": "IFIN-HH-MAG",
        "name": "IFIN-HH Nuclear Grid Node (Măgurele)",
        "short": "IFIN-HH",
        "capacity_mw": 5.0,
        "cooling_c": 27.2,
        "queue": 30,
        "color": "#FF4B4B",
    },
    {
        "id": "UTCN-HPC-CLJ",
        "name": "UTCN High-Performance Node (Cluj)",
        "short": "UTCN HPC",
        "capacity_mw": 2.5,
        "cooling_c": 19.8,
        "queue": 12,
        "color": "#A78BFA",
    },
]

# =====================================================================
# STYLE
# =====================================================================
st.markdown(
    """
<style>
#MainMenu, footer, header {visibility: hidden;}
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background-color: #0c0d12 !important;
    color: #E5E7EB;
}
.block-container {padding-top: 1.2rem; padding-bottom: 1.5rem; max-width: 100%;}
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
.vc-alert {
    border-radius: 10px; padding: 10px 16px; margin: 8px 0 12px 0;
    font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace; font-size: 0.85rem;
}
.vc-alert-red {background: rgba(255,75,75,0.12); border: 1px solid rgba(255,75,75,0.5); color: #FF7B7B;}
.vc-alert-green {background: rgba(0,255,157,0.07); border: 1px solid rgba(0,255,157,0.3); color: #00FF9D;}
.vc-modal {
    background: #14161e; border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px; padding: 18px; box-shadow: 0 12px 40px rgba(0,0,0,0.55);
}
.vc-modal-head {
    font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace;
    font-size: 0.78rem; color: #38BDF8; letter-spacing: 0.06em; margin-bottom: 10px;
}
.vc-modal-title {font-size: 1.05rem; font-weight: 700; color: #FFFFFF; margin-bottom: 8px;}
.vc-modal-body {font-size: 0.88rem; color: #D1D5DB; line-height: 1.5;}
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
# DATA, MODEL & AGENT
# =====================================================================


@st.cache_data(show_spinner=False)
def load_feed() -> Tuple[pd.DataFrame, str]:
    """Încarcă CSV-ul cronologic; dacă lipsește, generează 48h realiste la runtime."""
    if Path(DATA_PATH).exists():
        df = pd.read_csv(DATA_PATH)
        source = str(DATA_PATH)
    else:
        df = generate_synthetic_dataset(n_hours=48)
        source = "synthetic runtime feed (48h)"
    df = df.reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["timestamp"])
    if "gross_solar_mw" not in df.columns:
        derating = 1.0 - np.maximum(0.0, (df["ambient_temp_celsius"] - 25.0) * 0.004)
        df["gross_solar_mw"] = df["solar_irradiance_wm2"] / 1000.0 * 17.8 * derating
    return df, source


def ensure_model(agent: VoltComputeAgent, feed: pd.DataFrame) -> str:
    """Folosește models/model_surplus.pkl; dacă lipsește, antrenează un HistGradientBoosting."""
    if agent.model_artifact is not None and "model" in agent.model_artifact:
        return "model_surplus.pkl"
    if TARGET_COLUMN in feed.columns:
        engineered = add_engineered_features(feed)
        model = HistGradientBoostingRegressor(max_iter=200, max_depth=5, learning_rate=0.05, random_state=42)
        model.fit(engineered[ENGINEERED_FEATURE_COLUMNS], engineered[TARGET_COLUMN])
        agent.model_artifact = {"model": model, "feature_columns": ENGINEERED_FEATURE_COLUMNS}
        return "HistGradientBoosting (runtime fallback)"
    return "physics fallback"


feed, feed_source = load_feed()
N_ROWS = len(feed)
MAX_STEP = max(0, N_ROWS - 1 - LOOKAHEAD_HOURS)
# Scală verticală fixă pentru grafic, ca axa să nu "sară" la fiecare tick (ca pe un terminal de bursă)
Y_AXIS_MAX = float(feed["gross_solar_mw"].max()) * 1.08 + 0.5

if "agent" not in st.session_state:
    st.session_state.agent = VoltComputeAgent()
    st.session_state.model_source = ensure_model(st.session_state.agent, feed)
if "pred_cache" not in st.session_state:
    st.session_state.pred_cache = {}
if "autoplay" not in st.session_state:
    st.session_state.autoplay = False
if "pending_advance" not in st.session_state:
    st.session_state.pending_advance = False
if "accepted" not in st.session_state:
    st.session_state.accepted = {}
if "current_step" not in st.session_state:
    candidates = feed[(feed["hour"] == 13) & (feed.get("surplus_mw", 0) > 4.0)] if "surplus_mw" in feed.columns else feed.iloc[0:0]
    if len(candidates) > 0:
        default_step = int(candidates.index[0])
    else:
        noon = feed[feed["hour"].isin([12, 13])]
        default_step = int(noon.index[0]) if len(noon) > 0 else 0
    st.session_state.current_step = int(min(default_step, MAX_STEP))

# Avansarea automată trebuie aplicată înainte ca widgetul slider să fie creat
if st.session_state.pending_advance:
    st.session_state.pending_advance = False
    st.session_state.current_step = int(min(MAX_STEP, st.session_state.current_step + 1))

agent: VoltComputeAgent = st.session_state.agent


def net_injection_mw(row: pd.Series) -> float:
    local_load = 1.4 + 0.5 * float(np.sin(np.pi * int(row["hour"]) / 12.0))
    return round(max(0.0, float(row["gross_solar_mw"]) - local_load), 3)


def predict_step(idx: int) -> Tuple[float, float, Dict[str, float]]:
    """Surplus prezis de model (agent.estimate) pentru rândul `idx` din CSV."""
    cache = st.session_state.pred_cache
    if idx not in cache:
        row = feed.iloc[idx]
        grid = GridTelemetry(
            timestamp=str(row["timestamp"]),
            hour=int(row["hour"]),
            solar_irradiance_wm2=float(row["solar_irradiance_wm2"]),
            ambient_temp_celsius=float(row["ambient_temp_celsius"]),
            day_ahead_price_eur=float(row["day_ahead_price_eur"]),
            grid_capacity_limit_mw=float(row.get("grid_capacity_limit_mw", 10.0)),
            is_weekend=int(row["dt"].day_of_week >= 5),
        )
        cache[idx] = agent.estimate(grid)
    return cache[idx]


def allocate(surplus: float, confidence: float) -> List[Dict[str, Any]]:
    """
    Împarte surplusul între nodurile valide, proporțional cu coada de joburi
    (nodul cu coada cea mai mare primește cel mai mult), respectând capacitatea fiecărui nod.
    Nodurile peste limita termică sunt respinse automat.
    """
    results = []
    for node in DATACENTERS:
        results.append({**node, "allocated_mw": 0.0, "status": "STANDBY", "jobs": 0})

    valid = [r for r in results if r["cooling_c"] <= agent.THERMAL_LIMIT_CELSIUS]
    for r in results:
        if r["cooling_c"] > agent.THERMAL_LIMIT_CELSIUS:
            r["status"] = "REJECTED_THERMAL_SAFETY"

    if surplus > agent.MIN_SURPLUS_THRESHOLD_MW and confidence < agent.CONFIDENCE_THRESHOLD:
        for r in valid:
            r["status"] = "ESCALATE_TO_HUMAN"
        return results

    if surplus > agent.MIN_SURPLUS_THRESHOLD_MW:
        remaining = min(surplus, sum(r["capacity_mw"] for r in valid))
        active = list(valid)
        for _ in range(8):
            active = [r for r in active if r["allocated_mw"] < r["capacity_mw"] - 1e-9]
            if remaining < 0.005 or not active:
                break
            weight_sum = sum(r["queue"] for r in active)
            given = 0.0
            for r in active:
                share = remaining * r["queue"] / weight_sum
                grant = min(share, r["capacity_mw"] - r["allocated_mw"])
                r["allocated_mw"] += grant
                given += grant
            remaining -= given
        for r in valid:
            r["allocated_mw"] = round(r["allocated_mw"], 2)
            if r["allocated_mw"] > 0:
                r["status"] = "OFFER_GENERATED"
                r["jobs"] = max(1, int(round(r["queue"] * r["allocated_mw"] / r["capacity_mw"])))
    return results


# =====================================================================
# HEADER + TIMELINE CONTROL BAR
# =====================================================================
st.markdown(
    f"""
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
  <div class="vc-title"><span class="vc-live"></span>VOLTCOMPUTE // PPC GRID DISPATCH</div>
  <div class="vc-sub">FEED: {feed_source} · MODEL: {st.session_state.model_source} · {N_ROWS}h</div>
</div>
""",
    unsafe_allow_html=True,
)


def _step_back() -> None:
    st.session_state.autoplay = False
    st.session_state.current_step = max(0, int(st.session_state.current_step) - 1)


def _step_forward() -> None:
    st.session_state.autoplay = False
    st.session_state.current_step = min(MAX_STEP, int(st.session_state.current_step) + 1)


def _toggle_play() -> None:
    st.session_state.autoplay = not st.session_state.autoplay


bar_back, bar_fwd, bar_play, bar_slider = st.columns([1, 1, 1.7, 5], gap="small")
bar_back.button("◀ Pas Înapoi", on_click=_step_back, width="stretch")
bar_fwd.button("▶ Pas Înainte", on_click=_step_forward, width="stretch")
bar_play.button(
    "⏹ Stop Live Feed" if st.session_state.autoplay else "▶▶ Play Live Feed (Auto-step 1s)",
    on_click=_toggle_play,
    type="primary",
    width="stretch",
)
with bar_slider:
    st.select_slider(
        "Timeline",
        options=list(range(MAX_STEP + 1)),
        key="current_step",
        format_func=lambda i: feed.iloc[i]["dt"].strftime("%Y-%m-%d %H:%M"),
        label_visibility="collapsed",
    )

# =====================================================================
# STATE AT STEP T + LOOKAHEAD T+1..T+4
# =====================================================================
T = int(st.session_state.current_step)
row_T = feed.iloc[T]
surplus_T, conf_T, shap_T = predict_step(T)
allocs_T = allocate(surplus_T, conf_T)
total_alloc_T = round(sum(a["allocated_mw"] for a in allocs_T), 2)
revenue_T = round(total_alloc_T * FIXED_PRICE_EUR, 2)

is_accepted = (st.session_state.accepted.get(T) == "ACCEPTED")

# Calcul profit cumulat doar pentru tranzacțiile deja executate la ora curentă
def _get_alloc_for_step(step_idx: int) -> float:
    s_val, c_val, _ = predict_step(step_idx)
    return round(sum(a["allocated_mw"] for a in allocate(s_val, c_val)), 2)

cumulative_profit = sum(
    _get_alloc_for_step(idx) * FIXED_PRICE_EUR
    for idx, status in st.session_state.accepted.items()
    if status == "ACCEPTED" and idx <= T
)

# Curentul (injecția netă) și surplusul înainte vs după dispecerizare
raw_injection_T = net_injection_mw(row_T)
actual_injection_T = max(0.0, raw_injection_T - total_alloc_T) if is_accepted else raw_injection_T
grid_limit_mw = float(row_T.get("grid_capacity_limit_mw", 10.0))
actual_surplus_T = max(0.0, actual_injection_T - grid_limit_mw) if is_accepted else surplus_T
actual_profit_T = revenue_T if is_accepted else 0.0

future_idx = [i for i in range(T + 1, min(N_ROWS, T + 1 + LOOKAHEAD_HOURS))]
future_surplus = {i: predict_step(i)[0] for i in future_idx}
upcoming_spikes = [i for i in future_idx if predict_step(i)[0] > agent.MIN_SURPLUS_THRESHOLD_MW]
has_upcoming_spike = len(upcoming_spikes) > 0
first_spike_idx = upcoming_spikes[0] if has_upcoming_spike else None
is_first_spike_prebooked = (st.session_state.accepted.get(first_spike_idx) == "ACCEPTED") if first_spike_idx is not None else False

peak_candidates = [(T, surplus_T)] + [(i, s) for i, s in future_surplus.items()]
peak_idx, peak_mw = max(peak_candidates, key=lambda x: x[1])

if is_accepted and total_alloc_T > 0:
    alert_html = (
        f'<div class="vc-alert vc-alert-green">✅ REȚEA STABILIZATĂ: Surplusul de {total_alloc_T:.2f} MW '
        f'a fost evacuat la nodurile universitare la {row_T["dt"].strftime("%H:%M")}! '
        f'Curentul a scăzut de la {raw_injection_T:.2f} MW la {actual_injection_T:.2f} MW · '
        f'Profit PPC: +{actual_profit_T:,.2f} €</div>'
    )
elif has_upcoming_spike and total_alloc_T == 0:
    s_lead = predict_step(first_spike_idx)[0]
    lead_h = first_spike_idx - T
    t_lead_str = feed.iloc[first_spike_idx]["dt"].strftime("%H:%M")
    if is_first_spike_prebooked:
        alert_html = (
            f'<div class="vc-alert vc-alert-green">✅ PRE-DISPECERIZARE ACTIVĂ (T+{lead_h}): '
            f'Vârful de {s_lead:.1f} MW de la ora {t_lead_str} (în {lead_h}h) este deja PRE-REZERVAT! '
            f'Capacitatea academică e alocată, iar prognoza e stabilizată în siguranță sub 10 MW.</div>'
        )
    else:
        alert_html = (
            f'<div class="vc-alert vc-alert-red">🔔 PREAVIZ CURTAILMENT (T+{lead_h}h): '
            f'Modelul AI prognozează un spike critic de {s_lead:.1f} MW la ora {t_lead_str} '
            f'(peste limita liniei de {grid_limit_mw:.0f} MW) · Rezervă capacitatea universitară cu 2 ore înainte!</div>'
        )
elif peak_mw > 0.05:
    alert_html = (
        f'<div class="vc-alert vc-alert-red">⚠️ ALERTĂ CURTAILMENT: Vârf de surplus de {peak_mw:.1f} MW '
        f'detectat în următoarele ore (la {feed.iloc[peak_idx]["dt"].strftime("%H:%M")}, peste limita liniei de '
        f'{grid_limit_mw:.0f} MW) · Rezervă dispecerizarea pentru evacuare și profit!</div>'
    )
else:
    alert_html = (
        f'<div class="vc-alert vc-alert-green">● GRID NOMINAL: nicio depășire a limitei liniei prognozată '
        f'în următoarele {LOOKAHEAD_HOURS} ore</div>'
    )

col_main, col_modal = st.columns([0.72, 0.28], gap="large")

# =====================================================================
# MAIN COLUMN
# =====================================================================
with col_main:
    st.markdown(alert_html, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Iradiere Curentă",
        f"{float(row_T['solar_irradiance_wm2']):.0f} W/m²",
        f"{row_T['dt'].strftime('%H:%M')} · {float(row_T['ambient_temp_celsius']):.1f}°C",
    )
    if is_accepted:
        m2.metric(
            "Injecție Rețea PPC",
            f"{actual_injection_T:.2f} MW",
            f"-{total_alloc_T:.2f} MW evacuați (stabilizată ✅)",
            delta_color="normal",
        )
        m3.metric(
            "Surplus Rețea Rămas",
            f"{actual_surplus_T:.2f} MW",
            f"0.00 MW risc curtailment ✅",
            delta_color="normal",
        )
        m4.metric(
            "Profit Arbitraj PPC",
            f"+{cumulative_profit:,.2f} €",
            f"+{actual_profit_T:,.2f} € încasat la livrare",
            delta_color="normal",
        )
    else:
        m2.metric(
            "Injecție Rețea PPC",
            f"{raw_injection_T:.2f} MW",
            f"limită linie: {grid_limit_mw:.0f} MW",
            delta_color="inverse" if raw_injection_T > grid_limit_mw else "off",
        )
        m3.metric(
            "Surplus de Arbitrat",
            f"{surplus_T:.2f} MW",
            f"{total_alloc_T:.2f} MW ofertabili" if total_alloc_T > 0 else "nominal",
            delta_color="inverse" if surplus_T > 2.0 else "off",
        )
        m4.metric(
            "Profit Arbitraj PPC",
            f"+{cumulative_profit:,.2f} €",
            f"+{revenue_T:,.2f} € la livrarea curentă" if revenue_T > 0 else f"PZU {float(row_T['day_ahead_price_eur']):.1f} €/MWh",
            delta_color="normal" if revenue_T > 0 else "off",
        )

    # ---------------- Live ticker chart ----------------
    start_i = max(0, T - HISTORY_HOURS)
    end_i = min(N_ROWS - 1, T + LOOKAHEAD_HOURS)
    window = list(range(start_i, end_i + 1))
    xs = [feed.iloc[i]["dt"].to_pydatetime() for i in window]
    limit_line = [float(feed.iloc[i].get("grid_capacity_limit_mw", 10.0)) for i in window]

    def _dispatched_inj_for_step(step_idx: int) -> float:
        raw = net_injection_mw(feed.iloc[step_idx])
        if st.session_state.accepted.get(step_idx) == "ACCEPTED":
            alloc_val = _get_alloc_for_step(step_idx)
            return max(0.0, raw - alloc_val)
        return raw

    inj_hist = [_dispatched_inj_for_step(i) if i <= T else None for i in window]
    inj_raw_hist = [net_injection_mw(feed.iloc[i]) if i <= T else None for i in window]
    inj_fcst = [_dispatched_inj_for_step(i) if i >= T else None for i in window]
    inj_raw_fcst = [net_injection_mw(feed.iloc[i]) if i >= T else None for i in window]

    def _dispatched_surplus_for_step(step_idx: int) -> float:
        if st.session_state.accepted.get(step_idx) == "ACCEPTED":
            disp = _dispatched_inj_for_step(step_idx)
            lim = float(feed.iloc[step_idx].get("grid_capacity_limit_mw", 10.0))
            return max(0.0, disp - lim)
        return predict_step(step_idx)[0]

    sur_pred = [_dispatched_surplus_for_step(i) if i >= T else None for i in window]

    t_x = row_T["dt"].to_pydatetime()
    live_value = actual_injection_T
    prev_value = _dispatched_inj_for_step(T - 1) if T > 0 else live_value
    tick_delta = live_value - prev_value
    tick_color = "#00FF9D" if is_accepted else ("#00FF9D" if tick_delta >= 0 else "#FF4B4B")
    tick_arrow = "▲" if tick_delta >= 0 else "▼"
    if is_accepted:
        status_badge = '<span class="vc-badge-live" style="background:rgba(0,255,157,0.18);color:#00FF9D;border:1px solid rgba(0,255,157,0.5);">● SURPLUS ABSORBIT</span>'
    elif st.session_state.autoplay:
        status_badge = '<span class="vc-badge-live">● LIVE</span>'
    else:
        status_badge = '<span class="vc-badge-paused">❚❚ PAUSED</span>'

    st.markdown(
        f"""
<div class="vc-ticker">
  <div>{status_badge}<span class="vc-ticker-meta">PPC SOLAR INJECTION · NET MW</span></div>
  <div>
    <span class="vc-ticker-price" style="color:{tick_color};">{live_value:.2f} MW</span>
    <span class="vc-ticker-meta" style="color:{tick_color};margin-left:8px;">{tick_arrow} {abs(tick_delta):.2f} vs T-1h · {t_x.strftime('%d %b %H:%M')}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs, y=inj_hist, mode="lines", name="Injecție netă PPC",
            line=dict(color="#00FF9D", width=2.8, shape="spline", smoothing=0.6),
            fill="tozeroy", fillcolor="rgba(0,255,157,0.08)",
        )
    )
    if is_accepted:
        fig.add_trace(
            go.Scatter(
                x=xs, y=inj_raw_hist, mode="lines", name="Producție brută (fără arbitraj)",
                line=dict(color="#8E95A5", width=1.5, dash="dot"),
            )
        )
    any_future_prebooked = any(st.session_state.accepted.get(i) == "ACCEPTED" for i in future_idx)
    if any_future_prebooked:
        fig.add_trace(
            go.Scatter(
                x=xs, y=inj_raw_fcst, mode="lines", name="Prognoză brută (vârf nepreluat)",
                line=dict(color="#F87171", width=1.5, dash="dot"),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=xs, y=inj_fcst, mode="lines", name="Producție prognozată (T..T+4)",
            line=dict(color="#38BDF8", width=2.2, dash="dash", shape="spline", smoothing=0.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs, y=limit_line, mode="lines", name="Limită linie PPC (10 MW)",
            line=dict(color="#FF4B4B", width=2.5),
        )
    )
    # Punctul "live" de la capătul curbei, ca ultimul preț pe un terminal de bursă
    fig.add_trace(
        go.Scatter(
            x=[t_x], y=[live_value], mode="markers", name="Tick curent",
            marker=dict(size=13, color=tick_color, line=dict(color="#FFFFFF", width=2)),
            showlegend=False, hoverinfo="skip",
        )
    )
    fig.add_hline(
        y=live_value, line_width=1, line_dash="dot", line_color=tick_color, opacity=0.6,
        annotation_text=f" {live_value:.2f} MW ", annotation_position="right",
        annotation_font=dict(color="#0c0d12", size=11),
        annotation_bgcolor=tick_color,
    )
    fig.add_vrect(
        x0=t_x, x1=t_x + timedelta(hours=LOOKAHEAD_HOURS), fillcolor="rgba(142,149,165,0.08)", line_width=0,
        annotation_text="PROGNOZĂ ML", annotation_position="top left",
        annotation_font=dict(color="#F59E0B", size=11),
    )
    fig.add_shape(
        type="line", x0=t_x, x1=t_x, y0=0, y1=1, yref="paper",
        line=dict(color="#FF4B4B", width=2, dash="dot"),
    )
    fig.add_annotation(
        x=t_x, y=1, yref="paper", text=f"T = {t_x.strftime('%H:%M')}", showarrow=False,
        yanchor="bottom", font=dict(color="#FF7B7B", size=11),
    )

    # Anotații pe grafic pentru viitor (T+1..T+4)
    for f_i in future_idx:
        f_dt = feed.iloc[f_i]["dt"].to_pydatetime()
        s_f = predict_step(f_i)[0]
        if st.session_state.accepted.get(f_i) == "ACCEPTED":
            fig.add_annotation(
                x=f_dt, y=limit_line[window.index(f_i)],
                text=f"✅ T+{f_i - T} REZERVAT",
                showarrow=True, arrowhead=2, arrowcolor="#00FF9D",
                font=dict(color="#00FF9D", size=10),
                bgcolor="#12141b", bordercolor="#00FF9D",
            )
        elif s_f > agent.MIN_SURPLUS_THRESHOLD_MW:
            fig.add_annotation(
                x=f_dt, y=net_injection_mw(feed.iloc[f_i]),
                text=f"⚠️ VÂRF T+{f_i - T} ({s_f:.1f} MW)",
                showarrow=True, arrowhead=2, arrowcolor="#FF4B4B",
                font=dict(color="#FF7B7B", size=10),
                bgcolor="#12141b", bordercolor="#FF4B4B",
            )
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#8E95A5", size=11),
        height=580,
        margin=dict(l=10, r=60, t=30, b=10),
        hovermode="x unified",
        uirevision="voltcompute-live",
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            range=[t_x - timedelta(hours=HISTORY_HOURS), t_x + timedelta(hours=LOOKAHEAD_HOURS)],
            showspikes=True, spikecolor="rgba(255,255,255,0.35)", spikethickness=1, spikedash="dot",
        ),
        yaxis=dict(
            title="MW", side="right", range=[0, Y_AXIS_MAX],
            gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)",
            showspikes=True, spikecolor="rgba(255,255,255,0.35)", spikethickness=1, spikedash="dot",
        ),
    )
    st.plotly_chart(fig, width="stretch")

    # ---------------- Arbitrage table ----------------
    st.markdown('<div class="vc-sub" style="margin:4px 0;">ARBITRAJ MULTI-NOD · ORA T</div>', unsafe_allow_html=True)
    table = pd.DataFrame(
        [
            {
                "Nod": a["name"],
                "Temp Răcire (°C)": a["cooling_c"],
                "Capacitate (MW)": a["capacity_mw"],
                "Status Agent": a["status"],
                "Ofertă (MW)": a["allocated_mw"],
                "Preț Fix (€/MWh)": FIXED_PRICE_EUR if a["allocated_mw"] > 0 else 0.0,
                "Profit PPC (€)": round(a["allocated_mw"] * FIXED_PRICE_EUR, 2),
            }
            for a in allocs_T
        ]
    )

    def _status_color(value: str) -> str:
        palette = {
            "OFFER_GENERATED": "color:#00FF9D;font-weight:600",
            "REJECTED_THERMAL_SAFETY": "color:#FF4B4B;font-weight:600",
            "ESCALATE_TO_HUMAN": "color:#F59E0B;font-weight:600",
        }
        return palette.get(value, "color:#8E95A5")

    st.dataframe(
        table.style.map(_status_color, subset=["Status Agent"]).format(
            {"Temp Răcire (°C)": "{:.1f}", "Capacitate (MW)": "{:.1f}", "Ofertă (MW)": "{:.2f}", "Preț Fix (€/MWh)": "{:.2f}", "Profit PPC (€)": "{:,.2f}"}
        ),
        hide_index=True,
        width="stretch",
    )

# =====================================================================
# SECONDARY COLUMN: DISPATCH & ACADEMIC BATTERY VIEW
# =====================================================================
with col_modal:
    st.markdown('<div class="vc-sub" style="margin-bottom:8px;">DISPECERIZARE & BATERIE VIRTUALĂ</div>', unsafe_allow_html=True)
    upb = next(a for a in allocs_T if a["id"] == "ROEDUNET-UPB-BUC")

    # CAZUL 1: Există surplus activ la ora curentă T
    if total_alloc_T > 0:
        window_hours = 1
        for i in future_idx:
            if future_surplus[i] > agent.MIN_SURPLUS_THRESHOLD_MW:
                window_hours += 1
            else:
                break

        status_card = (
            '<div style="color:#00FF9D;font-weight:700;">● TRANZACȚIE PRE-REZERVATĂ (REȚEA STABILIZATĂ)</div>'
            if is_accepted
            else '<div style="color:#F59E0B;font-weight:700;">⚠️ OFERTĂ DISPONIBILĂ PENTRU PRE-REZERVARE</div>'
        )

        st.markdown(
            f"""
<div class="vc-modal">
  <div class="vc-modal-head">⚡ DISPECERAT PPC // CONTROL SURPLUS (ORA CURENTĂ)</div>
  <div class="vc-modal-title">ACORD ARBITRAJ ENERGETIC · {row_T['dt'].strftime('%H:%M')}</div>
  <div class="vc-modal-body">
    {status_card}
    <br>
    PPC redirecționează <span class="vc-mono" style="color:#00FF9D;">{total_alloc_T:.2f} MW</span> surplus către nodurile academice la
    <span class="vc-mono" style="color:#00FF9D;">{FIXED_PRICE_EUR:.2f} €/MWh</span>.
    <br><br>
    💰 <b>Profit la livrare:</b> <span class="vc-mono" style="color:#00FF9D;font-size:1.05rem;">+{revenue_T:,.2f} €</span>
    <br>
    📉 <b>Degrevare linie PPC:</b> <span class="vc-mono" style="color:#38BDF8;">-{total_alloc_T:.2f} MW</span> (curentul scade sub 10 MW)
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        k1, k2 = st.columns(2)
        if is_accepted:
            k1.metric("Profit Încasat", f"+{revenue_T:,.2f} €", delta="Curent livrat")
            k2.metric("Degrevare Curent", f"-{total_alloc_T:.2f} MW", delta="Rețea stabilă")
        else:
            k1.metric("Profit la Livrare", f"+{revenue_T:,.2f} €", delta="În așteptare")
            k2.metric("Surplus de Evacuat", f"{total_alloc_T:.2f} MW", delta="Depășire linie", delta_color="inverse")

        if is_accepted:
            st.success(f"✅ Curent livrat! +{revenue_T:,.2f} € profit încasat, surplus evacuat: {total_alloc_T:.2f} MW.")
        else:
            if st.button(f"⚡ Rezervă energia în avans (+{revenue_T:,.2f} €)", key=f"book_{T}", type="primary", width="stretch"):
                st.session_state.accepted[T] = "ACCEPTED"
                st.toast("✅ Energia a fost rezervată în avans. Profitul se încasează la livrare.")
                st.rerun()

    # CAZUL 2: La ora curentă T nu e surplus, DAR urmează un spike în lookahead (T+2)
    elif has_upcoming_spike:
        spike_row = feed.iloc[first_spike_idx]
        spike_dt = spike_row["dt"]
        lead_h = first_spike_idx - T
        s_spike, c_spike, _ = predict_step(first_spike_idx)
        allocs_spike = allocate(s_spike, c_spike)
        alloc_spike_mw = round(sum(a["allocated_mw"] for a in allocs_spike), 2)
        profit_spike = round(alloc_spike_mw * FIXED_PRICE_EUR, 2)
        is_spike_prebooked = (st.session_state.accepted.get(first_spike_idx) == "ACCEPTED")

        status_card = (
            '<div style="color:#00FF9D;font-weight:700;">● PRE-REZERVARE ACTIVĂ (VÂRF TĂIAT ÎN AVANS)</div>'
            if is_spike_prebooked
            else f'<div style="color:#F59E0B;font-weight:700;">⚠️ PREAVIZ DE REȚEA: SPIKE PESTE {lead_h} ORE</div>'
        )

        st.markdown(
            f"""
<div class="vc-modal">
  <div class="vc-modal-head">🔔 PRE-DISPECERIZARE ANTICIPATĂ // LOOKAHEAD T+{lead_h}h</div>
  <div class="vc-modal-title">SPIKE SOLAR PROGNOZAT LA {spike_dt.strftime('%H:%M')}</div>
  <div class="vc-modal-body">
    {status_card}
    <br>
    Modelul AI prognozează un surplus critic de <span class="vc-mono" style="color:#FF7B7B;">{s_spike:.2f} MW</span>
    la ora <span class="vc-mono" style="color:#FFFFFF;">{spike_dt.strftime('%H:%M')}</span> (în {lead_h} ore), depășind limita liniei de 10 MW!
    <br><br>
    ⚡ <b>De ce rezervăm cu 2 ore înainte?</b><br>
    Permite clusterului UPB să coboare temperatura camerelor de servere și să aloce nodurile Slurm în avans, fără șoc termic.
    <br><br>
    💰 <b>Valoare la livrare:</b> <span class="vc-mono" style="color:#00FF9D;font-size:1.05rem;">+{profit_spike:,.2f} €</span>
    <br>
    📉 <b>Degrevare viitoare prognozată:</b> <span class="vc-mono" style="color:#38BDF8;">-{alloc_spike_mw:.2f} MW</span> (vârf tăiat în avans)
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        k1, k2 = st.columns(2)
        if is_spike_prebooked:
            k1.metric(f"Profit la Livrare ({spike_dt.strftime('%H:%M')})", f"+{profit_spike:,.2f} €", delta="Rezervat ✅")
            k2.metric(f"Vârf Tăiat ({spike_dt.strftime('%H:%M')})", f"-{alloc_spike_mw:.2f} MW", delta="Sub 10 MW ✅")
            st.success(f"✅ Vârf rezervat pentru ora {spike_dt.strftime('%H:%M')}! Curba din grafic este tăiată în avans.")
        else:
            k1.metric("Valoare la Livrare", f"+{profit_spike:,.2f} €", delta=f"La {spike_dt.strftime('%H:%M')}")
            k2.metric("Spike Prognozat", f"{s_spike:.2f} MW", delta=f"În {lead_h}h", delta_color="inverse")
            if st.button(f"⚡ Rezervă energia în avans ({spike_dt.strftime('%H:%M')}) (+{profit_spike:,.2f} €)", key=f"book_spike_{first_spike_idx}", type="primary", width="stretch"):
                st.session_state.accepted[first_spike_idx] = "ACCEPTED"
                st.toast(f"✅ Energia a fost rezervată în avans pentru ora {spike_dt.strftime('%H:%M')}.")
                st.rerun()

    # CAZUL 3: Nici la T și nici în viitorul apropiat nu se prognozează surplus
    else:
        st.markdown(
            f"""
<div class="vc-modal">
  <div class="vc-modal-head">⚡ DISPECERAT PPC // REȚEA NOMINALĂ</div>
  <div class="vc-modal-title">FĂRĂ CONGESTIE PROGNOZATĂ</div>
  <div class="vc-modal-body">
    Nici la ora {row_T['dt'].strftime('%H:%M')} și nici în următoarele {LOOKAHEAD_HOURS} ore nu se prognozează depășirea limitei liniei PPC de 10.0 MW. Sistemul funcționează în regim nominal standard.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

# =====================================================================
# AUTO-STEP
# =====================================================================
if st.session_state.autoplay:
    time.sleep(AUTOPLAY_SECONDS)
    if st.session_state.current_step < MAX_STEP:
        st.session_state.pending_advance = True
    else:
        st.session_state.autoplay = False
    st.rerun()
