import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agent import VoltComputeAgent
from schemas import GridTelemetry
from simulator import VoltComputeSimulator
from train import (
    LIVE_TEST_DATA_PATH,
    MODEL_PATH,
    TRAIN_DATA_PATH,
    ensure_split_datasets,
    train_pipeline,
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
# DATA (TRAIN vs LIVE OUT-OF-SAMPLE RING BUFFER), MODEL & AGENT
# =====================================================================


@st.cache_data(show_spinner=False)
def load_feed() -> Tuple[pd.DataFrame, int, str]:
    """
    Încarcă EXCLUSIV setul de date noi (out-of-sample) `data/live_test_energy_data.csv`
    (pe care modelul NU a fost antrenat niciodată) pentru a fi rulat în Ring Buffer continuu.
    """
    train_df, live_df = ensure_split_datasets()
    live_df = live_df.reset_index(drop=True)
    live_df["dt"] = pd.to_datetime(live_df["timestamp"])
    if "gross_solar_mw" not in live_df.columns:
        derating = 1.0 - np.maximum(0.0, (live_df["ambient_temp_celsius"] - 25.0) * 0.004)
        live_df["gross_solar_mw"] = live_df["solar_irradiance_wm2"] / 1000.0 * 17.8 * derating
    if "net_injection_mw" not in live_df.columns:
        local_load = 1.4 + 0.5 * np.sin(np.pi * live_df["hour"] / 12.0)
        live_df["net_injection_mw"] = np.maximum(0.0, live_df["gross_solar_mw"] - local_load)
    source = f"{LIVE_TEST_DATA_PATH} (Date Noi Out-of-Sample)"
    return live_df, len(train_df), source


def ensure_model(agent: VoltComputeAgent) -> str:
    """
    Asigură că modelul este antrenat EXCLUSIV pe setul de antrenare `train_energy_data.csv`
    (1152h) și niciodată pe setul de date noi din live feed.
    """
    if (
        agent.model_artifact is None
        or "model" not in agent.model_artifact
        or "injection_model" not in agent.model_artifact
    ):
        train_pipeline()
        agent._load_model()
    if agent.model_artifact is not None and "model" in agent.model_artifact:
        return f"model_surplus.pkl (antrenat pe {TRAIN_DATA_PATH.name})"
    return "physics fallback"


feed, N_TRAIN_ROWS, feed_source = load_feed()
N_ROWS = len(feed)  # 288h (12 zile complete 00:00 -> 23:00 pentru Ring Buffer fără discontinuitate)
BASE_LIVE_DT: datetime = feed.iloc[0]["dt"].to_pydatetime()
# Scală verticală fixă pentru grafic, ca axa să nu "sară" la fiecare tick (ca pe un terminal de bursă)
Y_AXIS_MAX = float(feed["gross_solar_mw"].max()) * 1.08 + 0.5

if (
    "agent" not in st.session_state
    or not hasattr(st.session_state.agent, "estimate_injection")
    or "injection_model" not in (st.session_state.agent.model_artifact or {})
):
    st.session_state.agent = VoltComputeAgent(model_path=MODEL_PATH)
    st.session_state.model_source = ensure_model(st.session_state.agent)
    st.session_state.pred_cache = {}

if "pred_cache" not in st.session_state:
    st.session_state.pred_cache = {}
if "autoplay" not in st.session_state:
    st.session_state.autoplay = False
if "pending_advance" not in st.session_state:
    st.session_state.pending_advance = False
if "accepted" not in st.session_state:
    st.session_state.accepted = {}
if "ring_cycle" not in st.session_state:
    st.session_state.ring_cycle = 0
if "current_step" not in st.session_state:
    candidates = (
        feed[(feed["hour"] == 13) & (feed.get("surplus_mw", 0) > 3.5)]
        if "surplus_mw" in feed.columns
        else feed.iloc[0:0]
    )
    if len(candidates) > 1:
        default_step = int(candidates.index[1])
    elif len(candidates) > 0:
        default_step = int(candidates.index[0])
    else:
        noon = feed[feed["hour"].isin([12, 13])]
        default_step = int(noon.index[0]) if len(noon) > 0 else 13
    st.session_state.current_step = int(default_step % N_ROWS)

# Avansarea automată în Ring Buffer (fără oprire la capăt; trece lin din 23:00 în 00:00)
if st.session_state.pending_advance:
    st.session_state.pending_advance = False
    nxt_step = int(st.session_state.current_step) + 1
    if nxt_step >= N_ROWS:
        st.session_state.current_step = 0
        st.session_state.ring_cycle = int(st.session_state.ring_cycle) + 1
    else:
        st.session_state.current_step = nxt_step

agent: VoltComputeAgent = st.session_state.agent


def _ring_row(v_idx: int) -> pd.Series:
    """Returnează rândul din Ring Buffer corespunzător indexului virtual continuu `v_idx`."""
    return feed.iloc[v_idx % N_ROWS]


def _ring_dt(v_idx: int) -> datetime:
    """Returnează timestamp-ul continuu monoton pentru indexul virtual `v_idx` din Ring Buffer."""
    return BASE_LIVE_DT + timedelta(hours=int(v_idx))


def _cycle_variation(v_idx: int) -> Tuple[float, float, float]:
    """
    La ciclurile ulterioare ale Ring Buffer-ului (`v_cycle != 0`), aplică o micro-variație
    stocastică deterministă (nor tranzitoriu + fluctuație de sarcină) astfel încât fluxul live
    să genereze mereu date noi, fără să repete identic aceiași megawați.
    """
    v_cycle = v_idx // N_ROWS
    if v_cycle == 0:
        return 1.0, 0.0, 0.0
    rng = np.random.default_rng(seed=100_003 + int(v_idx) * 31)
    irr_factor = float(np.clip(1.0 + rng.normal(0.0, 0.045), 0.88, 1.12))
    temp_offset = float(rng.normal(0.0, 0.6))
    inj_noise = float(rng.normal(0.0, 0.28))
    return irr_factor, temp_offset, inj_noise


def net_injection_mw(v_idx: int) -> float:
    """
    Valoarea REALĂ (Ground Truth) a injecției nete măsurate în rețea la pasul `v_idx`
    din setul de date noi (include fluctuațiile reale de nori și consum local).
    """
    row = _ring_row(v_idx)
    irr_factor, _, inj_noise = _cycle_variation(v_idx)
    base_real_inj = float(row["net_injection_mw"])
    if int(row["hour"]) < 6 or int(row["hour"]) > 19 or base_real_inj <= 0.01:
        return 0.0
    return round(max(0.0, base_real_inj * irr_factor + inj_noise), 3)


def real_surplus_mw(v_idx: int) -> float:
    """Surplusul REAL măsurat în rețea peste limita liniei la pasul `v_idx`."""
    row = _ring_row(v_idx)
    lim = float(row.get("grid_capacity_limit_mw", 10.0))
    return round(max(0.0, net_injection_mw(v_idx) - lim), 3)


def predict_step(v_idx: int) -> Tuple[float, float, Dict[str, float], float]:
    """
    Calculează predicția modelului AI (antrenat DOAR pe train_energy_data.csv)
    pe datele noi din Ring Buffer la pasul `v_idx`:
    returnează (surplus_predicted_mw, confidence, shap_dict, predicted_injection_mw).
    """
    cache = st.session_state.pred_cache
    if v_idx not in cache or len(cache[v_idx]) != 4:
        row = _ring_row(v_idx)
        irr_factor, temp_offset, _ = _cycle_variation(v_idx)
        dt_val = _ring_dt(v_idx)
        irr_val = float(np.clip(float(row["solar_irradiance_wm2"]) * irr_factor, 0.0, 1400.0))
        temp_val = float(np.clip(float(row["ambient_temp_celsius"]) + temp_offset, -20.0, 55.0))
        grid = GridTelemetry(
            timestamp=dt_val.isoformat(),
            hour=int(row["hour"]),
            solar_irradiance_wm2=round(irr_val, 2),
            ambient_temp_celsius=round(temp_val, 2),
            day_ahead_price_eur=float(row["day_ahead_price_eur"]),
            grid_capacity_limit_mw=float(row.get("grid_capacity_limit_mw", 10.0)),
            is_weekend=int(dt_val.weekday() >= 5),
        )
        surplus_pred, conf, shap_dict = agent.estimate(grid)
        inj_pred = agent.estimate_injection(grid)
        cache[v_idx] = (surplus_pred, conf, shap_dict, inj_pred)
    return cache[v_idx]


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
def _step_back() -> None:
    st.session_state.autoplay = False
    prv = int(st.session_state.current_step) - 1
    if prv < 0:
        st.session_state.current_step = N_ROWS - 1
        st.session_state.ring_cycle = int(st.session_state.ring_cycle) - 1
    else:
        st.session_state.current_step = prv


def _step_forward() -> None:
    st.session_state.autoplay = False
    nxt = int(st.session_state.current_step) + 1
    if nxt >= N_ROWS:
        st.session_state.current_step = 0
        st.session_state.ring_cycle = int(st.session_state.ring_cycle) + 1
    else:
        st.session_state.current_step = nxt


def _toggle_play() -> None:
    st.session_state.autoplay = not st.session_state.autoplay


abs_T = int(st.session_state.ring_cycle) * N_ROWS + int(st.session_state.current_step)
# Calculăm eroarea MAE live pe ultimele 24h (doar orele de zi cu activitate solară) pentru afișare transparentă
_day_hist_errors = [
    abs(net_injection_mw(i) - predict_step(i)[3])
    for i in range(abs_T - HISTORY_HOURS, abs_T + 1)
    if 7 <= int(_ring_row(i)["hour"]) <= 18
]
live_window_mae = float(np.mean(_day_hist_errors)) if _day_hist_errors else 0.42

st.markdown(
    f"""
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
  <div class="vc-title"><span class="vc-live"></span>VOLTCOMPUTE // PPC GRID DISPATCH</div>
  <div class="vc-sub">TRAIN SET: {N_TRAIN_ROWS}h · LIVE RING BUFFER: {N_ROWS}h Date Noi (Ciclu #{int(st.session_state.ring_cycle) + 1}) · MAE Live (24h): ±{live_window_mae:.2f} MW</div>
</div>
""",
    unsafe_allow_html=True,
)

bar_back, bar_fwd, bar_play, bar_slider = st.columns([1, 1, 1.7, 5], gap="small")
bar_back.button("◀ Pas Înapoi", on_click=_step_back, width="stretch")
bar_fwd.button("▶ Pas Înainte", on_click=_step_forward, width="stretch")
bar_play.button(
    "⏹ Stop Live Feed" if st.session_state.autoplay else "▶▶ Play Live Feed (Ring Buffer Continuu 1s)",
    on_click=_toggle_play,
    type="primary",
    width="stretch",
)
with bar_slider:
    st.select_slider(
        "Timeline",
        options=list(range(N_ROWS)),
        key="current_step",
        format_func=lambda i: feed.iloc[i]["dt"].strftime("%Y-%m-%d %H:%M"),
        label_visibility="collapsed",
    )

# =====================================================================
# STATE AT STEP T + LOOKAHEAD T+1..T+4 (IN RING BUFFER)
# =====================================================================
T = int(st.session_state.ring_cycle) * N_ROWS + int(st.session_state.current_step)
row_T = _ring_row(T)
dt_T = _ring_dt(T)
surplus_T, conf_T, shap_T, pred_inj_T = predict_step(T)
allocs_T = allocate(surplus_T, conf_T)
total_alloc_T = round(sum(a["allocated_mw"] for a in allocs_T), 2)
revenue_T = round(total_alloc_T * FIXED_PRICE_EUR, 2)

is_accepted = (st.session_state.accepted.get(T) == "ACCEPTED")


def _get_alloc_for_step(step_idx: int) -> float:
    s_val, c_val, _, _ = predict_step(step_idx)
    return round(sum(a["allocated_mw"] for a in allocate(s_val, c_val)), 2)


cumulative_profit = sum(
    _get_alloc_for_step(idx) * FIXED_PRICE_EUR
    for idx, status in st.session_state.accepted.items()
    if status == "ACCEPTED" and idx <= T
)

# Curentul real (injecția netă măsurată în teren) vs predicția AI și după dispecerizare
raw_injection_T = net_injection_mw(T)
actual_injection_T = max(0.0, raw_injection_T - total_alloc_T) if is_accepted else raw_injection_T
grid_limit_mw = float(row_T.get("grid_capacity_limit_mw", 10.0))
actual_surplus_T = max(0.0, actual_injection_T - grid_limit_mw) if is_accepted else surplus_T
actual_profit_T = revenue_T if is_accepted else 0.0

future_idx = [T + k for k in range(1, LOOKAHEAD_HOURS + 1)]
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
        f'a fost evacuat la nodurile universitare la {dt_T.strftime("%H:%M")}! '
        f'Curentul a scăzut de la {raw_injection_T:.2f} MW la {actual_injection_T:.2f} MW · '
        f'Profit PPC: +{actual_profit_T:,.2f} €</div>'
    )
elif has_upcoming_spike and total_alloc_T == 0:
    s_lead = predict_step(first_spike_idx)[0]
    lead_h = first_spike_idx - T
    t_lead_str = _ring_dt(first_spike_idx).strftime("%H:%M")
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
        f'detectat în următoarele ore (la {_ring_dt(peak_idx).strftime("%H:%M")}, peste limita liniei de '
        f'{grid_limit_mw:.0f} MW) · Aprobă dispecerizarea pentru evacuare și profit!</div>'
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
        f"{dt_T.strftime('%H:%M')} · {float(row_T['ambient_temp_celsius']):.1f}°C",
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
            "Injecție Rețea PPC (Real)",
            f"{raw_injection_T:.2f} MW",
            f"Predicție AI: {pred_inj_T:.2f} MW (Δ {pred_inj_T - raw_injection_T:+.2f} MW)",
            delta_color="inverse" if raw_injection_T > grid_limit_mw else "off",
        )
        m3.metric(
            "Surplus Prezis AI",
            f"{surplus_T:.2f} MW",
            f"Real în rețea: {real_surplus_mw(T):.2f} MW · Conf: {conf_T * 100:.0f}%",
            delta_color="inverse" if surplus_T > 2.0 else "off",
        )
        m4.metric(
            "Profit Arbitraj PPC",
            f"+{cumulative_profit:,.2f} €",
            f"+{revenue_T:,.2f} € la livrarea curentă" if revenue_T > 0 else f"PZU {float(row_T['day_ahead_price_eur']):.1f} €/MWh",
            delta_color="normal" if revenue_T > 0 else "off",
        )

    # ---------------- Live ticker chart (Ring Buffer Continuu) ----------------
    start_i = T - HISTORY_HOURS
    end_i = T + LOOKAHEAD_HOURS
    window = list(range(start_i, end_i + 1))
    xs = [_ring_dt(i) for i in window]
    limit_line = [float(_ring_row(i).get("grid_capacity_limit_mw", 10.0)) for i in window]

    def _dispatched_real_inj_for_step(step_idx: int) -> float:
        """Valoarea reală măsurată în rețea (minus dispecerizare dacă e aprobată)."""
        raw = net_injection_mw(step_idx)
        if st.session_state.accepted.get(step_idx) == "ACCEPTED":
            alloc_val = _get_alloc_for_step(step_idx)
            return max(0.0, raw - alloc_val)
        return raw

    def _dispatched_ai_pred_inj_for_step(step_idx: int) -> float:
        """Valoarea prezisă de modelul AI pe setul de date noi (minus dispecerizare dacă e aprobată)."""
        pred_raw = predict_step(step_idx)[3]
        if st.session_state.accepted.get(step_idx) == "ACCEPTED":
            alloc_val = _get_alloc_for_step(step_idx)
            return max(0.0, pred_raw - alloc_val)
        return pred_raw

    # Curba verde: Injecția REALĂ măsurată în rețea până la momentul T
    inj_hist = [_dispatched_real_inj_for_step(i) if i <= T else None for i in window]
    inj_raw_hist = [net_injection_mw(i) if i <= T else None for i in window]

    # Curba albastră punctată: Predicția Modelului AI (atât pe istoric T-24..T pentru comparație live, cât și pe viitor T..T+4)
    ai_pred_full = [_dispatched_ai_pred_inj_for_step(i) for i in window]
    inj_raw_fcst = [predict_step(i)[3] if i >= T else None for i in window]

    t_x = dt_T
    live_value = actual_injection_T
    prev_value = _dispatched_real_inj_for_step(T - 1)
    tick_delta = live_value - prev_value
    tick_color = "#00FF9D" if is_accepted else ("#00FF9D" if tick_delta >= 0 else "#FF4B4B")
    tick_arrow = "▲" if tick_delta >= 0 else "▼"
    if is_accepted:
        status_badge = '<span class="vc-badge-live" style="background:rgba(0,255,157,0.18);color:#00FF9D;border:1px solid rgba(0,255,157,0.5);">● SURPLUS ABSORBIT</span>'
    elif st.session_state.autoplay:
        status_badge = '<span class="vc-badge-live">● LIVE RING BUFFER</span>'
    else:
        status_badge = '<span class="vc-badge-paused">❚❚ PAUSED</span>'

    st.markdown(
        f"""
<div class="vc-ticker">
  <div>
    {status_badge}
    <span class="vc-ticker-meta">
      <span style="color:#00FF9D;">━━ Real Măsurat</span> &nbsp;·&nbsp;
      <span style="color:#38BDF8;">╌╌ Predicție Model AI (Out-of-Sample)</span>
    </span>
  </div>
  <div>
    <span class="vc-ticker-price" style="color:{tick_color};">{live_value:.2f} MW</span>
    <span class="vc-ticker-meta" style="color:#38BDF8;margin-left:8px;">AI: {pred_inj_T:.2f} MW</span>
    <span class="vc-ticker-meta" style="color:{tick_color};margin-left:8px;">{tick_arrow} {abs(tick_delta):.2f} vs T-1h · {t_x.strftime('%d %b %H:%M')}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs, y=inj_hist, mode="lines", name="Injecție Reală Măsurată (≤ T)",
            line=dict(color="#00FF9D", width=2.8, shape="spline", smoothing=0.5),
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
                x=xs, y=inj_raw_fcst, mode="lines", name="Prognoză AI brută (vârf nepreluat)",
                line=dict(color="#F87171", width=1.5, dash="dot"),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=xs, y=ai_pred_full, mode="lines", name="Predicție Model AI (Date Noi)",
            line=dict(color="#38BDF8", width=2.3, dash="dash", shape="spline", smoothing=0.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs, y=limit_line, mode="lines", name="Limită linie PPC (10 MW)",
            line=dict(color="#FF4B4B", width=2.5),
        )
    )
    # Punctul "live" de la capătul curbei reale
    fig.add_trace(
        go.Scatter(
            x=[t_x], y=[live_value], mode="markers", name="Tick curent (Real)",
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
        annotation_text="PROGNOZĂ ML (DATE NOI)", annotation_position="top left",
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
        f_dt = _ring_dt(f_i)
        s_f, _, _, pred_inj_f = predict_step(f_i)
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
                x=f_dt, y=pred_inj_f,
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
  <div class="vc-modal-title">ACORD ARBITRAJ ENERGETIC · {dt_T.strftime('%H:%M')}</div>
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

        # Posibilitate de pre-rezervare și pentru un vârf viitor
        if has_upcoming_spike:
            spike_dt = _ring_dt(first_spike_idx)
            lead_h = first_spike_idx - T
            s_spike, c_spike, _, _ = predict_step(first_spike_idx)
            allocs_spike = allocate(s_spike, c_spike)
            alloc_spike_mw = round(sum(a["allocated_mw"] for a in allocs_spike), 2)
            profit_spike = round(alloc_spike_mw * FIXED_PRICE_EUR, 2)
            is_spike_prebooked = (st.session_state.accepted.get(first_spike_idx) == "ACCEPTED")

            st.markdown('<hr style="border-color:rgba(255,255,255,0.08);margin:12px 0;">', unsafe_allow_html=True)
            st.markdown(f'<div class="vc-sub">🔮 PRE-DISPECERIZARE VIITOR (T+{lead_h}h · {spike_dt.strftime("%H:%M")})</div>', unsafe_allow_html=True)
            if is_spike_prebooked:
                st.success(f"✅ Vârf {spike_dt.strftime('%H:%M')} (+{profit_spike:,.2f} €) deja pre-rezervat.")
            else:
                if st.button(f"⚡ Pre-rezervă și Vârful T+{lead_h}h ({spike_dt.strftime('%H:%M')}) (+{profit_spike:,.2f} €)", key=f"book_next_{first_spike_idx}", width="stretch"):
                    st.session_state.accepted[first_spike_idx] = "ACCEPTED"
                    st.toast(f"✅ Vârf viitor pre-rezervat pentru {spike_dt.strftime('%H:%M')}!")
                    st.rerun()

    # CAZUL 2: La ora curentă T nu e surplus, DAR urmează un spike în lookahead (T+2)
    elif has_upcoming_spike:
        spike_dt = _ring_dt(first_spike_idx)
        lead_h = first_spike_idx - T
        s_spike, c_spike, _, _ = predict_step(first_spike_idx)
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
    Nici la ora {dt_T.strftime('%H:%M')} și nici în următoarele {LOOKAHEAD_HOURS} ore nu se prognozează depășirea limitei liniei PPC de 10.0 MW. Sistemul funcționează în regim nominal standard.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

# =====================================================================
# AUTO-STEP CONTINUU ÎN RING BUFFER (FĂRĂ OPRIRE LA FINALUL DATELOR)
# =====================================================================
if st.session_state.autoplay:
    time.sleep(AUTOPLAY_SECONDS)
    st.session_state.pending_advance = True
    st.rerun()
