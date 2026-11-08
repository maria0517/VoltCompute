import calendar
import ctypes
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agent import VoltComputeAgent
from schemas import GridTelemetry
from services.router import (
    FIXED_TARIFF_EUR_MWH,
    HVAC_MAX_TEMP_C,
    POOL_ALL_DCS_EXHAUSTED,
    load_profiles,
    rank_datacenters,
)
from simulator import VoltComputeSimulator
from train import (
    LIVE_TEST_DATA_PATH,
    MODEL_PATH,
    TRAIN_DATA_PATH,
    ensure_split_datasets,
    train_pipeline,
)

st.set_page_config(
    page_title="VOLTCOMPUTE // HARTA REȚEA SCADA",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Doar dispeceratul PPC logat are acces la hartă (logarea se face din pagina principală)
if st.session_state.get("user", {}).get("role") != "ppc":
    st.warning("Harta rețelei este disponibilă doar pentru dispeceratul PPC. Autentifică-te din pagina principală.")
    st.stop()


FIXED_PRICE_EUR = FIXED_TARIFF_EUR_MWH
LOOKAHEAD_HOURS = 2
HISTORY_HOURS = 24
REFERENCE_TARIFF_EUR = VoltComputeSimulator.STANDARD_RETAIL_PRICE_EUR
AUTOPLAY_SECONDS = 1.0
DATACENTER_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "3_data_centers.csv"
THERMAL_SAFETY_MARGIN_C = 0.5
TEMP_RISE_PER_MW_C = 0.5
DC_PROFILES = load_profiles()
HVAC_LIMIT_C = float(DC_PROFILES.get("hvac_max_temp_c", HVAC_MAX_TEMP_C))
_PROFILE_BY_ID = {n["id"]: n for n in DC_PROFILES["datacenters"]}

# Coordonate geografice reale pentru Parcul Solar PPC și nodurile Data Center din România
PPC_SOLAR_PARK = {
    "id": "PPC_SOLAR_MUNTENIA",
    "name": "Parc Fotovoltaic PPC Renewables (Muntenia Sud)",
    "short": "Parc Solar PPC (18.5 MWp)",
    "lat": 44.235,
    "lon": 27.280,
}

DATACENTERS: List[Dict[str, Any]] = [
    {
        "id": "DC_AI_BUCURESTI",
        "name": "AI Data Center București",
        "short": "AI București",
        "city": "București (UPB / RoEduNet)",
        "lat": 44.4385,
        "lon": 26.0495,
        "queue": 45,
        "color": "#00FF9D",
        "waypoints": [(44.235, 27.280), (44.360, 26.650), (44.4385, 26.0495)],
    },
    {
        "id": "DC_CLOUD_CLUJ",
        "name": "Cloud Data Center Cluj",
        "short": "Cloud Cluj",
        "city": "Cluj-Napoca (UTCN Cloud)",
        "lat": 46.7712,
        "lon": 23.6236,
        "queue": 20,
        "color": "#38BDF8",
        "waypoints": [(44.235, 27.280), (44.950, 26.020), (45.650, 25.580), (46.300, 24.500), (46.7712, 23.6236)],
    },
    {
        "id": "DC_EDGE_TIMISOARA",
        "name": "Edge Data Center Timișoara",
        "short": "Edge Timișoara",
        "city": "Timișoara (UVT HPC Edge)",
        "lat": 45.7473,
        "lon": 21.2314,
        "queue": 30,
        "color": "#A78BFA",
        "waypoints": [(44.235, 27.280), (44.320, 23.800), (44.630, 22.660), (45.200, 21.900), (45.7473, 21.2314)],
    },
]

# Inelul național de înaltă tensiune 400 kV (pentru context vizual SCADA pe hartă)
GRID_400KV_BACKBONE = [
    (44.18, 28.63),  # Constanța / Dobrogea
    (44.33, 28.03),  # Cernavodă
    (44.235, 27.280),  # Parc Solar PPC Călărași
    (44.4385, 26.0495),  # București
    (44.85, 24.87),  # Pitești
    (44.32, 23.80),  # Craiova
    (44.63, 22.66),  # Porțile de Fier
    (45.7473, 21.2314),  # Timișoara
    (46.18, 21.31),  # Arad
    (47.05, 21.92),  # Oradea
    (46.7712, 23.6236),  # Cluj-Napoca
    (46.45, 24.23),  # Iernut / Mureș
    (45.65, 25.58),  # Brașov
    (44.95, 26.02),  # Ploiești
    (44.4385, 26.0495),  # București
]

# =====================================================================
# STYLE (Identic cu tema SCADA din app.py)
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
    font-size: 1.22rem; font-weight: 700; letter-spacing: 0.03em; color: #FFFFFF;
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
.vc-alert-amber {background: rgba(245,158,11,0.10); border: 1px solid rgba(245,158,11,0.5); color: #F59E0B;}
.vc-modal {
    background: #14161e; border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px; padding: 16px; box-shadow: 0 12px 40px rgba(0,0,0,0.55);
    margin-bottom: 10px;
}
.vc-modal-head {
    font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace;
    font-size: 0.76rem; color: #38BDF8; letter-spacing: 0.06em; margin-bottom: 8px;
}
.vc-modal-title {font-size: 1.0rem; font-weight: 700; color: #FFFFFF; margin-bottom: 6px;}
.vc-modal-body {font-size: 0.86rem; color: #D1D5DB; line-height: 1.48;}
.vc-mono {font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace;}
.vc-pane-head {
    font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace;
    font-size: 0.92rem; font-weight: 700; letter-spacing: 0.1em;
    padding: 8px 14px; border-radius: 8px; margin-bottom: 4px;
}
.vc-pane-ppc {color: #00FF9D; background: rgba(0,255,157,0.08); border: 1px solid rgba(0,255,157,0.3);}
.vc-pane-dc {color: #38BDF8; background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.3);}
.vc-pane-sub {font-size: 0.74rem; color: #8E95A5; margin: 0 0 10px 2px;}
.vc-dc-card {
    background: #12141b; border: 1px solid rgba(255,255,255,0.09);
    border-radius: 12px; padding: 12px 14px; margin-bottom: 10px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
}
.vc-bar-bg {
    width: 100%; height: 7px; background: rgba(255,255,255,0.07);
    border-radius: 4px; overflow: hidden; margin: 5px 0 8px 0;
}
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# DATA & SHARED SESSION STATE
# =====================================================================
@st.cache_data(show_spinner=False)
def load_feed() -> Tuple[pd.DataFrame, int, str]:
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


@st.cache_data(show_spinner=False)
def load_datacenter_feed() -> pd.DataFrame:
    datacenter_df = pd.read_csv(DATACENTER_DATA_PATH)
    datacenter_df["dt"] = pd.to_datetime(datacenter_df["timestamp"])
    if "hour" not in datacenter_df.columns:
        datacenter_df["hour"] = datacenter_df["dt"].dt.hour
    return datacenter_df


def ensure_model(agent: VoltComputeAgent) -> str:
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


class _SYSTEMTIME(ctypes.Structure):
    _fields_ = [
        ("wYear", ctypes.c_uint16),
        ("wMonth", ctypes.c_uint16),
        ("wDayOfWeek", ctypes.c_uint16),
        ("wDay", ctypes.c_uint16),
        ("wHour", ctypes.c_uint16),
        ("wMinute", ctypes.c_uint16),
        ("wSecond", ctypes.c_uint16),
        ("wMilliseconds", ctypes.c_uint16),
    ]


@st.cache_data(show_spinner=False)
def get_windows_month_start() -> datetime:
    """Interoghează API-ul nativ Windows (kernel32.GetLocalTime) pentru anul și luna de start."""
    try:
        if hasattr(ctypes, "windll") and hasattr(ctypes.windll, "kernel32"):
            st_time = _SYSTEMTIME()
            ctypes.windll.kernel32.GetLocalTime(ctypes.byref(st_time))
            if 2000 <= int(st_time.wYear) <= 2100 and 1 <= int(st_time.wMonth) <= 12:
                return datetime(int(st_time.wYear), int(st_time.wMonth), 1, 0, 0, 0)
    except Exception:
        pass
    now = datetime.now()
    return datetime(now.year, now.month, 1, 0, 0, 0)


feed, N_TRAIN_ROWS, feed_source = load_feed()
datacenter_feed = load_datacenter_feed()
DATACENTER_LOOKUP = datacenter_feed.set_index(["dt", "dc_id"]).to_dict("index")
DATACENTER_HOUR_LOOKUP = {
    (int(row["hour"]), row["dc_id"]): row.to_dict()
    for _, row in datacenter_feed.iterrows()
}
N_FEED = len(feed)
BASE_LIVE_DT: datetime = get_windows_month_start()


def _month_window(cycle: int) -> Tuple[datetime, int, int]:
    """Returnează (start_dt, hours_in_month, cumulative_offset_hours) pentru luna BASE + cycle."""
    total_m = (BASE_LIVE_DT.year * 12 + (BASE_LIVE_DT.month - 1)) + int(cycle)
    y = total_m // 12
    m = (total_m % 12) + 1
    days = calendar.monthrange(y, m)[1]
    start_dt = datetime(y, m, 1, 0, 0, 0)
    offset_hours = int(round((start_dt - BASE_LIVE_DT).total_seconds() / 3600.0))
    return start_dt, days * 24, offset_hours


if (
    "agent" not in st.session_state
    or type(st.session_state.agent) is not VoltComputeAgent
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
if "dc_history" not in st.session_state:
    st.session_state.dc_history = {
        n["id"]: dict(_PROFILE_BY_ID.get(n["id"], {}).get("history", {"offers_sent": 0, "offers_accepted": 0}))
        for n in DATACENTERS
    }
if "deals" not in st.session_state:
    st.session_state.deals = {}
if "refused" not in st.session_state:
    st.session_state.refused = {}
if "ring_cycle" not in st.session_state:
    st.session_state.ring_cycle = 0

_, _init_month_hours, _ = _month_window(int(st.session_state.ring_cycle))

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
    st.session_state.current_step = int(default_step % _init_month_hours)

_toast_msg = st.session_state.pop("toast_msg", None)
if _toast_msg:
    st.toast(_toast_msg)

if st.session_state.pending_advance:
    st.session_state.pending_advance = False
    _, _cur_m_hours, _ = _month_window(int(st.session_state.ring_cycle))
    nxt_step = int(st.session_state.current_step) + 1
    if nxt_step >= _cur_m_hours:
        st.session_state.current_step = 0
        st.session_state.ring_cycle = int(st.session_state.ring_cycle) + 1
    else:
        st.session_state.current_step = nxt_step

CYCLE_START_DT, N_ROWS, CYCLE_OFFSET_HOURS = _month_window(int(st.session_state.ring_cycle))
if int(st.session_state.current_step) >= N_ROWS:
    st.session_state.current_step = int(st.session_state.current_step) % N_ROWS

agent: VoltComputeAgent = st.session_state.agent


def _ring_row(v_idx: int) -> pd.Series:
    return feed.iloc[v_idx % N_FEED]


def _ring_dt(v_idx: int) -> datetime:
    return BASE_LIVE_DT + timedelta(hours=int(v_idx))


def _cycle_variation(v_idx: int) -> Tuple[float, float, float]:
    v_cycle = v_idx // N_FEED
    if v_cycle == 0:
        return 1.0, 0.0, 0.0
    rng = np.random.default_rng(seed=100_003 + int(v_idx) * 31)
    irr_factor = float(np.clip(1.0 + rng.normal(0.0, 0.045), 0.88, 1.12))
    temp_offset = float(rng.normal(0.0, 0.6))
    inj_noise = float(rng.normal(0.0, 0.28))
    return irr_factor, temp_offset, inj_noise


def net_injection_mw(v_idx: int) -> float:
    row = _ring_row(v_idx)
    irr_factor, _, inj_noise = _cycle_variation(v_idx)
    base_real_inj = float(row["net_injection_mw"])
    if int(row["hour"]) < 6 or int(row["hour"]) > 19 or base_real_inj <= 0.01:
        return 0.0
    return round(max(0.0, base_real_inj * irr_factor + inj_noise), 3)


def real_surplus_mw(v_idx: int) -> float:
    row = _ring_row(v_idx)
    lim = float(row.get("grid_capacity_limit_mw", 10.0))
    return round(max(0.0, net_injection_mw(v_idx) - lim), 3)


def predict_step(v_idx: int) -> Tuple[float, float, Dict[str, float], float]:
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


DC_BY_ID: Dict[str, Dict[str, Any]] = {n["id"]: n for n in DATACENTERS}
ELIGIBLE_NODES: List[Dict[str, Any]] = sorted(
    DATACENTERS,
    key=lambda n: -n["queue"],
)


def grid_telemetry(v_idx: int) -> GridTelemetry:
    row = _ring_row(v_idx)
    irr_factor, temp_offset, _ = _cycle_variation(v_idx)
    dt_val = _ring_dt(v_idx)
    irr_val = float(np.clip(float(row["solar_irradiance_wm2"]) * irr_factor, 0.0, 1400.0))
    temp_val = float(np.clip(float(row["ambient_temp_celsius"]) + temp_offset, -20.0, 55.0))
    return GridTelemetry(
        timestamp=dt_val.isoformat(),
        hour=int(row["hour"]),
        solar_irradiance_wm2=round(irr_val, 2),
        ambient_temp_celsius=round(temp_val, 2),
        day_ahead_price_eur=float(row["day_ahead_price_eur"]),
        grid_capacity_limit_mw=float(row.get("grid_capacity_limit_mw", 10.0)),
        is_weekend=int(dt_val.weekday() >= 5),
    )


def step_anomalies(v_idx: int) -> List[str]:
    return agent.detect_anomalies(grid_telemetry(v_idx))


def step_price(v_idx: int) -> float:
    return float(_ring_row(v_idx)["day_ahead_price_eur"])


def _surplus_trigger(step: int) -> bool:
    return predict_step(step)[0] > agent.MIN_SURPLUS_THRESHOLD_MW


def step_qualifies(step: int) -> bool:
    return (
        _surplus_trigger(step)
        and not step_anomalies(step)
        and predict_step(step)[1] >= agent.CONFIDENCE_THRESHOLD
    )


def step_escalated(step: int) -> bool:
    return (
        _surplus_trigger(step)
        and not step_anomalies(step)
        and predict_step(step)[1] < agent.CONFIDENCE_THRESHOLD
    )


def delivered_mw(step: int) -> float:
    return round(sum(st.session_state.deals.get(step, {}).values()), 3)


def remaining_surplus(step: int) -> float:
    return max(0.0, predict_step(step)[0] - delivered_mw(step))


def node_open_for_step(step: int, node: Dict[str, Any]) -> bool:
    return (
        node["id"] not in st.session_state.deals.get(step, {})
        and node["id"] not in st.session_state.refused.get(step, set())
    )


def datacenter_snapshot(step: int, node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    feed_row = _ring_row(step)
    timestamp = pd.Timestamp(feed_row["dt"])
    snapshot = DATACENTER_LOOKUP.get((timestamp, node["id"]))
    if snapshot is None:
        snapshot = DATACENTER_HOUR_LOOKUP.get((int(feed_row["hour"]), node["id"]))
    return dict(snapshot) if snapshot is not None else None


def delivered_to_node(step: int, node: Dict[str, Any]) -> float:
    return float(st.session_state.deals.get(step, {}).get(node["id"], 0.0))


def delivered_temperature(step: int, node: Dict[str, Any]) -> Optional[float]:
    snapshot = datacenter_snapshot(step, node)
    if snapshot is None:
        return None
    return float(snapshot["temp_servere_c"]) + delivered_to_node(step, node) * TEMP_RISE_PER_MW_C


def node_thermal_limit(snapshot: Dict[str, Any]) -> float:
    return min(float(snapshot["limita_termica_c"]), HVAC_LIMIT_C)


def thermal_rejected(step: int, node: Dict[str, Any]) -> bool:
    snapshot = datacenter_snapshot(step, node)
    if snapshot is None:
        return False
    return float(delivered_temperature(step, node)) > node_thermal_limit(snapshot)


def node_free_capacity(step: int, node: Dict[str, Any]) -> float:
    snapshot = datacenter_snapshot(step, node)
    if snapshot is None:
        return 0.0
    temp = float(delivered_temperature(step, node))
    free_capacity = max(0.0, float(snapshot["capacitate_libera_mw"]) - delivered_to_node(step, node))
    thermal_capacity = max(0.0, (node_thermal_limit(snapshot) - temp) / TEMP_RISE_PER_MW_C)
    return min(free_capacity, thermal_capacity)


def offer_capacity(step: int, node: Dict[str, Any], remaining_mw: float) -> float:
    if thermal_rejected(step, node):
        return 0.0
    return max(0.0, min(float(remaining_mw), node_free_capacity(step, node)))


def node_ranking(step: int) -> List[Dict[str, Any]]:
    basis_mw = remaining_surplus(step) if remaining_surplus(step) > 0.05 else predict_step(step)[0]
    candidates = []
    for node in DATACENTERS:
        snapshot = datacenter_snapshot(step, node)
        if snapshot is None:
            continue
        hist = st.session_state.dc_history[node["id"]]
        candidates.append(
            {
                "id": node["id"],
                "free_capacity_mw": node_free_capacity(step, node),
                "temp_c": float(delivered_temperature(step, node)),
                "thermal_limit_c": node_thermal_limit(snapshot),
                "offers_sent": hist["offers_sent"],
                "offers_accepted": hist["offers_accepted"],
            }
        )
    return rank_datacenters(basis_mw, candidates, hvac_max_temp_c=HVAC_LIMIT_C)


def build_offer(steps: List[int]) -> Tuple[Optional[Dict[str, Any]], bool]:
    pending_steps = [s for s in steps if step_qualifies(s) and remaining_surplus(s) > 0.05]
    if not pending_steps:
        return None, False
    for ranked in node_ranking(pending_steps[0]):
        if not ranked["eligible"]:
            continue
        node = DC_BY_ID[ranked["id"]]
        per_step: Dict[int, float] = {}
        for s in pending_steps:
            if node_open_for_step(s, node):
                mw = round(offer_capacity(s, node, remaining_surplus(s)), 2)
                if mw >= 0.05:
                    per_step[s] = mw
        if per_step:
            return {
                "node": node,
                "per_step": per_step,
                "total_mwh": round(sum(per_step.values()), 2),
                "match_score": ranked["match_score"],
            }, False
    return None, True


def _accept_offer(node_id: str, node_label: str, per_step: Dict[int, float]) -> None:
    for step, mw in per_step.items():
        st.session_state.deals.setdefault(step, {})[node_id] = mw
    st.session_state.pending_advance = False
    hist = st.session_state.dc_history[node_id]
    hist["offers_sent"] += 1
    hist["offers_accepted"] += 1
    total = sum(per_step.values())
    st.session_state.toast_msg = (
        f"✅ {node_label} a acceptat oferta: {total:.2f} MWh la {FIXED_PRICE_EUR:.0f} €/MWh."
    )


def _refuse_offer(node_id: str, node_label: str, per_step: Dict[int, float]) -> None:
    for step in per_step:
        st.session_state.refused.setdefault(step, set()).add(node_id)
    st.session_state.dc_history[node_id]["offers_sent"] += 1
    st.session_state.pending_advance = False
    st.session_state.toast_msg = f"⛔ {node_label} a refuzat oferta. Se redirecționează fluxul către următorul nod."


def _reset_offers(steps: List[int]) -> None:
    for step in steps:
        for node_id in st.session_state.deals.pop(step, {}):
            hist = st.session_state.dc_history[node_id]
            hist["offers_sent"] = max(0, hist["offers_sent"] - 1)
            hist["offers_accepted"] = max(0, hist["offers_accepted"] - 1)
        for node_id in st.session_state.refused.pop(step, set()):
            hist = st.session_state.dc_history[node_id]
            hist["offers_sent"] = max(0, hist["offers_sent"] - 1)
    st.session_state.pending_advance = False
    st.session_state.toast_msg = "↩️ Ofertele pentru interval au fost resetate."


def _step_back() -> None:
    st.session_state.autoplay = False
    prv = int(st.session_state.current_step) - 1
    if prv < 0:
        prev_cycle = int(st.session_state.ring_cycle) - 1
        _, prev_month_hours, _ = _month_window(prev_cycle)
        st.session_state.ring_cycle = prev_cycle
        st.session_state.current_step = prev_month_hours - 1
    else:
        st.session_state.current_step = prv


def _step_forward() -> None:
    st.session_state.autoplay = False
    _, cur_month_hours, _ = _month_window(int(st.session_state.ring_cycle))
    nxt = int(st.session_state.current_step) + 1
    if nxt >= cur_month_hours:
        st.session_state.current_step = 0
        st.session_state.ring_cycle = int(st.session_state.ring_cycle) + 1
    else:
        st.session_state.current_step = nxt


def _toggle_play() -> None:
    st.session_state.autoplay = not st.session_state.autoplay


# =====================================================================
# HEADER + NAVIGARE ÎNAPOI LA CONSOLA PRINCIPALĂ
# =====================================================================
top_nav, top_title = st.columns([1.4, 5.6], gap="small")
with top_nav:
    try:
        st.page_link("app.py", label="⬅️ Înapoi la Consola Principală", width="stretch")
    except Exception:
        st.markdown('<div class="vc-sub" style="padding-top:8px;">🗺️ MOD STANDALONE HARTĂ</div>', unsafe_allow_html=True)
with top_title:
    st.markdown(
        f"""
<div style="display:flex;justify-content:space-between;align-items:center;padding-top:4px;">
  <div class="vc-title"><span class="vc-live"></span>VOLTCOMPUTE // SCADA NATIONAL POWER FLOW MAP</div>
  <div class="vc-sub">LUNA CURENTĂ: {CYCLE_START_DT.strftime('%Y-%m')} (Windows API) · {N_ROWS}h în lună (Luna #{int(st.session_state.ring_cycle) + 1}) · TOPOLOGIE 400 kV</div>
</div>
""",
        unsafe_allow_html=True,
    )

current_cycle = int(st.session_state.ring_cycle)
slider_key = f"timeline_slider_map_{current_cycle}"
st.session_state[slider_key] = int(st.session_state.current_step)


def _on_slider_change(s_key: str) -> None:
    st.session_state.autoplay = False
    st.session_state.current_step = int(st.session_state[s_key])


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
        key=slider_key,
        on_change=_on_slider_change,
        args=(slider_key,),
        format_func=lambda i, off=CYCLE_OFFSET_HOURS: _ring_dt(off + int(i)).strftime("%Y-%m-%d %H:%M"),
        label_visibility="collapsed",
    )

# =====================================================================
# STATE CALCULATION AT STEP T
# =====================================================================
T = CYCLE_OFFSET_HOURS + int(st.session_state.current_step)
row_T = _ring_row(T)
dt_T = _ring_dt(T)
surplus_T, conf_T, shap_T, pred_inj_T = predict_step(T)

delivered_T = delivered_mw(T)
revenue_T = round(delivered_T * FIXED_PRICE_EUR, 2)
is_dispatched = delivered_T > 0.01

raw_injection_T = net_injection_mw(T)
actual_injection_T = max(0.0, raw_injection_T - delivered_T)
grid_limit_mw = float(row_T.get("grid_capacity_limit_mw", 10.0))

future_idx = [T + k for k in range(1, LOOKAHEAD_HOURS + 1)]
upcoming_spikes = [i for i in future_idx if step_qualifies(i)]
has_upcoming_spike = len(upcoming_spikes) > 0
first_spike_idx = upcoming_spikes[0] if has_upcoming_spike else None

if step_qualifies(T):
    offer_steps = [T]
elif has_upcoming_spike:
    offer_steps = upcoming_spikes
else:
    offer_steps = []

offer, offers_exhausted = build_offer(offer_steps)
table_steps = offer_steps if offer_steps else [T]


def _node_status(node: Dict[str, Any]) -> str:
    snapshots = [datacenter_snapshot(s, node) for s in table_steps]
    if any(snapshot is None for snapshot in snapshots):
        return "NO_TELEMETRY"
    if any(node["id"] in st.session_state.deals.get(s, {}) for s in table_steps):
        if any(
            delivered_temperature(s, node) > float(snapshot["limita_termica_c"])
            for s, snapshot in zip(table_steps, snapshots)
        ):
            return "REJECTED_THERMAL_SAFETY"
        return "ACCEPTED"
    if any(thermal_rejected(s, node) for s in table_steps):
        return "REJECTED_THERMAL_SAFETY"
    if any(node["id"] in st.session_state.refused.get(s, set()) for s in table_steps):
        return "REFUSED"
    if offer is not None and offer["node"]["id"] == node["id"]:
        return "OFFER_PENDING"
    if any(step_escalated(s) for s in table_steps):
        return "ESCALATE_TO_HUMAN"
    if any(step_qualifies(s) and remaining_surplus(s) > 0.05 for s in table_steps):
        return "IN_QUEUE"
    return "STANDBY"


def _interpolate_path(waypoints: List[Tuple[float, float]], n_points: int = 18) -> Tuple[List[float], List[float]]:
    """Generează puncte intermediare pe traseul liniei de transport pentru afișarea fluxului de putere."""
    lats = [p[0] for p in waypoints]
    lons = [p[1] for p in waypoints]
    t_orig = np.linspace(0.0, 1.0, len(waypoints))
    t_fine = np.linspace(0.0, 1.0, n_points)
    return list(np.interp(t_fine, t_orig, lats)), list(np.interp(t_fine, t_orig, lons))


# =====================================================================
# LAYOUT: HARTĂ REȚEA NAȚIONALĂ (STÂNGA) | CONTROL OFERTE & TELEMETRIE SALĂ SERVERE (DREAPTA)
# =====================================================================
col_map, col_dc = st.columns([0.68, 0.32], gap="large")

with col_map:
    st.markdown(
        '<div class="vc-pane-head vc-pane-ppc">HARTA NAȚIONALĂ SCADA // FLUX ENERGETIC PPC → DATA CENTRES</div>'
        '<div class="vc-pane-sub">Vizualizare geospațială în timp real a surplusului solar și a rutării către clusterele de calcul</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ora Sistem SCADA", dt_T.strftime("%H:%M"), f"Iradiere: {float(row_T['solar_irradiance_wm2']):.0f} W/m²")
    m2.metric("Injecție Parc PPC", f"{actual_injection_T:.2f} MW", f"Brut: {raw_injection_T:.2f} MW (lim. {grid_limit_mw:.0f} MW)")
    m3.metric("Surplus Prezis AI", f"{surplus_T:.2f} MW", f"Confidență ML: {conf_T * 100:.0f}%")
    total_contracted_now = sum(
        sum(st.session_state.deals.get(s, {}).get(n["id"], 0.0) for s in table_steps)
        for n in DATACENTERS
    )
    m4.metric(
        "Transfer Activ către DC",
        f"{total_contracted_now:.2f} MWh",
        f"Profit PPC: +{total_contracted_now * FIXED_PRICE_EUR:,.2f} €",
    )

    # ---------------- Construcție Hartă SCADA Plotly ----------------
    map_fig = go.Figure()

    # 1. Inelul Energetic Național 400 kV (fundal rețea)
    map_fig.add_trace(
        go.Scattermap(
            lat=[p[0] for p in GRID_400KV_BACKBONE],
            lon=[p[1] for p in GRID_400KV_BACKBONE],
            mode="lines",
            line=dict(width=1.5, color="rgba(142, 149, 165, 0.28)"),
            name="Inel 400 kV Transelectrica",
            hoverinfo="skip",
        )
    )

    # 2. Coridoarele de transport PPC Solar Park -> fiecare Data Center
    for node in DATACENTERS:
        status = _node_status(node)
        contracted_mwh = round(
            sum(st.session_state.deals.get(s, {}).get(node["id"], 0.0) for s in table_steps), 2
        )
        pending_mwh = offer["total_mwh"] if (offer is not None and offer["node"]["id"] == node["id"]) else 0.0
        path_lats, path_lons = _interpolate_path(node["waypoints"], n_points=22)

        if status == "ACCEPTED":
            line_color = node["color"]
            line_width = 5.0
            flow_label = f"⚡ {contracted_mwh:.2f} MWh ACTIV"
        elif status == "OFFER_PENDING":
            line_color = "#F59E0B"
            line_width = 3.8
            flow_label = f"🔔 OFERTĂ: {pending_mwh:.2f} MWh"
        elif status == "REJECTED_THERMAL_SAFETY":
            line_color = "#FF4B4B"
            line_width = 2.2
            flow_label = "🔥 BLOCAT TERMIC"
        elif status == "REFUSED":
            line_color = "rgba(248, 113, 113, 0.45)"
            line_width = 2.0
            flow_label = "⛔ REFUZAT → RERUTAT"
        else:
            line_color = "rgba(56, 189, 248, 0.30)"
            line_width = 2.0
            flow_label = "STANDBY"

        # Linia de înaltă tensiune către nod
        map_fig.add_trace(
            go.Scattermap(
                lat=path_lats,
                lon=path_lons,
                mode="lines",
                line=dict(width=line_width, color=line_color),
                name=f"Flux → {node['short']} ({flow_label})",
                hoverinfo="text",
                hovertext=f"Coridor PPC → {node['name']}: {flow_label}",
            )
        )

        # Particule luminoase pe traseu când există transfer aprobat sau ofertă activă
        if status in ("ACCEPTED", "OFFER_PENDING"):
            particle_idx = list(range(2, len(path_lats) - 2, 3))
            map_fig.add_trace(
                go.Scattermap(
                    lat=[path_lats[i] for i in particle_idx],
                    lon=[path_lons[i] for i in particle_idx],
                    mode="markers",
                    marker=dict(
                        size=9 if status == "ACCEPTED" else 7,
                        color="#FFFFFF" if status == "ACCEPTED" else "#F59E0B",
                        opacity=0.9,
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # 3. Nodul Sursă: Parcul Fotovoltaic PPC
    park_Execution_color = "#00FF9D" if (is_dispatched or surplus_T <= 2.0) else "#FF4B4B"
    park_halo_size = 38 + min(26, int(raw_injection_T * 1.4))
    map_fig.add_trace(
        go.Scattermap(
            lat=[PPC_SOLAR_PARK["lat"]],
            lon=[PPC_SOLAR_PARK["lon"]],
            mode="markers",
            marker=dict(size=park_halo_size, color=park_Execution_color, opacity=0.22),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    map_fig.add_trace(
        go.Scattermap(
            lat=[PPC_SOLAR_PARK["lat"]],
            lon=[PPC_SOLAR_PARK["lon"]],
            mode="markers+text",
            marker=dict(size=20, color=park_Execution_color),
            text=[f"⚡ PARC SOLAR PPC ({actual_injection_T:.1f} MW)"],
            textposition="bottom center",
            textfont=dict(family="JetBrains Mono, monospace", size=12, color="#FFFFFF"),
            name="Parc Solar PPC",
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Injecție curentă: %{customdata[1]:.2f} MW<br>"
                "Surplus prezis AI: %{customdata[2]:.2f} MW<br>"
                "Iradianță: %{customdata[3]:.0f} W/m²<extra></extra>"
            ),
            customdata=[[
                PPC_SOLAR_PARK["name"],
                actual_injection_T,
                surplus_T,
                float(row_T["solar_irradiance_wm2"]),
            ]],
        )
    )

    # 4. Nodurile Data Center (București, Cluj, Timișoara)
    for node in DATACENTERS:
        status = _node_status(node)
        snap = datacenter_snapshot(table_steps[0], node)
        cur_temp = delivered_temperature(table_steps[0], node) or 22.0
        lim_temp = float(snap["limita_termica_c"]) if snap else 26.0
        free_cap = float(snap["capacitate_libera_mw"]) if snap else 0.0
        contracted_mwh = round(
            sum(st.session_state.deals.get(s, {}).get(node["id"], 0.0) for s in table_steps), 2
        )

        if status == "ACCEPTED":
            node_color = "#00FF9D"
            badge_txt = f"✅ {node['short']} (+{contracted_mwh:.1f} MWh)"
        elif status == "OFFER_PENDING":
            node_color = "#F59E0B"
            badge_txt = f"🔔 {node['short']} (OFERTĂ: {offer['total_mwh']:.1f} MWh)"
        elif status == "REJECTED_THERMAL_SAFETY":
            node_color = "#FF4B4B"
            badge_txt = f"🔥 {node['short']} (TERMIC {cur_temp:.1f}°C)"
        elif status == "REFUSED":
            node_color = "#F87171"
            badge_txt = f"⛔ {node['short']} (REFUZAT)"
        else:
            node_color = node["color"]
            badge_txt = f"🖥️ {node['short']} ({cur_temp:.1f}°C · {free_cap:.1f} MW liberi)"

        # Halou exterior pentru nodul Data Center
        map_fig.add_trace(
            go.Scattermap(
                lat=[node["lat"]],
                lon=[node["lon"]],
                mode="markers",
                marker=dict(
                    size=36 if status in ("ACCEPTED", "OFFER_PENDING", "REJECTED_THERMAL_SAFETY") else 26,
                    color=node_color,
                    opacity=0.25,
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        map_fig.add_trace(
            go.Scattermap(
                lat=[node["lat"]],
                lon=[node["lon"]],
                mode="markers+text",
                marker=dict(size=16, color=node_color),
                text=[badge_txt],
                textposition="top center",
                textfont=dict(family="JetBrains Mono, monospace", size=12, color="#FFFFFF"),
                name=node["name"],
                hovertemplate=(
                    f"<b>{node['name']}</b> ({node['city']})<br>"
                    f"Status Agent: <b>{status}</b><br>"
                    f"Temperatură Servere: {cur_temp:.1f}°C / Limită {lim_temp:.1f}°C<br>"
                    f"Capacitate Liberă: {free_cap:.2f} MW<br>"
                    f"MWh Contractați: {contracted_mwh:.2f} MWh<extra></extra>"
                ),
            )
        )

    map_fig.update_layout(
        map=dict(
            style="carto-darkmatter",
            center=dict(lat=45.45, lon=24.65),
            zoom=5.75,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=610,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(12, 13, 18, 0.82)",
            bordercolor="rgba(255,255,255,0.12)",
            borderwidth=1,
            font=dict(family="JetBrains Mono, monospace", size=10, color="#E5E7EB"),
        ),
        uirevision="voltcompute-scada-map",
    )
    st.plotly_chart(map_fig, width="stretch")

# =====================================================================
# DREAPTA: OFERTĂ ACTIVĂ (INTERACTIVĂ) + TELEMETRIE SALĂ SERVERE PE NODURI
# =====================================================================
with col_dc:
    st.markdown(
        '<div class="vc-pane-head vc-pane-dc">CONTROL RUTARE & SALĂ SERVERE</div>'
        '<div class="vc-pane-sub">Acceptă sau refuză oferta ca să vezi redirecționarea live pe hartă</div>',
        unsafe_allow_html=True,
    )

    savings_per_mwh = REFERENCE_TARIFF_EUR - FIXED_PRICE_EUR

    if offer is not None:
        node = offer["node"]
        per_step = offer["per_step"]
        steps_sorted = sorted(per_step)
        n_hours = len(steps_sorted)
        total_mwh = offer["total_mwh"]
        avg_mw = total_mwh / n_hours
        first_step = steps_sorted[0]
        slot_start = _ring_dt(first_step).strftime("%H:%M")
        slot_end = _ring_dt(steps_sorted[-1] + 1).strftime("%H:%M")
        lead = first_step - T
        when_txt = (
            f"acum ({slot_start} – {slot_end})"
            if lead <= 0
            else f"în T+{lead}h ({slot_start} – {slot_end})"
        )
        client_savings = total_mwh * savings_per_mwh
        queue_after = [n["short"] for n in ELIGIBLE_NODES if n["id"] != node["id"] and node_open_for_step(first_step, n)]
        queue_txt = " → ".join(queue_after) if queue_after else "niciun alt data center"

        st.markdown(
            f"""
<div class="vc-modal">
  <div class="vc-modal-head">🔔 RUTARE ACTIVĂ PE HARTĂ → {node['short'].upper()}</div>
  <div class="vc-modal-title">{node['name']}</div>
  <div class="vc-modal-body">
    Flux propus: <span class="vc-mono" style="color:#00FF9D;">{avg_mw:.2f} MW</span> ({total_mwh:.2f} MWh) {when_txt}
    la <span class="vc-mono" style="color:#00FF9D;">{FIXED_PRICE_EUR:.0f} €/MWh</span>.
    <br>
    💰 <b>Economie centru:</b> <span class="vc-mono" style="color:#00FF9D;">+{client_savings:,.2f} €</span>
    <br>
    <span style="color:#8E95A5;font-size:0.78rem;">La refuz, linia de pe hartă comută automat către: {queue_txt}</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        b_acc, b_ref = st.columns(2)
        b_acc.button(
            "✅ Acceptă Fluxul",
            key=f"map_accept_{node['id']}_{first_step}",
            type="primary",
            width="stretch",
            on_click=_accept_offer,
            args=(node["id"], node["short"], dict(per_step)),
        )
        b_ref.button(
            "⛔ Refuză (Rerutează)",
            key=f"map_refuse_{node['id']}_{first_step}",
            width="stretch",
            on_click=_refuse_offer,
            args=(node["id"], node["short"], dict(per_step)),
        )
    elif offer_steps:
        st.button(
            "↩️ Resetează ofertele pe acest interval",
            key=f"map_reset_{offer_steps[0]}",
            width="stretch",
            on_click=_reset_offers,
            args=(list(offer_steps),),
        )

    st.markdown(
        '<div class="vc-sub" style="margin:10px 0 6px 0;">TELEMETRIE LIVE CHILLER & CAPACITATE PE FIECARE NOD</div>',
        unsafe_allow_html=True,
    )

    for node in DATACENTERS:
        status = _node_status(node)
        snap = datacenter_snapshot(table_steps[0], node)
        cur_temp = delivered_temperature(table_steps[0], node) or 22.0
        lim_temp = float(snap["limita_termica_c"]) if snap else 26.0
        free_cap = float(snap["capacitate_libera_mw"]) if snap else 0.0
        contracted = round(
            sum(st.session_state.deals.get(s, {}).get(node["id"], 0.0) for s in table_steps), 2
        )
        temp_pct = int(np.clip((cur_temp / max(lim_temp, 1.0)) * 100.0, 0, 100))
        temp_bar_color = "#FF4B4B" if cur_temp >= lim_temp - THERMAL_SAFETY_MARGIN_C else ("#F59E0B" if cur_temp >= lim_temp - 2.0 else "#00FF9D")

        status_color = {
            "ACCEPTED": "#00FF9D",
            "OFFER_PENDING": "#F59E0B",
            "REJECTED_THERMAL_SAFETY": "#FF4B4B",
            "REFUSED": "#F87171",
            "IN_QUEUE": "#38BDF8",
        }.get(status, "#8E95A5")

        st.markdown(
            f"""
<div class="vc-dc-card" style="border-left: 3px solid {status_color};">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span style="font-weight:700;color:#FFFFFF;font-size:0.88rem;">{node['name']}</span>
    <span style="color:{status_color};font-size:0.74rem;font-weight:700;">● {status}</span>
  </div>
  <div style="color:#8E95A5;font-size:0.74rem;margin-top:2px;">{node['city']} · Coadă: {node['queue']} joburi AI</div>
  <div style="display:flex;justify-content:space-between;font-size:0.76rem;margin-top:8px;">
    <span>🌡️ Temp Servere: <b style="color:{temp_bar_color};">{cur_temp:.1f}°C</b> / {lim_temp:.1f}°C</span>
    <span>⚡ Liber: <b>{max(0.0, free_cap - contracted):.2f} MW</b></span>
  </div>
  <div class="vc-bar-bg">
    <div style="width:{temp_pct}%;height:100%;background:{temp_bar_color};"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#D1D5DB;">
    <span>MWh Contractați: <b style="color:#00FF9D;">{contracted:.2f} MWh</b></span>
    <span>Economie: <b style="color:#38BDF8;">+{contracted * savings_per_mwh:,.0f} €</b></span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

if st.session_state.autoplay:
    time.sleep(AUTOPLAY_SECONDS)
    st.session_state.pending_advance = True
    st.rerun()
