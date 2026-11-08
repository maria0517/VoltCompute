import calendar
import ctypes
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    page_title="VOLTCOMPUTE // PPC GRID DISPATCH",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

FIXED_PRICE_EUR = FIXED_TARIFF_EUR_MWH  # 30 EUR/MWh — NU scade la refuz
LOOKAHEAD_HOURS = 2
HISTORY_HOURS = 24
AUTOPLAY_SECONDS = 1.0
REFERENCE_TARIFF_EUR = VoltComputeSimulator.STANDARD_RETAIL_PRICE_EUR
DATACENTER_DATA_PATH = Path(__file__).parent / "data" / "3_data_centers.csv"
TEMP_RISE_PER_MW_C = 0.5

# Profilurile + istoricul ofertelor (input pentru Match Score) vin din data/dc_profiles_and_history.json
DC_PROFILES = load_profiles()
DATACENTERS: List[Dict[str, Any]] = DC_PROFILES["datacenters"]
DC_BY_ID: Dict[str, Dict[str, Any]] = {n["id"]: n for n in DATACENTERS}
HVAC_LIMIT_C = float(DC_PROFILES.get("hvac_max_temp_c", HVAC_MAX_TEMP_C))

# =====================================================================
# STYLE
# =====================================================================
st.markdown(
    """
<style>
#MainMenu, footer, header {visibility: hidden;}
/* Anti-pâlpâire: Streamlit estompează elementele la fiecare rerulare; le ținem stabile */
[data-stale="true"], .stale-element, div[data-testid="stElementContainer"][data-stale="true"] {
    opacity: 1 !important;
    transition: none !important;
}
.stButton button {transition: none !important;}
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
.vc-alert-amber {background: rgba(245,158,11,0.10); border: 1px solid rgba(245,158,11,0.5); color: #F59E0B;}
.vc-modal {
    background: #14161e; border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px; padding: 18px; box-shadow: 0 12px 40px rgba(0,0,0,0.55);
    margin-bottom: 10px;
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
/* Separare vizuală PPC (stânga) | Data Centre (dreapta) */
div[data-testid="stHorizontalBlock"]:has(#vc-split-marker) {align-items: stretch !important;}
div[data-testid="stColumn"]:has(#vc-dc-marker), div[data-testid="column"]:has(#vc-dc-marker) {
    border-left: 2px solid rgba(56,189,248,0.35);
    padding-left: 1.6rem;
}
.vc-pane-head {
    font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace;
    font-size: 0.95rem; font-weight: 700; letter-spacing: 0.1em;
    padding: 8px 14px; border-radius: 8px; margin-bottom: 4px;
}
.vc-pane-ppc {color: #00FF9D; background: rgba(0,255,157,0.08); border: 1px solid rgba(0,255,157,0.3);}
.vc-pane-dc {color: #38BDF8; background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.3);}
.vc-pane-sub {font-size: 0.74rem; color: #8E95A5; margin: 0 0 10px 2px;}
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


@st.cache_data(show_spinner=False)
def load_datacenter_feed() -> pd.DataFrame:
    """Încarcă telemetria orară reală pentru capacitatea și temperatura fiecărui nod."""
    datacenter_df = pd.read_csv(DATACENTER_DATA_PATH)
    datacenter_df["dt"] = pd.to_datetime(datacenter_df["timestamp"])
    if "hour" not in datacenter_df.columns:
        datacenter_df["hour"] = datacenter_df["dt"].dt.hour
    return datacenter_df


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
    """
    Preia anul și luna de start printr-un apel API nativ către Windows (`kernel32.dll -> GetLocalTime`).
    Include fallback pe ceasul sistemului dacă rulează în Linux/Docker.
    """
    try:
        if hasattr(ctypes, "windll") and hasattr(ctypes.windll, "kernel32"):
            sys_time = _SYSTEMTIME()
            ctypes.windll.kernel32.GetLocalTime(ctypes.byref(sys_time))
            return datetime(int(sys_time.wYear), int(sys_time.wMonth), 1, 0, 0, 0)
    except Exception:
        pass
    now = datetime.now()
    return datetime(now.year, now.month, 1, 0, 0, 0)


feed, N_TRAIN_ROWS, feed_source = load_feed()
N_FEED = len(feed)
datacenter_feed = load_datacenter_feed()
DATACENTER_LOOKUP = datacenter_feed.set_index(["dt", "dc_id"]).to_dict("index")
DATACENTER_HOUR_LOOKUP = {
    (int(row["hour"]), row["dc_id"]): row.to_dict()
    for _, row in datacenter_feed.iterrows()
}
BASE_LIVE_DT: datetime = get_windows_month_start()


def _month_window(cycle: int) -> Tuple[datetime, int, int]:
    """
    Returnează (data_start_lună, număr_ore_în_lună, offset_ore_față_de_BASE_LIVE_DT)
    pentru luna corespunzătoare ciclului `cycle` (0 = luna din Windows API, 1 = luna următoare etc.).
    """
    total_month_idx = (BASE_LIVE_DT.month - 1) + int(cycle)
    year = BASE_LIVE_DT.year + (total_month_idx // 12)
    month = (total_month_idx % 12) + 1
    days_in_month = calendar.monthrange(year, month)[1]
    month_start = datetime(year, month, 1, 0, 0, 0)
    offset_hours = int(round((month_start - BASE_LIVE_DT).total_seconds() / 3600.0))
    return month_start, days_in_month * 24, offset_hours


# Scală verticală fixă pentru grafic, ca axa să nu "sară" la fiecare tick (ca pe un terminal de bursă)
Y_AXIS_MAX = float(feed["gross_solar_mw"].max()) * 1.08 + 0.5

if (
    "agent" not in st.session_state
    # După o modificare de cod, Streamlit reîncarcă modulul `agent`, dar sesiunea păstrează
    # obiectul creat din clasa veche -> îl recreăm, altfel lipsesc atributele/metodele noi.
    or type(st.session_state.agent) is not VoltComputeAgent
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
    # Istoricul ofertelor per nod; actualizat la fiecare Aprobă / Refuză -> clasamentul se adaptează
    st.session_state.dc_history = {
        n["id"]: dict(n.get("history", {"offers_sent": 0, "offers_accepted": 0})) for n in DATACENTERS
    }
if "deals" not in st.session_state:
    # deals[pas][id_datacenter] = MW contractați la pasul respectiv
    st.session_state.deals = {}
if "refused" not in st.session_state:
    # refused[pas] = set cu id-urile datacenter-elor care au refuzat oferta pentru pasul respectiv
    st.session_state.refused = {}
if "ring_cycle" not in st.session_state:
    st.session_state.ring_cycle = 0

_, _init_month_hours, _ = _month_window(int(st.session_state.ring_cycle))
if "current_step" not in st.session_state:
    # Pornim cu 2 ore ÎNAINTEA unui vârf de surplus (> 5 MW, PZU < 25), ca prima ofertă să fie pe T+2
    candidates = (
        feed[(feed["surplus_mw"] > 5.0) & (feed["day_ahead_price_eur"] < 25.0)]
        if "surplus_mw" in feed.columns
        else feed.iloc[0:0]
    )
    if len(candidates) > 1:
        peak_step = int(candidates.index[1])
    elif len(candidates) > 0:
        peak_step = int(candidates.index[0])
    else:
        noon = feed[feed["hour"].isin([12, 13])]
        peak_step = int(noon.index[0]) if len(noon) > 0 else 13
    st.session_state.current_step = int((peak_step - LOOKAHEAD_HOURS) % N_ROWS)

# Notificările generate în callback-uri sunt afișate la începutul următoarei rulări

# Notificările generate în callback-uri sunt afișate la începutul următoarei rulări
_toast_msg = st.session_state.pop("toast_msg", None)
if _toast_msg:
    st.toast(_toast_msg)

# Avansarea automată în Ring Buffer (când se termină luna curentă la 23:00, trece în ziua 1 ora 00:00 a lunii următoare)
if st.session_state.pending_advance:
    st.session_state.pending_advance = False
    _, _cur_month_hours, _ = _month_window(int(st.session_state.ring_cycle))
    nxt_step = int(st.session_state.current_step) + 1
    if nxt_step >= _cur_month_hours:
        st.session_state.current_step = 0
        st.session_state.ring_cycle = int(st.session_state.ring_cycle) + 1
    else:
        st.session_state.current_step = nxt_step

CYCLE_START_DT, N_ROWS, CYCLE_OFFSET_HOURS = _month_window(int(st.session_state.ring_cycle))
st.session_state.current_step = int(min(max(0, int(st.session_state.current_step)), N_ROWS - 1))

agent: VoltComputeAgent = st.session_state.agent


def _ring_row(v_idx: int) -> pd.Series:
    """Returnează rândul din Ring Buffer corespunzător indexului virtual continuu `v_idx`."""
    return feed.iloc[v_idx % N_FEED]


def _ring_dt(v_idx: int) -> datetime:
    """Returnează timestamp-ul continuu monoton pentru indexul virtual `v_idx` (pornind din luna Windows API)."""
    return BASE_LIVE_DT + timedelta(hours=int(v_idx))


def _cycle_variation(v_idx: int) -> Tuple[float, float, float]:
    """
    La ciclurile ulterioare ale Ring Buffer-ului (`v_cycle != 0`), aplică o micro-variație
    stocastică deterministă (nor tranzitoriu + fluctuație de sarcină) astfel încât fluxul live
    să genereze mereu date noi, fără să repete identic aceiași megawați.
    """
    v_cycle = v_idx // N_FEED
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


def grid_telemetry(v_idx: int) -> GridTelemetry:
    """Telemetria rețelei (meteo + PZU) la pasul `v_idx`, validată prin contractul Pydantic."""
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
    """Senzori cu valori > 3σ la pasul `v_idx` (-> Status UNCERTAIN)."""
    return agent.detect_anomalies(grid_telemetry(v_idx))


def step_price(v_idx: int) -> float:
    return float(_ring_row(v_idx)["day_ahead_price_eur"])


def predict_step(v_idx: int) -> Tuple[float, float, Dict[str, float], float]:
    """
    Calculează predicția modelului AI (antrenat DOAR pe train_energy_data.csv)
    pe datele noi din Ring Buffer la pasul `v_idx`:
    returnează (surplus_predicted_mw, confidence, shap_dict, predicted_injection_mw).
    """
    cache = st.session_state.pred_cache
    if v_idx not in cache or len(cache[v_idx]) != 4:
        grid = grid_telemetry(v_idx)
        surplus_pred, conf, shap_dict = agent.estimate(grid)
        inj_pred = agent.estimate_injection(grid)
        cache[v_idx] = (surplus_pred, conf, shap_dict, inj_pred)
    return cache[v_idx]


# =====================================================================
# MOTORUL DE OFERTE PE DATA CENTRE (SECVENȚIAL, CU REFUZ -> URMĂTORUL NOD)
# =====================================================================
# Ordinea ofertării NU e fixă: la fiecare oră nodurile sunt clasate determinist după Match Score
# (services/router.py): 0.4·VolumFit + 0.4·RatăAcceptareIstorică − 0.2·PenalizareTermică.


def _surplus_trigger(step: int) -> bool:
    """Prag de activare: orice depășire prezisă a liniei de 10 MW primește ofertă."""
    return predict_step(step)[0] > agent.MIN_SURPLUS_THRESHOLD_MW


def offer_target_mw(step: int) -> float:
    """
    Cât se contractează pentru ora `step`: DOAR surplusul prezis peste linia de 10 MW.
    Când estimarea ajunge la limită, oferta se oprește (nu vindem energie de sub linie).
    """
    return predict_step(step)[0]


def step_uncertain(step: int) -> bool:
    """Surplus de arbitrat, dar senzorii raportează valori anormale (> 3σ) -> UNCERTAIN."""
    return _surplus_trigger(step) and bool(step_anomalies(step))


def step_qualifies(step: int) -> bool:
    """Pasul are surplus peste prag, PZU sub prag, senzori valizi și încredere suficientă."""
    return (
        _surplus_trigger(step)
        and not step_anomalies(step)
        and predict_step(step)[1] >= agent.CONFIDENCE_THRESHOLD
    )


def step_escalated(step: int) -> bool:
    """Pasul are surplus, dar încrederea modelului e sub prag -> decizie la dispecerul uman."""
    return (
        _surplus_trigger(step)
        and not step_anomalies(step)
        and predict_step(step)[1] < agent.CONFIDENCE_THRESHOLD
    )


def delivered_mw(step: int) -> float:
    """MW deja contractați (acceptați de data centre) la pasul respectiv."""
    return round(sum(st.session_state.deals.get(step, {}).values()), 3)


def remaining_surplus(step: int) -> float:
    """Surplus prezis rămas peste linie, care nu a fost încă preluat de niciun DC."""
    return max(0.0, offer_target_mw(step) - delivered_mw(step))


def node_open_for_step(step: int, node: Dict[str, Any]) -> bool:
    """Nodul nu a acceptat și nu a refuzat încă oferta pentru acest pas."""
    return (
        node["id"] not in st.session_state.deals.get(step, {})
        and node["id"] not in st.session_state.refused.get(step, set())
    )


def datacenter_snapshot(step: int, node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Returnează telemetria CSV pentru nodul și ora din ring buffer."""
    feed_row = _ring_row(step)
    timestamp = pd.Timestamp(feed_row["dt"])
    snapshot = DATACENTER_LOOKUP.get((timestamp, node["id"]))
    if snapshot is None:
        snapshot = DATACENTER_HOUR_LOOKUP.get((int(feed_row["hour"]), node["id"]))
    return dict(snapshot) if snapshot is not None else None


def delivered_to_node(step: int, node: Dict[str, Any]) -> float:
    """MW livrați nodului la pasul respectiv, după acceptarea ofertei."""
    return float(st.session_state.deals.get(step, {}).get(node["id"], 0.0))


def delivered_temperature(step: int, node: Dict[str, Any]) -> Optional[float]:
    """Temperatura actualizată după ce energia acceptată ajunge la nod."""
    snapshot = datacenter_snapshot(step, node)
    if snapshot is None:
        return None
    return float(snapshot["temp_servere_c"]) + delivered_to_node(step, node) * TEMP_RISE_PER_MW_C


def node_thermal_limit(snapshot: Dict[str, Any]) -> float:
    """Limita termică efectivă: cea din telemetrie, dar niciodată peste filtrul HVAC de 26°C."""
    return min(float(snapshot["limita_termica_c"]), HVAC_LIMIT_C)


def thermal_rejected(step: int, node: Dict[str, Any]) -> bool:
    """Filtru siguranță termică: nodul este exclus dacă HVAC > 26°C (după energia deja livrată)."""
    snapshot = datacenter_snapshot(step, node)
    if snapshot is None:
        return False
    return float(delivered_temperature(step, node)) > node_thermal_limit(snapshot)


def node_free_capacity(step: int, node: Dict[str, Any]) -> float:
    """MW pe care nodul îi mai poate absorbi: min(capacitate liberă, rezervă termică până la 26°C)."""
    snapshot = datacenter_snapshot(step, node)
    if snapshot is None:
        return 0.0
    temp = float(delivered_temperature(step, node))
    free_capacity = max(0.0, float(snapshot["capacitate_libera_mw"]) - delivered_to_node(step, node))
    thermal_capacity = max(0.0, (node_thermal_limit(snapshot) - temp) / TEMP_RISE_PER_MW_C)
    return min(free_capacity, thermal_capacity)


def offer_capacity(step: int, node: Dict[str, Any], remaining_mw: float) -> float:
    """Oferta maximă pentru nod: plafonată de surplusul rămas, capacitate și rezerva termică."""
    if thermal_rejected(step, node):
        return 0.0
    return max(0.0, min(float(remaining_mw), node_free_capacity(step, node)))


def node_ranking(step: int) -> List[Dict[str, Any]]:
    """
    DECIDE: clasamentul determinist al nodurilor pentru ora `step` (Match Score + filtru HVAC),
    folosind telemetria CSV și istoricul ofertelor (actualizat live la Aprobă / Refuză).
    """
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
    """
    Alege oferta curentă: data center-ul cu cea mai mare capacitate disponibilă la primul pas ofertat.
    Returnează (oferta, epuizat). `epuizat=True` înseamnă că a mai rămas surplus necontractat
    și nu mai există niciun data center disponibil căruia să i se trimită oferta.
    """
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
    """Callback pentru butonul Acceptă (rulează înaintea scriptului, deci nu se pierde în Play)."""
    for step, mw in per_step.items():
        st.session_state.deals.setdefault(step, {})[node_id] = mw
    st.session_state.pending_advance = False
    hist = st.session_state.dc_history[node_id]
    hist["offers_sent"] += 1
    hist["offers_accepted"] += 1
    total = sum(per_step.values())
    st.session_state.toast_msg = (
        f"✅ Tranzacție aprobată: {node_label} trece în COMPUTING cu {total:.2f} MWh la {FIXED_PRICE_EUR:.0f} €/MWh."
    )


def _refuse_offer(node_id: str, node_label: str, per_step: Dict[int, float]) -> None:
    """Callback pentru butonul Refuză: oferta trece la următorul data center disponibil."""
    for step in per_step:
        st.session_state.refused.setdefault(step, set()).add(node_id)
    st.session_state.dc_history[node_id]["offers_sent"] += 1
    st.session_state.pending_advance = False
    st.session_state.toast_msg = (
        f"⛔ {node_label} a refuzat. Tariful rămâne {FIXED_PRICE_EUR:.0f} €/MWh — oferta sare la următorul nod."
    )


# =====================================================================
# HEADER + TIMELINE CONTROL BAR
# =====================================================================
def _reset_offers(steps: List[int]) -> None:
    """
    Șterge acceptările și refuzurile pentru intervalul curent (util în demo) și anulează
    efectul lor asupra istoricului de acceptare, ca Match Score-ul să revină la valoarea anterioară.
    """
    for step in steps:
        for node_id in st.session_state.deals.pop(step, {}):
            hist = st.session_state.dc_history[node_id]
            hist["offers_sent"] = max(0, hist["offers_sent"] - 1)
            hist["offers_accepted"] = max(0, hist["offers_accepted"] - 1)
        for node_id in st.session_state.refused.pop(step, set()):
            hist = st.session_state.dc_history[node_id]
            hist["offers_sent"] = max(0, hist["offers_sent"] - 1)
    st.session_state.pending_advance = False
    st.session_state.toast_msg = "↩️ Ofertele pentru interval au fost resetate (istoricul DC restaurat)."


def _step_back() -> None:
    """Pas Înapoi: revine cu o oră (trece din 00:00 în 23:00 a ciclului anterior)."""
    st.session_state.autoplay = False
    prv = int(st.session_state.current_step) - 1
    if prv < 0:
        prev_cycle = int(st.session_state.ring_cycle) - 1
        _, prev_month_hours, _ = _month_window(prev_cycle)
        st.session_state.ring_cycle = prev_cycle
        st.session_state.current_step = prev_month_hours - 1
    else:
        st.session_state.current_step = prv


def _next_hour() -> None:
    """Pas Înainte / Simulează Ora Următoare: avansează o oră (Ring Buffer, 23:00 -> 00:00 fără oprire)."""
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


abs_T = CYCLE_OFFSET_HOURS + int(st.session_state.current_step)
# Calculăm eroarea MAE live pe ultimele 24h (doar orele de zi cu activitate solară) pentru afișare transparentă
_day_hist_errors = [
    abs(net_injection_mw(i) - predict_step(i)[3])
    for i in range(abs_T - HISTORY_HOURS, abs_T + 1)
    if 7 <= int(_ring_row(i)["hour"]) <= 18
]
live_window_mae = float(np.mean(_day_hist_errors)) if _day_hist_errors else 0.42

hdr_main, hdr_map_btn = st.columns([5.6, 1.4], gap="small")
with hdr_main:
    st.markdown(
        f"""
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
  <div class="vc-title"><span class="vc-live"></span>VOLTCOMPUTE // PPC GRID DISPATCH</div>
  <div class="vc-sub">LUNA CURENTĂ: {CYCLE_START_DT.strftime('%Y-%m')} (Windows API) · {N_ROWS}h în lună (Luna #{int(st.session_state.ring_cycle) + 1}) · MAE Live (24h): ±{live_window_mae:.2f} MW</div>
</div>
""",
        unsafe_allow_html=True,
    )
with hdr_map_btn:
    st.page_link("pages/1_Harta_Retea_SCADA.py", label="🗺️ Harta Rețea SCADA", width="stretch")

current_cycle = int(st.session_state.ring_cycle)
slider_key = f"timeline_slider_{current_cycle}"
st.session_state[slider_key] = int(st.session_state.current_step)


def _on_slider_change(s_key: str) -> None:
    st.session_state.autoplay = False
    st.session_state.current_step = int(st.session_state[s_key])


bar_back, bar_fwd, bar_play, bar_slider = st.columns([1, 1.5, 1.5, 4.5], gap="small")
bar_back.button(
    "◀ Pas Înapoi",
    on_click=_step_back,
    width="stretch",
    help="Revino cu o oră în dataset (oprește Live Feed).",
)
bar_fwd.button(
    "⏭ Simulează Ora Următoare",
    on_click=_next_hour,
    type="primary",
    width="stretch",
    help="Pas Înainte: avansează o oră în telemetria din fișiere.",
)
bar_play.button(
    "⏹ Stop Live Feed" if st.session_state.autoplay else "▶▶ Play Live Feed",
    on_click=_toggle_play,
    width="stretch",
    help=(
        f"Avansează automat câte o oră la fiecare {AUTOPLAY_SECONDS:.0f}s. Se oprește singur când "
        "apare o ofertă care cere decizia dispecerului (Aprobă / Refuză)."
    ),
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
        help="Sari direct la orice oră din setul de date live (out-of-sample).",
    )

# =====================================================================
# STATE AT STEP T + LOOKAHEAD (IN RING BUFFER)
# =====================================================================
T = CYCLE_OFFSET_HOURS + int(st.session_state.current_step)
row_T = _ring_row(T)
dt_T = _ring_dt(T)
surplus_T, conf_T, shap_T, pred_inj_T = predict_step(T)

delivered_T = delivered_mw(T)
revenue_T = round(delivered_T * FIXED_PRICE_EUR, 2)  # profit PPC (apare DOAR în interfața PPC)
is_dispatched = delivered_T > 0.01

# Profitul PPC se contabilizează la contractare și rămâne cumulat (nu dispare când trece ora)
cumulative_profit = sum(delivered_mw(step) * FIXED_PRICE_EUR for step in st.session_state.deals)

# Curentul real (injecția netă măsurată în teren) vs predicția AI și după dispecerizare
raw_injection_T = net_injection_mw(T)
actual_injection_T = max(0.0, raw_injection_T - delivered_T)
grid_limit_mw = float(row_T.get("grid_capacity_limit_mw", 10.0))
actual_surplus_T = max(0.0, actual_injection_T - grid_limit_mw) if is_dispatched else surplus_T

future_idx = [T + k for k in range(1, LOOKAHEAD_HOURS + 1)]
future_surplus = {i: predict_step(i)[0] for i in future_idx}

# Ora țintă = T+2. La fiecare oră nouă se ofertează DOAR ora T+2, pe baza predicției făcute acum.
# Fiecare oră țintă pornește cu 0 oferte; oferta merge la un singur DC o dată (în ordinea clasamentului).
target_idx = T + LOOKAHEAD_HOURS
target_str = _ring_dt(target_idx).strftime("%H:%M")
start_str = _ring_dt(target_idx).strftime("%H:%M")
end_str = _ring_dt(target_idx + 1).strftime("%H:%M")
target_pred_mw, target_conf, _, _ = predict_step(target_idx)
target_contracted_mw = delivered_mw(target_idx)
target_remaining_mw = max(0.0, target_pred_mw - target_contracted_mw)

# FEREASTRA DE OFERTARE (2 ore): când la ora T modelul prezice surplus de arbitraj pentru H = T+2,
# oferta pentru H apare imediat ca notificare și rămâne deschisă 2 ore (la T = H-2 și T = H-1).
# Când ora H ajunge pe grafic (T = H), oferta expiră. Fiecare oră țintă pornește cu 0 oferte.
open_targets = [h for h in range(T + 1, T + LOOKAHEAD_HOURS + 1) if step_qualifies(h)]
offers_by_target = {h: build_offer([h]) for h in open_targets}  # h -> (ofertă curentă, epuizat)
pending_offers = {h: o for h, (o, _) in offers_by_target.items() if o is not None}
exhausted_targets = [h for h, (o, ex) in offers_by_target.items() if o is None and ex]
offer_steps = open_targets
offer = pending_offers[min(pending_offers)] if pending_offers else None  # cea mai urgentă ofertă
pending_node_ids = {o["node"]["id"] for o in pending_offers.values()}
table_steps = open_targets if open_targets else [target_idx]
if open_targets:
    start_str = _ring_dt(min(open_targets)).strftime("%H:%M")
    end_str = _ring_dt(max(open_targets) + 1).strftime("%H:%M")
# Starea pool-ului de ofertare (ALL_DCS_EXHAUSTED = au refuzat toate nodurile eligibile)
pool_status = POOL_ALL_DCS_EXHAUSTED if exhausted_targets else None

uncertain_steps = [h for h in future_idx if step_uncertain(h)]
escalated_steps = [h for h in future_idx if step_escalated(h)]

peak_candidates = [(T, surplus_T)] + [(i, s) for i, s in future_surplus.items()]
peak_idx, peak_mw = max(peak_candidates, key=lambda x: x[1])

if is_dispatched:
    alert_html = (
        f'<div class="vc-alert vc-alert-green">✅ REȚEA STABILIZATĂ: {delivered_T:.2f} MW de surplus '
        f'au fost preluați de data centre la {dt_T.strftime("%H:%M")}. '
        f'Curentul a scăzut de la {raw_injection_T:.2f} MW la {actual_injection_T:.2f} MW · '
        f'Profit PPC: +{revenue_T:,.2f} €</div>'
    )
elif pool_status == POOL_ALL_DCS_EXHAUSTED:
    alert_html = (
        '<div class="vc-alert vc-alert-red">🚫 ALL_DCS_EXHAUSTED: toate data centerele eligibile au refuzat '
        f'oferta la {FIXED_PRICE_EUR:.0f} €/MWh. Se solicită intervenția dispeceratului; riscul de curtailment rămâne.</div>'
    )
elif uncertain_steps:
    u = uncertain_steps[0]
    alert_html = (
        f'<div class="vc-alert vc-alert-amber">❓ STATUS: UNCERTAIN la {_ring_dt(u).strftime("%H:%M")} — senzori cu '
        f'valori anormale (&gt; 3σ): {"; ".join(step_anomalies(u))}. Alocarea automată este suspendată, '
        f'operatorul uman a fost notificat.</div>'
    )
elif escalated_steps:
    e = escalated_steps[0]
    alert_html = (
        f'<div class="vc-alert vc-alert-amber">🧑‍💼 ESCALATE_TO_HUMAN la {_ring_dt(e).strftime("%H:%M")}: '
        f'confidența modelului ({predict_step(e)[1] * 100:.0f}%) este sub pragul de '
        f'{agent.CONFIDENCE_THRESHOLD * 100:.0f}%. Decizia rămâne la dispecer.</div>'
    )
else:
    alert_html = ""  # Rețea în parametri normali: nu afișăm niciun banner

col_ppc, col_dc = st.columns([0.72, 0.28], gap="large")

# =====================================================================
# STÂNGA: INTERFAȚA PPC (dispecerat, profit, grafic, tabel)
# =====================================================================
with col_ppc:
    st.markdown('<span id="vc-split-marker"></span>', unsafe_allow_html=True)
    # Slot fix pentru banner: când apare/dispare, restul elementelor NU își schimbă poziția
    alert_slot = st.empty()
    if alert_html:
        alert_slot.markdown(alert_html, unsafe_allow_html=True)
    else:
        alert_slot.empty()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Iradiere Curentă",
        f"{float(row_T['solar_irradiance_wm2']):.0f} W/m²",
        f"{dt_T.strftime('%H:%M')} · {float(row_T['ambient_temp_celsius']):.1f}°C",
    )
    if is_dispatched:
        m2.metric(
            "Injecție Rețea PPC",
            f"{actual_injection_T:.2f} MW",
            f"-{delivered_T:.2f} MW evacuați (stabilizată ✅)",
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
        f"Surplus Prezis AI (T+2 · {target_str})",
        f"{target_remaining_mw:.2f} MW",
        f"prezis {target_pred_mw:.2f} − contractat {target_contracted_mw:.2f} MW",
        delta_color="off",
    )
    m4.metric(
        "Profit Arbitraj PPC",
        f"+{cumulative_profit:,.2f} €",
        f"+{revenue_T:,.2f} € la ora curentă" if revenue_T > 0 else f"PZU {float(row_T['day_ahead_price_eur']):.1f} €/MWh",
        delta_color="normal" if revenue_T > 0 else "off",
    )

    # ---------------- Live ticker chart (Ring Buffer Continuu) ----------------
    start_i = T - HISTORY_HOURS
    end_i = T + LOOKAHEAD_HOURS
    window = list(range(start_i, end_i + 1))
    xs = [_ring_dt(i) for i in window]
    limit_line = [float(_ring_row(i).get("grid_capacity_limit_mw", 10.0)) for i in window]

    def _dispatched_real_inj_for_step(step_idx: int) -> float:
        """Valoarea reală măsurată în rețea (minus MW contractați de data centre)."""
        return max(0.0, net_injection_mw(step_idx) - delivered_mw(step_idx))

    def _dispatched_ai_pred_inj_for_step(step_idx: int) -> float:
        """Valoarea prezisă de modelul AI (minus MW contractați de data centre)."""
        return max(0.0, predict_step(step_idx)[3] - delivered_mw(step_idx))

    # Curba verde: Injecția REALĂ măsurată în rețea până la momentul T
    inj_hist = [_dispatched_real_inj_for_step(i) if i <= T else None for i in window]

    # Curba albastră punctată: Predicția Modelului AI (istoric T-24..T pentru comparație live + viitor)
    ai_pred_full = [_dispatched_ai_pred_inj_for_step(i) for i in window]

    t_x = dt_T
    live_value = actual_injection_T
    prev_value = _dispatched_real_inj_for_step(T - 1)
    tick_delta = live_value - prev_value
    tick_color = "#00FF9D" if is_dispatched else ("#00FF9D" if tick_delta >= 0 else "#FF4B4B")
    tick_arrow = "▲" if tick_delta >= 0 else "▼"
    if is_dispatched:
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

    # Anotații pe grafic pentru viitor
    for f_i in future_idx:
        f_dt = _ring_dt(f_i)
        s_f, _, _, pred_inj_f = predict_step(f_i)
        if delivered_mw(f_i) > 0.01:
            fig.add_annotation(
                x=f_dt, y=limit_line[window.index(f_i)],
                text=f"✅ T+{f_i - T} CONTRACTAT",
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

    # ---------------- Tabel arbitraj (sincronizat cu ofertele și răspunsurile data centerelor) ----------------
    ranking = node_ranking(table_steps[0])
    rank_pos = {r["id"]: i + 1 for i, r in enumerate(ranking)}

    def node_status(node: Dict[str, Any]) -> str:
        if any(datacenter_snapshot(s, node) is None for s in table_steps):
            return "NO_TELEMETRY"
        if any(node["id"] in st.session_state.deals.get(s, {}) for s in table_steps):
            return "COMPUTING"
        if any(node["id"] in st.session_state.refused.get(s, set()) for s in table_steps):
            return "REJECTED"
        if any(thermal_rejected(s, node) for s in table_steps):
            return "INELIGIBLE_HVAC"
        if node["id"] in pending_node_ids:
            return "OFFER_PENDING"
        if any(step_uncertain(s) for s in table_steps):
            return "UNCERTAIN"
        if any(step_escalated(s) for s in table_steps):
            return "ESCALATE_TO_HUMAN"
        if any(step_qualifies(s) and remaining_surplus(s) > 0.05 for s in table_steps):
            return "NO_CAPACITY" if pool_status == POOL_ALL_DCS_EXHAUSTED else "IN_QUEUE"
        return "STANDBY"

    statuses = {n["id"]: node_status(n) for n in DATACENTERS}

    # Denumirile de status din tabelul original
    TABLE_STATUS = {
        "COMPUTING": "ACCEPTED",
        "REJECTED": "REFUSED",
        "INELIGIBLE_HVAC": "REJECTED_THERMAL_SAFETY",
    }

    table_rows = []
    for node in DATACENTERS:
        table_snapshot = datacenter_snapshot(table_steps[0], node)
        contracted = round(
            sum(st.session_state.deals.get(s, {}).get(node["id"], 0.0) for s in table_steps), 2
        )
        pending_mwh = round(
            sum(o["total_mwh"] for o in pending_offers.values() if o["node"]["id"] == node["id"]), 2
        )
        # Profit / economie: cumulat din TOATE orele contractate (nu dispare când trece ora)
        total_contracted = sum(d.get(node["id"], 0.0) for d in st.session_state.deals.values())
        current_temp = delivered_temperature(table_steps[0], node)
        if current_temp is None:
            current_temp = float("nan")
        free_capacity = (
            max(0.0, float(table_snapshot["capacitate_libera_mw"]) - contracted)
            if table_snapshot is not None
            else 0.0
        )
        contract_price = (
            float(table_snapshot["pret_contract_eur_mwh"]) if table_snapshot is not None else float("nan")
        )
        table_rows.append(
            {
                "Nod": node["name"],
                "Preț Contract (€/MWh)": contract_price,
                "Temp Servere (°C)": current_temp,
                "Capacitate liberă (MW)": free_capacity,
                "Status Agent": TABLE_STATUS.get(statuses[node["id"]], statuses[node["id"]]),
                "MWh în Ofertă": pending_mwh,
                "MWh Contractați": contracted,
                "Profit PPC (€)": round(total_contracted * FIXED_PRICE_EUR, 2),
                "Economie Data Center (€)": round(total_contracted * (REFERENCE_TARIFF_EUR - FIXED_PRICE_EUR), 2),
            }
        )
    table = pd.DataFrame(table_rows)

    st.markdown(
        '<div class="vc-sub" style="margin:4px 0;">ARBITRAJ MULTI-NOD · '
        + f"FEREASTRĂ DE OFERTARE: {start_str} – {end_str}"
        + "</div>",
        unsafe_allow_html=True,
    )

    def _status_color(value: str) -> str:
        palette = {
            "ACCEPTED": "color:#00FF9D;font-weight:600",
            "OFFER_PENDING": "color:#F59E0B;font-weight:600",
            "REFUSED": "color:#F87171;font-weight:600",
            "REJECTED_THERMAL_SAFETY": "color:#FF4B4B;font-weight:600",
            "ESCALATE_TO_HUMAN": "color:#F59E0B;font-weight:600",
            "UNCERTAIN": "color:#F59E0B;font-weight:600",
            "IN_QUEUE": "color:#38BDF8;font-weight:600",
            "NO_CAPACITY": "color:#F87171;font-weight:600",
            "NO_TELEMETRY": "color:#8E95A5;font-weight:600",
        }
        return palette.get(value, "color:#8E95A5")

    st.dataframe(
        table.style.map(_status_color, subset=["Status Agent"]).format(
            {
                "Preț Contract (€/MWh)": "{:.2f}",
                "Temp Servere (°C)": "{:.1f}",
                "Capacitate liberă (MW)": "{:.2f}",
                "MWh în Ofertă": "{:.2f}",
                "MWh Contractați": "{:.2f}",
                "Profit PPC (€)": "{:,.2f}",
                "Economie Data Center (€)": "{:,.2f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )

# =====================================================================
# DREAPTA: ④ ACT — carduri Data Center + Human-in-the-Loop (Aprobă / Refuză)
# =====================================================================
with col_dc:
    st.markdown(
        '<span id="vc-dc-marker"></span>'
        '<div class="vc-pane-head vc-pane-dc">④ ACT · DATA CENTERS</div>'
        f'<div class="vc-pane-sub">Pool routing la tarif FIX {FIXED_PRICE_EUR:.0f} €/MWh · refuzul nu scade prețul</div>',
        unsafe_allow_html=True,
    )

    savings_per_mwh = REFERENCE_TARIFF_EUR - FIXED_PRICE_EUR

    # Sloturi fixe (câte unul pentru fiecare oră din fereastra de 2 ore + unul pentru „fără ofertă”):
    # pozițiile elementelor nu se schimbă între rulări, deci nu rămân carduri/butoane „fantomă”.
    offer_slots = [st.empty() for _ in range(LOOKAHEAD_HOURS)]
    no_offer_slot = st.empty()
    for slot_i in range(len(open_targets), LOOKAHEAD_HOURS):
        offer_slots[slot_i].empty()

    for slot_i, h in enumerate(sorted(open_targets)):
        with offer_slots[slot_i].container():
            h_offer, h_exhausted = offers_by_target[h]
            h_label = f"{_ring_dt(h).strftime('%H:%M')} – {_ring_dt(h + 1).strftime('%H:%M')}"
            hours_left = h - T  # 2 = oferta tocmai a apărut, 1 = ultima oră pentru decizie
            expiry_txt = (
                "expiră peste 1 oră — ultima oră pentru decizie" if hours_left == 1 else f"expiră peste {hours_left} ore"
            )
            h_responded = bool(st.session_state.deals.get(h) or st.session_state.refused.get(h))

            if h_offer is not None:
                node = h_offer["node"]
                per_step = h_offer["per_step"]
                total_mwh = h_offer["total_mwh"]
                queue_after = [
                    DC_BY_ID[r["id"]]["short"]
                    for r in node_ranking(h)
                    if r["eligible"] and r["id"] != node["id"] and node_open_for_step(h, DC_BY_ID[r["id"]])
                ]
                queue_txt = " → ".join(queue_after) if queue_after else "niciun alt nod (→ ALL_DCS_EXHAUSTED)"
                st.markdown(
                    f"""
    <div class="vc-modal" style="border:1px solid rgba(245,158,11,0.6);">
      <div class="vc-modal-head">🔔 OFERTĂ PENTRU ORA {h_label} · {expiry_txt.upper()}</div>
      <div class="vc-modal-title">{node['name']} (Match Score {h_offer['match_score']:.3f})</div>
      <div class="vc-modal-body">
        Modelul prezice <span class="vc-mono">{predict_step(h)[0]:.2f} MW</span> peste linie la {_ring_dt(h).strftime('%H:%M')}.
        Oferta: <span class="vc-mono" style="color:#00FF9D;">{total_mwh:.2f} MWh</span> la tariful fix
        <span class="vc-mono" style="color:#00FF9D;">{FIXED_PRICE_EUR:.0f} €/MWh</span>
        (vs {REFERENCE_TARIFF_EUR:.0f} €/MWh tarif standard).
        <br>💶 Venit PPC: <span class="vc-mono">{total_mwh * FIXED_PRICE_EUR:,.2f} €</span>
        · Economie DC: <span class="vc-mono">{total_mwh * savings_per_mwh:,.2f} €</span>
        <br><span style="color:#8E95A5;font-size:0.8rem;">La refuz, aceeași ofertă (același preț) pleacă la: {queue_txt}</span>
      </div>
    </div>
    """,
                    unsafe_allow_html=True,
                )
                btn_accept, btn_refuse = st.columns(2)
                btn_accept.button(
                    "✅ Aprobă Tranzacția",
                    key=f"offer_accept_{h}",  # cheie stabilă pe toată fereastra de 2 ore a orei țintă
                    type="primary",
                    width="stretch",
                    on_click=_accept_offer,
                    args=(node["id"], node["short"], dict(per_step)),
                )
                btn_refuse.button(
                    "⛔ Refuză Oferta",
                    key=f"offer_refuse_{h}",
                    width="stretch",
                    on_click=_refuse_offer,
                    args=(node["id"], node["short"], dict(per_step)),
                )
                if h_responded:
                    st.button(
                        "↩️ Resetează ofertele",
                        key=f"offer_reset_pending_{h}",
                        width="stretch",
                        on_click=_reset_offers,
                        args=([h],),
                        help="Anulează răspunsurile deja date pentru această oră.",
                    )

            elif h_exhausted:
                st.markdown(
                    f"""
    <div class="vc-modal" style="border:1px solid rgba(255,75,75,0.6);">
      <div class="vc-modal-head" style="color:#FF7B7B;">🚫 ALL_DCS_EXHAUSTED · ORA {h_label}</div>
      <div class="vc-modal-body">
        Niciun Data Center eligibil nu mai poate prelua oferta la {FIXED_PRICE_EUR:.0f} €/MWh.
        <span class="vc-mono" style="color:#FF7B7B;">{remaining_surplus(h):.2f} MWh</span> rămân necontractați —
        se solicită intervenția dispeceratului ({expiry_txt}).
      </div>
    </div>
    """,
                    unsafe_allow_html=True,
                )
                st.button(
                    "↩️ Resetează ofertele",
                    key=f"offer_reset_exhausted_{h}",
                    width="stretch",
                    on_click=_reset_offers,
                    args=([h],),
                    help="Anulează refuzurile/acceptările pentru această oră și reia ofertarea de la primul clasat.",
                )

            else:
                contracted_total = sum(st.session_state.deals.get(h, {}).values())
                st.markdown(
                    f"""
    <div class="vc-modal" style="border:1px solid rgba(0,255,157,0.5);">
      <div class="vc-modal-head" style="color:#00FF9D;">✅ SURPLUS ALOCAT INTEGRAL · ORA {h_label}</div>
      <div class="vc-modal-body">
        {contracted_total:.2f} MWh contractați la {FIXED_PRICE_EUR:.0f} €/MWh ·
        venit PPC {contracted_total * FIXED_PRICE_EUR:,.2f} €.
      </div>
    </div>
    """,
                    unsafe_allow_html=True,
                )
                st.button(
                    "↩️ Resetează ofertele",
                    key=f"offer_reset_done_{h}",
                    width="stretch",
                    on_click=_reset_offers,
                    args=([h],),
                    help="Anulează tranzacțiile pentru această oră și reia ofertarea de la primul clasat.",
                )

    if open_targets:
        no_offer_slot.empty()
    else:
        no_offer_slot.markdown(
            f"""
<div class="vc-modal">
  <div class="vc-modal-head">⚡ FĂRĂ OFERTĂ</div>
  <div class="vc-modal-body">
    Modelul nu prezice nicio depășire a liniei de 10 MW în următoarele {LOOKAHEAD_HOURS} ore.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

# =====================================================================
# AUTO-STEP CONTINUU ÎN RING BUFFER (FĂRĂ OPRIRE LA FINALUL DATELOR)
# =====================================================================
# Human-in-the-Loop: Live Feed-ul se oprește singur când o ofertă așteaptă decizia dispecerului.
if st.session_state.autoplay:
    paused_targets = st.session_state.setdefault("autopaused_targets", set())
    new_targets = [h for h in sorted(pending_offers) if h not in paused_targets]
    if new_targets:
        # Pauză o singură dată per oră țintă nouă; la un nou Play, feed-ul poate trece mai departe
        paused_targets.update(new_targets)
        st.session_state.autoplay = False
        st.toast(
            f"⏸ Live Feed oprit: ofertă nouă pentru ora {_ring_dt(new_targets[0]).strftime('%H:%M')} "
            "— ai 2 ore să o aprobi sau să o refuzi."
        )
        st.rerun()
    else:
        time.sleep(AUTOPLAY_SECONDS)
        st.session_state.pending_advance = True
        st.rerun()
