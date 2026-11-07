import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agent import VoltComputeAgent
from schemas import (
    AgentDecision,
    DispatchAction,
    GridTelemetry,
    ResearchCenterTelemetry,
)
from simulator import SCENARIO_DESCRIPTIONS, VoltComputeSimulator

st.set_page_config(
    page_title="VOLTCOMPUTE // PPC Dispatch System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =====================================================================
# 1. SWISS CLEAN DARK MODE DESIGN SYSTEM & CUSTOM CSS
# =====================================================================
SWISS_DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg-abyssal: #090A0F;
    --bg-surface: #11131A;
    --bg-surface-elevated: #161922;
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-hover: rgba(255, 255, 255, 0.16);
    --text-primary: #F3F4F6;
    --text-muted: #8E95A5;
    --accent-green: #00FF9D;
    --accent-green-dim: rgba(0, 255, 157, 0.14);
    --accent-red: #FF4B4B;
    --accent-red-dim: rgba(255, 75, 75, 0.14);
    --accent-amber: #F59E0B;
    --accent-amber-dim: rgba(245, 158, 11, 0.15);
    --accent-blue: #38BDF8;
    --accent-blue-dim: rgba(56, 189, 248, 0.14);
}

/* Hide Streamlit Branding & Chrome */
#MainMenu {visibility: hidden !important;}
footer {visibility: hidden !important;}
header {visibility: hidden !important;}
[data-testid="stToolbar"] {visibility: hidden !important;}
[data-testid="stDecoration"] {display: none !important;}

/* Global Background & Typography */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: var(--bg-abyssal) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    letter-spacing: -0.02em;
}

.block-container {
    padding-top: 1.1rem !important;
    padding-bottom: 1.5rem !important;
    padding-left: 1.8rem !important;
    padding-right: 1.8rem !important;
    max-width: 100% !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: var(--bg-surface) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

/* Top Navigation & Status Bar */
.vc-topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 12px 20px;
    margin-bottom: 16px;
}

.vc-topbar-left {
    display: flex;
    align-items: center;
    gap: 14px;
}

.vc-brand {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 1.15rem;
    letter-spacing: 0.04em;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    gap: 8px;
}

.vc-brand-slash {
    color: var(--accent-green);
    font-weight: 700;
}

.vc-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(0, 255, 157, 0.08);
    border: 1px solid rgba(0, 255, 157, 0.28);
    color: var(--accent-green);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 4px 11px;
    border-radius: 999px;
    letter-spacing: 0.03em;
}

.vc-pulse-dot {
    width: 8px;
    height: 8px;
    background-color: var(--accent-green);
    border-radius: 50%;
    display: inline-block;
    animation: pulse 1.8s infinite ease-in-out;
}

@keyframes pulse {
    0% {
        transform: scale(0.92);
        box-shadow: 0 0 0 0 rgba(0, 255, 157, 0.75);
    }
    70% {
        transform: scale(1.08);
        box-shadow: 0 0 0 7px rgba(0, 255, 157, 0);
    }
    100% {
        transform: scale(0.92);
        box-shadow: 0 0 0 0 rgba(0, 255, 157, 0);
    }
}

.vc-topbar-right {
    display: flex;
    align-items: center;
    gap: 10px;
}

.vc-meta-badge {
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-subtle);
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    padding: 5px 11px;
    border-radius: 8px;
}

.vc-meta-badge strong {
    color: var(--text-primary);
}

/* Bento Surface Cards */
.bento-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    transition: border-color 0.15s ease;
}

.bento-card:hover {
    border-color: var(--border-hover);
}

.bento-section-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Telemetry Micro-Cards (Left Panel) */
.telemetry-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 12px;
}

.micro-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 12px 14px;
}

.micro-label {
    font-size: 0.71rem;
    font-weight: 500;
    color: var(--text-muted);
    margin-bottom: 5px;
    letter-spacing: -0.01em;
}

.micro-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.22rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.15;
}

.micro-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    margin-top: 4px;
}

/* Hero Decision Banners (Center Panel) */
.hero-banner {
    border-radius: 12px;
    padding: 22px 24px;
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
}

.hero-offer {
    background: linear-gradient(135deg, rgba(0, 255, 157, 0.11) 0%, rgba(17, 19, 26, 0.96) 100%);
    border: 1px solid rgba(0, 255, 157, 0.45);
    box-shadow: 0 0 25px rgba(0, 255, 157, 0.15);
}

.hero-escalate {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.14) 0%, rgba(17, 19, 26, 0.96) 100%);
    border: 1px solid rgba(245, 158, 11, 0.55);
    box-shadow: 0 0 22px rgba(245, 158, 11, 0.12);
}

.hero-thermal {
    background: linear-gradient(135deg, rgba(255, 75, 75, 0.14) 0%, rgba(17, 19, 26, 0.96) 100%);
    border: 1px solid rgba(255, 75, 75, 0.55);
    box-shadow: 0 0 22px rgba(255, 75, 75, 0.14);
}

.hero-normal {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
}

.hero-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.hero-headline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.2;
    margin-bottom: 10px;
}

.hero-meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-muted);
    border-top: 1px solid rgba(255, 255, 255, 0.07);
    padding-top: 12px;
    margin-top: 12px;
}

/* SHAP Horizontal Micro-Bars (Right Panel) */
.shap-row {
    margin-bottom: 11px;
}

.shap-header {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    margin-bottom: 4px;
}

.shap-bar-track {
    width: 100%;
    height: 7px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 4px;
    overflow: hidden;
}

.shap-bar-fill {
    height: 100%;
    border-radius: 4px;
}

/* Retro Terminal Reasoning Box */
.terminal-box {
    background: #06070B;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    padding: 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.55;
    color: #D1D5DB;
}

.terminal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    padding-bottom: 8px;
    margin-bottom: 10px;
    font-size: 0.68rem;
    color: var(--text-muted);
}

.terminal-dots {
    display: inline-flex;
    gap: 5px;
}

.terminal-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
}

/* Pitch Tab KPI Cards */
.kpi-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 18px 20px;
}

.kpi-bracket-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.52rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    margin-bottom: 6px;
}

.kpi-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-primary);
}

.kpi-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-top: 3px;
}

/* Custom Styling for Streamlit Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: var(--bg-surface);
    padding: 6px;
    border-radius: 10px;
    border: 1px solid var(--border-subtle);
}

.stTabs [data-baseweb="tab"] {
    height: 36px;
    border-radius: 7px;
    padding: 0 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-muted);
}

.stTabs [aria-selected="true"] {
    background-color: var(--bg-surface-elevated) !important;
    color: var(--accent-green) !important;
    border: 1px solid rgba(0, 255, 157, 0.25) !important;
}
</style>
"""

st.markdown(SWISS_DARK_CSS, unsafe_allow_html=True)

# =====================================================================
# INITIALIZE STATE & SERVICES
# =====================================================================
if "agent" not in st.session_state:
    st.session_state.agent = VoltComputeAgent()
if "simulator" not in st.session_state:
    st.session_state.simulator = VoltComputeSimulator(agent=st.session_state.agent)
if "dispatch_history" not in st.session_state:
    st.session_state.dispatch_history = []
if "last_decision" not in st.session_state:
    st.session_state.last_decision = None
if "last_center" not in st.session_state:
    st.session_state.last_center = None

agent: VoltComputeAgent = st.session_state.agent
simulator: VoltComputeSimulator = st.session_state.simulator

# Load ML MAE metric for header badge
metrics_path = Path("models/metrics.json")
mae_display = "0.039 MW"
if metrics_path.exists():
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        mae_val = metrics_data.get("xgboost_regressor", {}).get("mae", 0.039)
        mae_display = f"{mae_val:.3f} MW"
    except Exception:
        mae_display = "0.039 MW"

now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

# =====================================================================
# 2. HEADER & STATUS BAR WITH LIVE PULSE
# =====================================================================
st.markdown(
    f"""
<div class="vc-topbar">
    <div class="vc-topbar-left">
        <div class="vc-brand">
            <span class="vc-brand-slash">⚡</span> VOLTCOMPUTE
        </div>
        <div class="vc-status-pill">
            <span class="vc-pulse-dot"></span>
            ● PPC DISPATCH SYSTEM — ONLINE
        </div>
    </div>
    <div class="vc-topbar-right">
        <div class="vc-meta-badge">
            MODEL: <strong>XGBoost v1.2 (MAE: {mae_display})</strong>
        </div>
        <div class="vc-meta-badge">
            SYNC: <strong>{now_str}</strong>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


def _execute_dispatch_action(
    decision_obj: AgentDecision,
    center_obj: ResearchCenterTelemetry,
    action_type: str,
) -> Dict[str, Any]:
    """Executes APPROVED or REJECTED dispatch via agent.act() and logs it."""
    action_msg = DispatchAction(
        decision_id=decision_obj.decision_id,
        center_id=center_obj.center_id,
        action=action_type,  # type: ignore[arg-type]
    )
    if action_type == "APPROVED" and decision_obj.status == "ESCALATE_TO_HUMAN":
        decision_obj.allocated_mw = round(
            min(decision_obj.surplus_predicted_mw, center_obj.max_absorb_capacity_mw),
            3,
        )
    act_result = agent.act(decision_obj, center_obj, action_msg)
    st.session_state.dispatch_history.insert(
        0,
        {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "decision_id": action_msg.decision_id,
            "center_id": action_msg.center_id,
            "action": action_msg.action,
            "effective_mw": act_result["effective_mw"],
            "execution_state": act_result["execution_state"],
            "tariff_eur": act_result["tariff_eur_mwh"],
        },
    )
    return act_result


# =====================================================================
# NAVIGATION TABS
# =====================================================================
tab_bento, tab_hitl, tab_pitch = st.tabs(
    [
        "01 // LIVE BENTO TELEMETRY & DISPATCH",
        "02 // HUMAN-IN-THE-LOOP CONSOLE",
        "03 // LIVE 24H PITCH DEMO",
    ]
)

# =====================================================================
# TAB 1: BENTO GRID (25% TELEMETRY | 50% HERO DECISION | 25% XAI & AUDIT)
# =====================================================================
with tab_bento:
    # Interactive Control Strip inside a clean collapsible or compact bar
    with st.expander("🎛️ TELEMETRY SIMULATOR CONTROLS (ADJUST LIVE SENSORS)", expanded=True):
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        with s_col1:
            hour = st.slider("Hour of Day (00-23h)", min_value=0, max_value=23, value=13)
            solar_irradiance = st.slider(
                "Solar Irradiance (W/m²)",
                min_value=0.0,
                max_value=1300.0,
                value=1080.0,
                step=10.0,
            )
        with s_col2:
            ambient_temp = st.slider(
                "Ambient Temp (°C)",
                min_value=-10.0,
                max_value=45.0,
                value=26.5,
                step=0.5,
            )
            da_price = st.slider(
                "PZU Spot Price (€/MWh)",
                min_value=-20.0,
                max_value=200.0,
                value=10.0,
                step=1.0,
            )
        with s_col3:
            grid_limit = st.slider(
                "PPC Line Limit (MW)",
                min_value=5.0,
                max_value=20.0,
                value=10.0,
                step=0.5,
            )
            confidence_slider = st.slider(
                "Model Confidence (0.30 - 1.00)",
                min_value=0.30,
                max_value=1.00,
                value=0.95,
                step=0.01,
            )
        with s_col4:
            center_id = st.selectbox(
                "Target Academic Node",
                [
                    "ROEDUNET-UPB-AI-CLUSTER",
                    "UVT-TIMISOARA-HPC",
                    "TUCN-CLUJ-CLOUD",
                    "IFIN-HH-MAGURELE",
                ],
            )
            cooling_temp = st.slider(
                "Rack Cooling (°C) [Max 25°C]",
                min_value=12.0,
                max_value=35.0,
                value=19.5,
                step=0.5,
            )
            max_absorb = st.slider(
                "Max Node Absorb (MW)",
                min_value=0.5,
                max_value=10.0,
                value=6.0,
                step=0.5,
            )
            pending_jobs = st.slider(
                "Pending HPC Jobs",
                min_value=0,
                max_value=50,
                value=28,
                step=1,
            )

    # Execute Agent Cycle explicitly: perceive() -> estimate() -> decide()
    raw_grid_input = {
        "timestamp": datetime.now().isoformat(),
        "hour": hour,
        "solar_irradiance_wm2": solar_irradiance,
        "ambient_temp_celsius": ambient_temp,
        "day_ahead_price_eur": da_price,
        "grid_capacity_limit_mw": grid_limit,
        "weather_stability_index": confidence_slider,
    }
    raw_center_input = {
        "center_id": center_id,
        "cooling_temp_celsius": cooling_temp,
        "max_absorb_capacity_mw": max_absorb,
        "pending_compute_jobs": pending_jobs,
    }

    grid_valid, center_valid, perceive_err = agent.perceive(
        raw_grid_input, raw_center_input
    )
    if perceive_err is not None or grid_valid is None or center_valid is None:
        decision = perceive_err  # type: ignore[assignment]
        center_obj = ResearchCenterTelemetry(
            center_id=center_id,
            cooling_temp_celsius=cooling_temp,
            max_absorb_capacity_mw=max_absorb,
            pending_compute_jobs=pending_jobs,
        )
    else:
        surplus_mw, conf_score, top_shap = agent.estimate(
            grid_valid, confidence_override=confidence_slider
        )
        decision = agent.decide(
            grid_valid, center_valid, surplus_mw, conf_score, top_shap
        )
        center_obj = center_valid

    st.session_state.last_decision = decision
    st.session_state.last_center = center_obj

    # Compute Net Grid Load for telemetry display
    temp_derating = 1.0 - max(0.0, (ambient_temp - 25.0) * 0.004)
    gross_solar_mw = (solar_irradiance / 1000.0) * 17.8 * temp_derating
    local_load_mw = 1.4 + 0.5 * float(np.sin(np.pi * hour / 12.0))
    net_grid_load_mw = round(max(0.0, gross_solar_mw - local_load_mw), 2)

    # =================================================================
    # 3. BENTO GRID 3-COLUMN LAYOUT (25% | 50% | 25%)
    # =================================================================
    col_left, col_center, col_right = st.columns([1, 2, 1], gap="medium")

    # -----------------------------------------------------------------
    # LEFT PANEL (25%): TELEMETRY MICRO-CARDS
    # -----------------------------------------------------------------
    with col_left:
        grid_over_limit = net_grid_load_mw > grid_limit
        grid_load_color = "#FF4B4B" if grid_over_limit else "#00FF9D"
        cooling_over_limit = cooling_temp > agent.THERMAL_LIMIT_CELSIUS
        cooling_color = "#FF4B4B" if cooling_over_limit else "#38BDF8"
        pzu_color = "#00FF9D" if da_price <= 30.0 else "#F3F4F6"

        st.markdown(
            f"""
<div class="bento-card">
    <div class="bento-section-title">
        <span>01 // GRID & NODE TELEMETRY</span>
        <span style="font-family:'JetBrains Mono',monospace;color:#38BDF8;">LIVE</span>
    </div>
    <div class="telemetry-grid">
        <div class="micro-card">
            <div class="micro-label">Solar Irradiance</div>
            <div class="micro-value">{solar_irradiance:.0f} <span style="font-size:0.78rem;color:#8E95A5;">W/m²</span></div>
            <div class="micro-sub" style="color:#00FF9D;">Ambient: {ambient_temp:.1f}°C</div>
        </div>
        <div class="micro-card">
            <div class="micro-label">Grid Load</div>
            <div class="micro-value" style="color:{grid_load_color};">{net_grid_load_mw:.2f} <span style="font-size:0.78rem;color:#8E95A5;">MW</span></div>
            <div class="micro-sub" style="color:{grid_load_color};">Limit: {grid_limit:.1f} MW</div>
        </div>
        <div class="micro-card">
            <div class="micro-label">Rack Cooling</div>
            <div class="micro-value" style="color:{cooling_color};">{cooling_temp:.1f} <span style="font-size:0.78rem;color:#8E95A5;">°C</span></div>
            <div class="micro-sub" style="color:{cooling_color};">Max Safe: 25.0°C</div>
        </div>
        <div class="micro-card">
            <div class="micro-label">PZU Spot Price</div>
            <div class="micro-value" style="color:{pzu_color};">{da_price:.1f} <span style="font-size:0.78rem;color:#8E95A5;">€/MWh</span></div>
            <div class="micro-sub" style="color:#8E95A5;">Green Fixed: 30.0 €</div>
        </div>
    </div>
    <div class="micro-card" style="border-color:rgba(56, 189, 248, 0.25);background:rgba(56, 189, 248, 0.05);">
        <div class="micro-label" style="color:#38BDF8;">ACADEMIC COMPUTE TARGET</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.9rem;font-weight:700;color:#F3F4F6;margin-bottom:4px;">
            {center_id}
        </div>
        <div style="display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#8E95A5;">
            <span>SLURM QUEUE: <strong style="color:#38BDF8;">{pending_jobs} JOBS</strong></span>
            <span>CAP: <strong style="color:#F3F4F6;">{max_absorb:.1f} MW</strong></span>
        </div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------
    # CENTER PANEL (50%): HERO DECISION BANNER & EXECUTION METRICS
    # -----------------------------------------------------------------
    with col_center:
        if decision.status == "OFFER_GENERATED":
            banner_class = "hero-offer"
            accent_hex = "#00FF9D"
            tag_label = "● AUTONOMOUS ARBITRAGE ACTIVE // VIRTUAL BATTERY ENGAGED"
            headline_text = (
                f"OFFER DISPATCHED: {decision.allocated_mw:.2f} MW @ {decision.fixed_price_eur:.2f} €/MWh"
            )
        elif decision.status == "ESCALATE_TO_HUMAN":
            banner_class = "hero-escalate"
            accent_hex = "#F59E0B"
            tag_label = "▲ HUMAN-IN-THE-LOOP REQUIRED // LOW CONFIDENCE THRESHOLD (<0.75)"
            headline_text = (
                f"ESCALATE TO DISPATCHER: {decision.surplus_predicted_mw:.2f} MW PENDING"
            )
        elif cooling_temp > agent.THERMAL_LIMIT_CELSIUS:
            banner_class = "hero-thermal"
            accent_hex = "#FF4B4B"
            tag_label = "■ THERMAL INTERLOCK ENGAGED // RACK TEMP > 25.0°C"
            headline_text = (
                f"ALLOCATION BLOCKED: 0.00 MW ({cooling_temp:.1f}°C OVERHEAT RISK)"
            )
        elif decision.status == "DATA_ERROR":
            banner_class = "hero-thermal"
            accent_hex = "#FF4B4B"
            tag_label = "■ TELEMETRY CORRUPTION DETECTED // PYDANTIC V2 GUARD"
            headline_text = "DATA_ERROR: ZERO MW DISPATCHED"
        else:
            banner_class = "hero-normal"
            accent_hex = "#38BDF8"
            tag_label = "● NORMAL GRID OPERATION // NO CONGESTION HAZARD"
            headline_text = (
                f"GRID BALANCED: SURPLUS {decision.surplus_predicted_mw:.2f} MW (< 2.0 MW THRESHOLD)"
            )

        st.markdown(
            f"""
<div class="hero-banner {banner_class}">
    <div class="hero-tag" style="color:{accent_hex};">
        {tag_label}
    </div>
    <div class="hero-headline" style="color:{accent_hex};">
        {headline_text}
    </div>
    <div style="font-size:0.86rem;color:#D1D5DB;line-height:1.45;">
        Target Node: <strong style="font-family:'JetBrains Mono',monospace;color:#38BDF8;">{center_id}</strong>
        &nbsp;|&nbsp; Decision ID: <span style="font-family:'JetBrains Mono',monospace;color:#8E95A5;">{decision.decision_id}</span>
    </div>
    <div class="hero-meta-row">
        <span>PREDICTED SURPLUS: <strong style="color:#F3F4F6;">{decision.surplus_predicted_mw:.2f} MW</strong></span>
        <span>ALLOCATED HPC: <strong style="color:{accent_hex};">{decision.allocated_mw:.2f} MW</strong></span>
        <span>CONFIDENCE: <strong style="color:{accent_hex};">{decision.confidence_score * 100:.1f}%</strong></span>
        <span>HOURLY PPC REV: <strong style="color:#00FF9D;">+{decision.allocated_mw * decision.fixed_price_eur:.2f} €</strong></span>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # Manual confirmation button directly inside Center Panel when ESCALATE_TO_HUMAN is active
        if decision.status == "ESCALATE_TO_HUMAN":
            esc_col1, esc_col2 = st.columns(2)
            with esc_col1:
                if st.button(
                    f"⚡ CONFIRM & DISPATCH {min(decision.surplus_predicted_mw, max_absorb):.2f} MW MANUALLY",
                    key="bento_approve_btn",
                    use_container_width=True,
                    type="primary",
                ):
                    res = _execute_dispatch_action(decision, center_obj, "APPROVED")
                    st.success(
                        f"MANUAL OVERRIDE EXECUTED: {res['effective_mw']:.2f} MW dispatched to {center_id}."
                    )
            with esc_col2:
                if st.button(
                    "⛔ REJECT & CURTAIL SURPLUS",
                    key="bento_reject_btn",
                    use_container_width=True,
                ):
                    _execute_dispatch_action(decision, center_obj, "REJECTED")
                    st.error(f"DISPATCH REJECTED BY OPERATOR ({decision.decision_id}).")

        # Secondary Bento Card: Real-Time Power Balance Breakdown
        unabsorbed_mw = max(0.0, decision.surplus_predicted_mw - decision.allocated_mw)
        st.markdown(
            f"""
<div class="bento-card">
    <div class="bento-section-title">
        <span>02 // REAL-TIME POWER ARBITRAGE BALANCE</span>
        <span style="font-family:'JetBrains Mono',monospace;color:#8E95A5;">HOUR {hour:02d}:00</span>
    </div>
    <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:12px;">
        <div class="micro-card">
            <div class="micro-label">Predicted Surplus</div>
            <div class="micro-value" style="color:#FF4B4B;">{decision.surplus_predicted_mw:.2f} MW</div>
            <div class="micro-sub" style="color:#8E95A5;">Over {grid_limit:.1f} MW Line Cap</div>
        </div>
        <div class="micro-card">
            <div class="micro-label">Virtual Battery Absorb</div>
            <div class="micro-value" style="color:#00FF9D;">{decision.allocated_mw:.2f} MW</div>
            <div class="micro-sub" style="color:#38BDF8;">{pending_jobs} AI Jobs Triggered</div>
        </div>
        <div class="micro-card">
            <div class="micro-label">Residual Curtailment</div>
            <div class="micro-value" style="color:#F59E0B;">{unabsorbed_mw:.2f} MW</div>
            <div class="micro-sub" style="color:#8E95A5;">Unabsorbed Power</div>
        </div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------
    # RIGHT PANEL (25%): EXPLAINABILITY (SHAP BARS) & TERMINAL REASONING
    # -----------------------------------------------------------------
    with col_right:
        shap_html_rows = ""
        if decision.top_shap_features:
            max_abs_shap = max(
                max(abs(v) for v in decision.top_shap_features.values()), 0.1
            )
            for feat_name, feat_val in decision.top_shap_features.items():
                width_pct = min(100, int((abs(feat_val) / max_abs_shap) * 100))
                bar_color = "#00FF9D" if feat_val >= 0 else "#38BDF8"
                sign_prefix = "+" if feat_val >= 0 else ""
                shap_html_rows += f"""
<div class="shap-row">
    <div class="shap-header">
        <span style="color:#D1D5DB;">{feat_name}</span>
        <span style="color:{bar_color};font-weight:600;">{sign_prefix}{feat_val:.3f} MW</span>
    </div>
    <div class="shap-bar-track">
        <div class="shap-bar-fill" style="width:{width_pct}%;background:{bar_color};"></div>
    </div>
</div>
"""
        else:
            shap_html_rows = '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;color:#8E95A5;">No SHAP attributions available.</div>'

        st.markdown(
            f"""
<div class="bento-card">
    <div class="bento-section-title">
        <span>03 // SHAP FEATURE ATTRIBUTION</span>
        <span style="font-family:'JetBrains Mono',monospace;color:#00FF9D;">XAI</span>
    </div>
    {shap_html_rows}
</div>
<div class="bento-card">
    <div class="bento-section-title">
        <span>04 // AGENT REASONING TRACE</span>
        <span style="font-family:'JetBrains Mono',monospace;color:#8E95A5;">AUDIT</span>
    </div>
    <div class="terminal-box">
        <div class="terminal-header">
            <span class="terminal-dots">
                <span class="terminal-dot" style="background:#FF4B4B;"></span>
                <span class="terminal-dot" style="background:#F59E0B;"></span>
                <span class="terminal-dot" style="background:#00FF9D;"></span>
            </span>
            <span>voltcompute://decision-trace</span>
        </div>
        <div style="color:#00FF9D;margin-bottom:6px;">
            &gt; status: {decision.status} (conf={decision.confidence_score:.2f})
        </div>
        <div>
            &gt; {decision.reasoning}
        </div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        with st.expander("📄 Raw Pydantic Contract (`AgentDecision`)"):
            st.json(decision.model_dump())

# =====================================================================
# TAB 2: HUMAN-IN-THE-LOOP DISPATCHER CONSOLE
# =====================================================================
with tab_hitl:
    active_decision: AgentDecision = st.session_state.last_decision
    active_center: ResearchCenterTelemetry = st.session_state.last_center

    hitl_col1, hitl_col2 = st.columns([1.2, 1.8], gap="medium")

    with hitl_col1:
        st.markdown(
            f"""
<div class="bento-card">
    <div class="bento-section-title">
        <span>DISPATCHER OVERRIDE TERMINAL</span>
        <span style="font-family:'JetBrains Mono',monospace;color:#F59E0B;">HITL GATE</span>
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:#8E95A5;margin-bottom:12px;">
        DECISION ID: <strong style="color:#F3F4F6;">{active_decision.decision_id}</strong><br>
        TARGET NODE: <strong style="color:#38BDF8;">{active_center.center_id}</strong><br>
        CURRENT STATUS: <strong style="color:#00FF9D;">{active_decision.status}</strong><br>
        PREDICTED SURPLUS: <strong style="color:#F3F4F6;">{active_decision.surplus_predicted_mw:.2f} MW</strong>
    </div>
    <div class="terminal-box" style="margin-bottom:14px;">
        &gt; {active_decision.reasoning}
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        b1, b2 = st.columns(2)
        with b1:
            if st.button(
                "✅ [APROBĂ] DISPATCH (APPROVED)",
                key="hitl_tab_approve",
                use_container_width=True,
                type="primary",
            ):
                act_res = _execute_dispatch_action(
                    active_decision, active_center, "APPROVED"
                )
                st.success(
                    f"APPROVED: {act_res['effective_mw']:.2f} MW routed to {active_center.center_id} @ 30.00 €/MWh."
                )
        with b2:
            if st.button(
                "⛔ [RESPINGE] BLOCK (REJECTED)",
                key="hitl_tab_reject",
                use_container_width=True,
            ):
                act_res = _execute_dispatch_action(
                    active_decision, active_center, "REJECTED"
                )
                st.error(
                    f"REJECTED: Allocation aborted for {active_decision.decision_id} (0.0 MW)."
                )

    with hitl_col2:
        st.markdown(
            """
<div class="bento-card">
    <div class="bento-section-title">
        <span>SCADA & SLURM AUDIT TRAIL (`DispatchAction` LOG)</span>
        <span style="font-family:'JetBrains Mono',monospace;color:#00FF9D;">IMMUTABLE LOG</span>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.session_state.dispatch_history:
            st.dataframe(
                pd.DataFrame(st.session_state.dispatch_history),
                use_container_width=True,
                height=280,
            )
        else:
            st.info(
                "No manual dispatcher actions recorded in this session yet. Use [APROBĂ] or [RESPINGE] to log a SCADA dispatch event."
            )

# =====================================================================
# TAB 3: LIVE 24H PITCH DEMO (FLUID PLAYBACK & ULTRA-CLEAN PLOTLY)
# =====================================================================
with tab_pitch:
    ctrl_col1, ctrl_col2 = st.columns([1.3, 2.7], gap="medium")

    with ctrl_col1:
        scenario_choice = st.selectbox(
            "SELECT 24H SIMULATION SCENARIO",
            options=list(SCENARIO_DESCRIPTIONS.keys()),
            format_func=lambda k: {
                "CRITICAL_SOLAR_SUNDAY": "01 // CRITICAL_SOLAR_SUNDAY (Peak Arbitrage)",
                "UNCERTAIN_WEATHER_SHOCK": "02 // UNCERTAIN_WEATHER_SHOCK (Conf < 0.75 -> HITL)",
                "DATACENTER_THERMAL_RISK": "03 // DATACENTER_THERMAL_RISK (28.5°C Thermal Trip)",
            }.get(k, k),
        )
    with ctrl_col2:
        st.markdown(
            f"""
<div class="bento-card" style="padding:12px 16px;margin-bottom:8px;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#38BDF8;margin-bottom:4px;">
        SCENARIO BRIEF // {scenario_choice}
    </div>
    <div style="font-size:0.84rem;color:#D1D5DB;">
        {SCENARIO_DESCRIPTIONS[scenario_choice]}
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    timeline = simulator.generate_24h_scenario(scenario_choice)
    decisions_list = [step["decision"] for step in timeline]
    impact = simulator.calculate_impact(decisions_list)

    mwh_saved = impact["mwh_saved_from_curtailment"]
    ppc_rev = impact["ppc_revenue_eur"]
    res_sav = impact["researcher_savings_eur"]
    co2_tons = impact["co2_avoided_tons"]

    # 4. FOUR KPI METRIC CARDS WITH MONOSPACE BRACKET FIGURES
    k_col1, k_col2, k_col3, k_col4 = st.columns(4, gap="medium")

    with k_col1:
        st.markdown(
            f"""
<div class="kpi-card">
    <div class="kpi-bracket-value" style="color:#00FF9D;">[ {mwh_saved:.1f} MWh ]</div>
    <div class="kpi-title">Curtailment Salvat</div>
    <div class="kpi-subtitle">din {impact['total_surplus_mwh']:.1f} MWh surplus total</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with k_col2:
        st.markdown(
            f"""
<div class="kpi-card">
    <div class="kpi-bracket-value" style="color:#00FF9D;">[ +{ppc_rev:,.2f} € ]</div>
    <div class="kpi-title">Venit Generat PPC</div>
    <div class="kpi-subtitle">{impact['offers_generated_hours']}h arbitraj @ 30.00 €/MWh</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with k_col3:
        st.markdown(
            f"""
<div class="kpi-card">
    <div class="kpi-bracket-value" style="color:#38BDF8;">[ -{res_sav:,.2f} € ]</div>
    <div class="kpi-title">Economii Noduri Cercetare</div>
    <div class="kpi-subtitle">vs tarif retail (115.00 €/MWh)</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with k_col4:
        st.markdown(
            f"""
<div class="kpi-card">
    <div class="kpi-bracket-value" style="color:#00FF9D;">[ {co2_tons:.2f} t ]</div>
    <div class="kpi-title">CO₂ Evitat</div>
    <div class="kpi-subtitle">Escaladări HITL: {impact['human_escalations_hours']}h</div>
</div>
""",
            unsafe_allow_html=True,
        )

    # ULTRA-CLEAN PLOTLY CHART (DARK THEME, TRANSPARENT BG, NEON LINES)
    hours_labels = [step["hour_label"] for step in timeline]
    net_solar = [step["net_solar_generation_mw"] for step in timeline]
    grid_limits = [step["grid_capacity_limit_mw"] for step in timeline]
    allocated_mw = [step["decision"]["allocated_mw"] for step in timeline]
    surplus_mw = [step["decision"]["surplus_predicted_mw"] for step in timeline]
    statuses = [step["decision"]["status"] for step in timeline]
    absorbed_top_curve = [
        limit + alloc for limit, alloc in zip(grid_limits, allocated_mw)
    ]

    fig = go.Figure()

    # 1. Solar Curve with subtle green gradient fill to zero
    fig.add_trace(
        go.Scatter(
            x=hours_labels,
            y=net_solar,
            mode="lines+markers",
            name="Net Solar Injection (MW)",
            line=dict(color="#00FF9D", width=2.5),
            marker=dict(size=6, color="#00FF9D"),
            fill="tozeroy",
            fillcolor="rgba(0, 255, 157, 0.10)",
            customdata=list(zip(surplus_mw, allocated_mw, statuses)),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Net Solar: %{y:.2f} MW<br>"
                "Predicted Surplus: %{customdata[0]:.2f} MW<br>"
                "HPC Absorbed: %{customdata[1]:.2f} MW<br>"
                "Status: %{customdata[2]}<extra></extra>"
            ),
        )
    )

    # 2. Grid Capacity Line (10 MW) dashed red
    fig.add_trace(
        go.Scatter(
            x=hours_labels,
            y=grid_limits,
            mode="lines",
            name="PPC Grid Safety Limit (10.0 MW)",
            line=dict(color="#FF4B4B", width=2, dash="dash"),
        )
    )

    # 3. Absorbed Virtual Battery Envelope (above 10 MW line)
    fig.add_trace(
        go.Scatter(
            x=hours_labels,
            y=absorbed_top_curve,
            mode="lines",
            name="Absorbed by RoEduNet HPC (MW)",
            line=dict(color="#38BDF8", width=2),
            fill="tonexty",
            fillcolor="rgba(56, 189, 248, 0.28)",
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#8E95A5", size=11),
        title=dict(
            text=f"24H TELEMETRY PLAYBACK // {scenario_choice}",
            font=dict(family="JetBrains Mono, monospace", size=13, color="#F3F4F6"),
            x=0.0,
        ),
        xaxis=dict(
            title="HOUR OF DAY (UTC)",
            gridcolor="rgba(255, 255, 255, 0.05)",
            zerolinecolor="rgba(255, 255, 255, 0.08)",
        ),
        yaxis=dict(
            title="POWER (MW)",
            gridcolor="rgba(255, 255, 255, 0.05)",
            zerolinecolor="rgba(255, 255, 255, 0.08)",
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#D1D5DB"),
        ),
        height=430,
        margin=dict(l=20, r=20, t=55, b=20),
    )

    st.markdown('<div class="bento-card" style="margin-top:12px;">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📋 24H HOURLY DISPATCH TELEMETRY MATRIX"):
        table_rows = []
        for step in timeline:
            dec = step["decision"]
            grid_t = step["grid_telemetry"]
            center_t = step["center_telemetry"]
            table_rows.append(
                {
                    "Hour": step["hour_label"],
                    "Irradiance (W/m²)": grid_t["solar_irradiance_wm2"],
                    "Net Load (MW)": step["net_solar_generation_mw"],
                    "Surplus (MW)": dec["surplus_predicted_mw"],
                    "Confidence": dec["confidence_score"],
                    "Rack Temp (°C)": center_t["cooling_temp_celsius"],
                    "Status": dec["status"],
                    "Allocated HPC (MW)": dec["allocated_mw"],
                    "Reasoning": dec["reasoning"],
                }
            )
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, height=340)
