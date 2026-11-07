import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agent import VoltComputeAgent
from schemas import DispatchAction, GridTelemetry, ResearchCenterTelemetry
from simulator import SCENARIO_DESCRIPTIONS, VoltComputeSimulator

st.set_page_config(
    page_title="VoltCompute | 24h to Power PPC",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inițializare agent și simulator în session_state
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

# Sidebar cu informații despre proiect și metricile modelului ML
with st.sidebar:
    st.markdown("## ⚡ VoltCompute")
    st.caption("**Smarthack XI — 24h to Power PPC**")
    st.markdown(
        """
        **Baterie Virtuală Academică:**
        Arbitraj AI în timp real al surplusului solar PPC către centre universitare de calcul
        (RoEduNet, clustere AI/HPC) la tarif verde fix de **30 EUR/MWh**.
        """
    )
    st.divider()
    st.markdown("### Constrângeri Siguranță Agent")
    st.markdown(
        """
        - **Prag Confidență:** `>= 0.75` *(sub prag -> Escaladare Dispecer)*
        - **Limită Termică Servere:** `<= 25.0 °C` *(peste limită -> Blocare Alocare)*
        - **Prag Activare Surplus:** `> 2.0 MW`
        - **Tarif Verde Fix:** `30.0 EUR/MWh`
        """
    )

    metrics_path = Path("models/metrics.json")
    if metrics_path.exists():
        st.divider()
        st.markdown("### Performanță Model ML")
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        st.metric(
            "XGBoost MAE (Test)",
            f"{metrics['xgboost_regressor']['mae']:.3f} MW",
            f"-{metrics['improvement_pct']['mae_reduction_pct']}% vs Baseline",
        )
        st.metric(
            "XGBoost RMSE (Test)",
            f"{metrics['xgboost_regressor']['rmse']:.3f} MW",
            f"-{metrics['improvement_pct']['rmse_reduction_pct']}% vs Baseline",
        )

st.title("⚡ VoltCompute — AI Arbitrage & Virtual Academic Battery for PPC")
st.markdown(
    "Transformăm riscul de **curtailment solar** și supraîncălzire a liniilor PPC în **putere de calcul pentru cercetarea universitară**."
)

tab_manual, tab_hitl, tab_pitch = st.tabs(
    [
        "🎛️ Tab 1: Control Manual & Testare Live",
        "🛡️ Tab 2: Human-in-the-Loop (Dispecerat PPC)",
        "🚀 Tab 3: Live 24h Pitch Demo",
    ]
)

# =====================================================================
# TAB 1: CONTROL MANUAL CU SLIDERE PENTRU TESTARE LIVE
# =====================================================================
with tab_manual:
    st.subheader("Simulare Telemetrie în Timp Real (Parc Solar PPC & Nod RoEduNet)")

    col_grid, col_center = st.columns(2)

    with col_grid:
        st.markdown("#### ☀️ Telemetrie Rețea & Parc Solar PPC")
        hour = st.slider("Ora din zi (0 - 23)", min_value=0, max_value=23, value=13)
        solar_irradiance = st.slider(
            "Iradianță Solară (W/m²)", min_value=0.0, max_value=1300.0, value=1060.0, step=10.0
        )
        ambient_temp = st.slider(
            "Temperatură Ambientală (°C)", min_value=-10.0, max_value=45.0, value=27.0, step=0.5
        )
        da_price = st.slider(
            "Preț PZU / Day-Ahead (EUR/MWh)", min_value=-20.0, max_value=200.0, value=12.0, step=1.0
        )
        grid_limit = st.slider(
            "Limită Capacitate Linie PPC (MW)", min_value=5.0, max_value=20.0, value=10.0, step=0.5
        )
        confidence_slider = st.slider(
            "Stabilitate Meteo / Confidență Model (0.0 - 1.0)",
            min_value=0.30,
            max_value=1.00,
            value=0.94,
            step=0.01,
        )

    with col_center:
        st.markdown("#### 🖥️ Telemetrie Centru Universitar (HPC / AI)")
        center_id = st.selectbox(
            "Nod Academic Conectat",
            ["ROEDUNET-UPB-AI-CLUSTER", "UVT-TIMISOARA-HPC", "TUCN-CLUJ-CLOUD", "IFIN-HH-MAGURELE"],
        )
        cooling_temp = st.slider(
            "Temperatură Răcire Servere (°C) [Limită max: 25.0°C]",
            min_value=12.0,
            max_value=35.0,
            value=19.5,
            step=0.5,
        )
        max_absorb = st.slider(
            "Capacitate Maximă Absorbție Instantanee (MW)",
            min_value=0.5,
            max_value=10.0,
            value=6.0,
            step=0.5,
        )
        pending_jobs = st.slider(
            "Joburi AI / Simulări în Așteptare (Slurm Queue)",
            min_value=0,
            max_value=50,
            value=24,
            step=1,
        )

    grid_obj = GridTelemetry(
        timestamp=datetime.now().isoformat(),
        hour=hour,
        solar_irradiance_wm2=solar_irradiance,
        ambient_temp_celsius=ambient_temp,
        day_ahead_price_eur=da_price,
        grid_capacity_limit_mw=grid_limit,
        weather_stability_index=confidence_slider,
    )
    center_obj = ResearchCenterTelemetry(
        center_id=center_id,
        cooling_temp_celsius=cooling_temp,
        max_absorb_capacity_mw=max_absorb,
        pending_compute_jobs=pending_jobs,
    )

    decision = agent.run_cycle(
        grid_obj, center_obj, confidence_override=confidence_slider
    )
    st.session_state.last_decision = decision
    st.session_state.last_center = center_obj

    st.divider()
    st.markdown("### 🧠 Decizia Agentului VoltCompute")

    status_color_map = {
        "OFFER_GENERATED": "🟢 OFFER_GENERATED (Alocare Activă către Centrul Universitar)",
        "NORMAL_GRID_OPERATION": "🔵 NORMAL_GRID_OPERATION (Fără Transfer / Protecție Activă)",
        "ESCALATE_TO_HUMAN": "🟠 ESCALATE_TO_HUMAN (Necesită Intervenția Dispecerului PPC)",
        "DATA_ERROR": "🔴 DATA_ERROR (Date Senzori Invalide)",
    }
    st.markdown(f"**Status Decizie:** {status_color_map.get(decision.status, decision.status)}")
    st.info(f"**Explicație Agent (Reasoning):** {decision.reasoning}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Surplus Solar Estimat", f"{decision.surplus_predicted_mw:.2f} MW")
    m2.metric("Putere Alocată HPC", f"{decision.allocated_mw:.2f} MW")
    m3.metric("Scor Confidență", f"{decision.confidence_score * 100:.1f}%")
    m4.metric("Tarif Verde Fix", f"{decision.fixed_price_eur:.0f} EUR/MWh")

    col_shap, col_json = st.columns(2)
    with col_shap:
        st.markdown("#### 📊 Explicabilitate Predicție (Top SHAP Features)")
        if decision.top_shap_features:
            shap_df = pd.DataFrame(
                {
                    "Variabilă": list(decision.top_shap_features.keys()),
                    "Contribuție (MW)": list(decision.top_shap_features.values()),
                }
            )
            st.bar_chart(shap_df.set_index("Variabilă"))
    with col_json:
        st.markdown("#### 📄 Contract Pydantic `AgentDecision`")
        st.json(decision.model_dump())

# =====================================================================
# TAB 2: DASHBOARD DECIZIE HUMAN-IN-THE-LOOP ([APROBĂ] / [RESPINGE])
# =====================================================================
with tab_hitl:
    st.subheader("🛡️ Consolă Dispecer PPC — Human-in-the-Loop (HITL)")
    st.markdown(
        "Când scorul de confidență scade sub `0.75` sau când operatorul dorește auditarea manuală a ofertei, "
        "dispecerul validează decizia înainte de comutarea sarcinii în SCADA."
    )

    active_decision = st.session_state.last_decision
    active_center = st.session_state.last_center

    if active_decision is not None and active_center is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("ID Decizie Curentă", active_decision.decision_id)
        c2.metric("Nod Universitar Țintă", active_center.center_id)
        c3.metric(
            "Putere Propuse / Surplus",
            f"{active_decision.allocated_mw:.2f} / {active_decision.surplus_predicted_mw:.2f} MW",
        )

        st.warning(f"**Mesaj Agent:** {active_decision.reasoning}")

        btn_col1, btn_col2, _ = st.columns([1, 1, 2])
        with btn_col1:
            if st.button("✅ [Aprobă] Dispecerizarea (APPROVED)", use_container_width=True, type="primary"):
                action_obj = DispatchAction(
                    decision_id=active_decision.decision_id,
                    center_id=active_center.center_id,
                    action="APPROVED",
                )
                # Dacă era escaladată la om, la aprobare alocăm min(surplus, max_absorb)
                if active_decision.status == "ESCALATE_TO_HUMAN":
                    active_decision.allocated_mw = round(
                        min(active_decision.surplus_predicted_mw, active_center.max_absorb_capacity_mw),
                        3,
                    )
                act_res = agent.act(active_decision, active_center, action_obj)
                st.session_state.dispatch_history.insert(
                    0,
                    {
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "decision_id": action_obj.decision_id,
                        "center_id": action_obj.center_id,
                        "action": action_obj.action,
                        "effective_mw": act_res["effective_mw"],
                        "execution_state": act_res["execution_state"],
                    },
                )
                st.success(
                    f"Comandă APROBATĂ! S-au direcționat {act_res['effective_mw']} MW către {active_center.center_id}."
                )

        with btn_col2:
            if st.button("⛔ [Respinge] Comanda (REJECTED)", use_container_width=True):
                action_obj = DispatchAction(
                    decision_id=active_decision.decision_id,
                    center_id=active_center.center_id,
                    action="REJECTED",
                )
                act_res = agent.act(active_decision, active_center, action_obj)
                st.session_state.dispatch_history.insert(
                    0,
                    {
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "decision_id": action_obj.decision_id,
                        "center_id": action_obj.center_id,
                        "action": action_obj.action,
                        "effective_mw": act_res["effective_mw"],
                        "execution_state": act_res["execution_state"],
                    },
                )
                st.error(
                    f"Comandă RESPINSĂ de dispecer pentru {active_decision.decision_id}. Alocare blocată (0.0 MW)."
                )

    st.divider()
    st.markdown("#### 📜 Jurnal Audit Comenzi Dispecer (`DispatchAction` Log)")
    if st.session_state.dispatch_history:
        st.dataframe(pd.DataFrame(st.session_state.dispatch_history), use_container_width=True)
    else:
        st.caption("Nicio comandă manuală înregistrată încă în această sesiune. Folosește butoanele [Aprobă] sau [Respinge] de mai sus.")

# =====================================================================
# TAB 3: LIVE 24H PITCH DEMO
# =====================================================================
with tab_pitch:
    st.subheader("🚀 Live 24h Pitch Demo — Simulare Zi Completă")

    scenario_choice = st.selectbox(
        "Selectează Scenariul de Simulare pe 24h:",
        options=list(SCENARIO_DESCRIPTIONS.keys()),
        format_func=lambda k: {
            "CRITICAL_SOLAR_SUNDAY": "1️⃣ CRITICAL_SOLAR_SUNDAY — Supraproducție la prânz, absorbție maximă în nodul AI",
            "UNCERTAIN_WEATHER_SHOCK": "2️⃣ UNCERTAIN_WEATHER_SHOCK — Șoc meteo, confidență < 0.75, escaladare la dispecer",
            "DATACENTER_THERMAL_RISK": "3️⃣ DATACENTER_THERMAL_RISK — Răcire servere la 28.5°C (> 25°C), protecție termică activă",
        }.get(k, k),
    )

    st.info(f"**Descriere Scenariu:** {SCENARIO_DESCRIPTIONS[scenario_choice]}")

    timeline = simulator.generate_24h_scenario(scenario_choice)
    decisions_list = [step["decision"] for step in timeline]
    impact = simulator.calculate_impact(decisions_list)

    # Carduri mari de KPI
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "⚡ Energie Salvată (MWh)",
        f"{impact['mwh_saved_from_curtailment']:.2f} MWh",
        f"din {impact['total_surplus_mwh']:.2f} MWh surplus total",
    )
    k2.metric(
        "💶 Venit Nou PPC (30 EUR/MWh)",
        f"€ {impact['ppc_revenue_eur']:,.2f}",
        f"{impact['offers_generated_hours']} ore de arbitraj activ",
    )
    k3.metric(
        "🎓 Economii Centre Cercetare",
        f"€ {impact['researcher_savings_eur']:,.2f}",
        "vs tarif standard (115 EUR/MWh)",
    )
    k4.metric(
        "🌱 Emisii CO₂ Evitate",
        f"{impact['co2_avoided_tons']:.2f} tone CO₂",
        f"Escaladări umane: {impact['human_escalations_hours']} ore",
    )

    # Grafic Plotly: Producție solară vs Limita de 10 MW cu zona de surplus absorbit evidențiată
    hours_labels = [step["hour_label"] for step in timeline]
    net_solar = [step["net_solar_generation_mw"] for step in timeline]
    grid_limits = [step["grid_capacity_limit_mw"] for step in timeline]
    allocated_mw = [step["decision"]["allocated_mw"] for step in timeline]
    surplus_mw = [step["decision"]["surplus_predicted_mw"] for step in timeline]
    statuses = [step["decision"]["status"] for step in timeline]

    # Baza peste care se adaugă surplusul absorbit pentru umplerea zonei dintre 10 MW și (10 MW + allocated_mw)
    absorbed_top_curve = [
        limit + alloc for limit, alloc in zip(grid_limits, allocated_mw)
    ]

    fig = go.Figure()

    # 1. Linia de limită a rețelei (10 MW)
    fig.add_trace(
        go.Scatter(
            x=hours_labels,
            y=grid_limits,
            mode="lines",
            name="Limită Siguranță Rețea PPC (10.0 MW)",
            line=dict(color="#EF553B", width=2, dash="dash"),
        )
    )

    # 2. Zona de surplus absorbit de bateria virtuală academică (între 10 MW și 10 MW + allocated_mw)
    fig.add_trace(
        go.Scatter(
            x=hours_labels,
            y=absorbed_top_curve,
            mode="lines",
            name="Surplus Absorbit de Centrul Universitar (MW)",
            line=dict(color="#00CC96", width=2),
            fill="tonexty",
            fillcolor="rgba(0, 204, 150, 0.35)",
        )
    )

    # 3. Curba producției solare nete injectate în linie
    fig.add_trace(
        go.Scatter(
            x=hours_labels,
            y=net_solar,
            mode="lines+markers",
            name="Injecție Solară Netă Parc PPC (MW)",
            line=dict(color="#FFA15A", width=3),
            marker=dict(size=7),
            customdata=list(zip(surplus_mw, allocated_mw, statuses)),
            hovertemplate=(
                "<b>Ora %{x}</b><br>"
                "Injecție Solară Netă: %{y:.2f} MW<br>"
                "Surplus Peste Limită: %{customdata[0]:.2f} MW<br>"
                "Absorbit de HPC: %{customdata[1]:.2f} MW<br>"
                "Status Agent: %{customdata[2]}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"Profil 24h: Injecție Solară PPC vs Limită 10 MW & Surplus Absorbit ({scenario_choice})",
        xaxis_title="Ora din Zi (00:00 - 23:00)",
        yaxis_title="Putere Electrică (MW)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480,
        margin=dict(l=30, r=30, t=70, b=30),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Tabel detaliat pe cele 24 de ore
    st.markdown("#### 📋 Desfășurător Orar al Deciziilor pe 24h")
    table_rows = []
    for step in timeline:
        dec = step["decision"]
        grid_t = step["grid_telemetry"]
        center_t = step["center_telemetry"]
        table_rows.append(
            {
                "Ora": step["hour_label"],
                "Iradianță (W/m²)": grid_t["solar_irradiance_wm2"],
                "Injecție Netă (MW)": step["net_solar_generation_mw"],
                "Surplus Estimat (MW)": dec["surplus_predicted_mw"],
                "Confidență": dec["confidence_score"],
                "Temp Răcire (°C)": center_t["cooling_temp_celsius"],
                "Status Decizie": dec["status"],
                "Alocat HPC (MW)": dec["allocated_mw"],
                "Explicație Agent": dec["reasoning"],
            }
        )

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, height=380)

