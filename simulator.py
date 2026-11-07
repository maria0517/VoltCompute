from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal

import numpy as np

from agent import VoltComputeAgent
from schemas import AgentDecision, DispatchAction, GridTelemetry, ResearchCenterTelemetry
from train_pipeline import load_grid_status
from copy import deepcopy

ScenarioName = Literal[
    "CRITICAL_SOLAR_SUNDAY",
    "UNCERTAIN_WEATHER_SHOCK",
    "DATACENTER_THERMAL_RISK",
]

SCENARIO_DESCRIPTIONS: Dict[str, str] = {
    "CRITICAL_SOLAR_SUNDAY": (
        "Duminică însorită cu consum casnic/industrial redus: între orele 11:00-15:00 parcul solar PPC "
        "produce un surplus masiv peste limita de 10 MW a liniei. Centrul universitar RoEduNet este optim răcit "
        "(19.5°C) și absoarbe automat surplusul la 30 EUR/MWh pentru antrenarea de modele AI."
    ),
    "UNCERTAIN_WEATHER_SHOCK": (
        "Front atmosferic instabil cu nori rapizi și variații bruște de iradianță la prânz: scorul de confidență "
        "al agentului scade sub pragul de 0.75, declanșând automat ESCALATE_TO_HUMAN pentru intervenția dispecerului."
    ),
    "DATACENTER_THERMAL_RISK": (
        "Zi caniculară cu surplus solar ridicat, însă sistemul de răcire al centrului de cercetare suferă o avarie "
        "parțială și atinge 28.5°C la prânz (> 25.0°C). Agentul blochează transferul energetic pentru protecția "
        "termică a serverelor."
    ),
}


def load_scenario_status(scenario_name="REAL_CSV", step=35, custom=None):
    """Shared API/UI scenario contract. Named demos inject +8 MW during 09:00–16:00.

    The CSV grid/model_features remain the ML population; injection is a separate
    dispatch quota. Solar demo market price -15 is a separate scenario quote, not
    a fabricated model input. Returns load_grid_status keys plus base_grid,
    scenario_name, scenario_active and demo_injected_surplus_mw.
    """
    if scenario_name not in {"REAL_CSV", *SCENARIO_DESCRIPTIONS}:
        raise ValueError("Unknown scenario")
    status = deepcopy(load_grid_status(step))
    status["base_grid"] = deepcopy(status["grid"])
    grid, centers = status["grid"], status["data_centers"]
    active = scenario_name != "REAL_CSV" and 9 <= grid["hour"] <= 16
    grid["scenario_name"] = scenario_name
    grid["demo_injected_surplus_mw"] = 8. if active else 0.
    if active and scenario_name == "CRITICAL_SOLAR_SUNDAY":
        grid["demo_market_price_eur_mwh"] = -15.
        status["price_pzu_eur_mwh"] = -15.
    if active and scenario_name == "UNCERTAIN_WEATHER_SHOCK":
        grid["weather_stability_index"] = .58
    if active and scenario_name == "DATACENTER_THERMAL_RISK":
        centers[0]["temp_servere_c"] = 28.5
    custom = custom or {}
    grid.update(custom.get("grid", {}))
    status["data_centers"] = deepcopy(custom.get("data_centers", centers))
    status.update(scenario_name=scenario_name, scenario_active=active,
                  demo_injected_surplus_mw=grid["demo_injected_surplus_mw"],
                  confidence_override=custom.get("confidence_override"))
    return status


class VoltComputeSimulator:
    """
    Motorul de simulare pe 24 de ore folosit în Pitch & Demo pentru a demonstra:
    - Arbitrajul automat în condiții de supraproducție solară (CRITICAL_SOLAR_SUNDAY)
    - Escaladarea sigură la operatorul uman pe vreme instabilă (UNCERTAIN_WEATHER_SHOCK)
    - Protecția hardware-ului universitar la risc termic (DATACENTER_THERMAL_RISK)
    """

    STANDARD_RETAIL_PRICE_EUR: float = 115.0  # Preț standard plătit de universități în regim normal (EUR/MWh)
    FIXED_GREEN_PRICE_EUR: float = 30.0       # Tarif verde preferențial PPC prin VoltCompute (EUR/MWh)
    CO2_TONS_PER_MWH: float = 0.28            # Factor mediu intensitate carbon evitat în mixul regional (tCO2/MWh)

    def __init__(self, agent: VoltComputeAgent | None = None):
        self.agent = agent or VoltComputeAgent()

    def generate_24h_scenario(self, scenario_name: str) -> List[Dict[str, Any]]:
        """
        Generează cele 24 de intervale orare (00:00 -> 23:00) pentru scenariul selectat,
        rulând agentul VoltCompute la fiecare oră.
        """
        if scenario_name not in SCENARIO_DESCRIPTIONS:
            raise ValueError("Unknown scenario")

        base_date = datetime(2026, 6, 21, 0, 0, 0)
        timeline: List[Dict[str, Any]] = []

        # Profil orar determinist pe 24h pentru iradianță în zi senină de vară (W/m²)
        sunny_irradiance = [
            0, 0, 0, 0, 0, 35,
            160, 340, 560, 760, 910, 1020,
            1080, 1095, 1040, 930, 750, 520,
            290, 95, 0, 0, 0, 0,
        ]

        for hour in range(24):
            ts = (base_date + timedelta(hours=hour)).isoformat()
            irr = float(sunny_irradiance[hour])
            ambient_temp = round(17.0 + 11.0 * float(np.sin(np.pi * max(0, hour - 5) / 15.0)), 1)
            grid_limit = 10.0

            # Valori implicite centru de cercetare
            cooling_temp = 19.5
            max_absorb_mw = 6.0
            pending_jobs = 14 if 9 <= hour <= 18 else 4
            stability_idx = 0.94
            da_price = 82.0

            if scenario_name == "CRITICAL_SOLAR_SUNDAY":
                if 11 <= hour <= 15:
                    # Supraproducție puternică de duminică, preț PZU prăbușit aproape de 0
                    irr = float(sunny_irradiance[hour] * 1.04)
                    da_price = 5.0 if hour in (12, 13, 14) else 14.0
                    cooling_temp = 19.2
                    max_absorb_mw = 6.5
                    pending_jobs = 28
                    stability_idx = 0.96

            elif scenario_name == "UNCERTAIN_WEATHER_SHOCK":
                if 9 <= hour <= 16:
                    # Fluctuații severe de iradianță + volatilitate preț -> confidență < 0.75
                    irr = float(sunny_irradiance[hour] * 0.92)
                    da_price = 110.0
                    cooling_temp = 20.5
                    max_absorb_mw = 6.0
                    pending_jobs = 22
                    stability_idx = 0.58  # Sub pragul de 0.75 -> ESCALATE_TO_HUMAN

            elif scenario_name == "DATACENTER_THERMAL_RISK":
                if 9 <= hour <= 16:
                    # Surplus mare disponibil, dar chillerul centrului de date este supraîncărcat (28.5°C)
                    irr = float(sunny_irradiance[hour] * 1.03)
                    ambient_temp = 35.5
                    da_price = 8.0
                    cooling_temp = 28.5  # Peste limita de 25.0°C -> blocare alocare (NORMAL_GRID_OPERATION)
                    max_absorb_mw = 6.5
                    pending_jobs = 25
                    stability_idx = 0.93

            grid_telemetry = GridTelemetry(
                timestamp=ts,
                hour=hour,
                solar_irradiance_wm2=round(irr, 1),
                ambient_temp_celsius=ambient_temp,
                day_ahead_price_eur=da_price,
                grid_capacity_limit_mw=grid_limit,
                weather_stability_index=stability_idx,
            )

            center_telemetry = ResearchCenterTelemetry(
                center_id="ROEDUNET-UPB-AI-CLUSTER",
                cooling_temp_celsius=cooling_temp,
                max_absorb_capacity_mw=max_absorb_mw,
                pending_compute_jobs=pending_jobs,
            )

            decision = self.agent.run_cycle(
                grid_telemetry,
                center_telemetry,
                confidence_override=stability_idx,
            )

            # Historical demo explicitly simulates an operator accepting each safe offer.
            # This is not automatic production dispatch and sends no network request.
            if decision.status == "OFFER_GENERATED":
                result = self.agent.act(decision, dispatch_action=DispatchAction(
                    decision_id=decision.decision_id, center_id=center_telemetry.center_id, action="APPROVED"))
                decision = AgentDecision.model_validate(result["decision"])

            # Calculăm puterea netă injectată în rețea pentru vizualizarea Plotly
            temp_derating = 1.0 - max(0.0, (ambient_temp - 25.0) * 0.004)
            gross_solar_mw = (irr / 1000.0) * 17.8 * temp_derating
            local_load_mw = 1.4 + 0.5 * float(np.sin(np.pi * hour / 12.0))
            net_solar_generation_mw = round(max(0.0, gross_solar_mw - local_load_mw), 3)

            timeline.append(
                {
                    "hour": hour,
                    "hour_label": f"{hour:02d}:00",
                    "net_solar_generation_mw": net_solar_generation_mw,
                    "grid_capacity_limit_mw": grid_limit,
                    "grid_telemetry": grid_telemetry.model_dump(),
                    "center_telemetry": center_telemetry.model_dump(),
                    "decision": decision.model_dump(),
                    "simulation_mode": "legacy_synthetic_operator_approvals_simulated",
                }
            )

        return timeline

    def generate_real_24h_scenario(self, scenario_name, custom=None, start_step=48, simulate_approvals=False):
        """Real CSV timeline with explicit scenario interventions; default is planning only.

        custom: optional {grid: partial telemetry, data_centers: CSV-shaped list,
        confidence_override: float}. Overrides are validated by the agent, never evaluated as code.
        """
        if scenario_name not in SCENARIO_DESCRIPTIONS:
            raise ValueError("Unknown scenario")
        if not isinstance(start_step, int) or start_step < 24 or start_step+23 >= 720:
            raise ValueError("Scenario requires 24 available hours after lag warm-up")
        custom = custom or {}
        timeline = []
        for step in range(start_step, start_step+24):
            status = load_scenario_status(scenario_name, step, custom)
            grid, centers = status["grid"], status["data_centers"]
            hour = grid["hour"]
            decision = self.agent.run_cycle(grid, data_centers=centers,
                                           confidence_override=custom.get("confidence_override"))
            if simulate_approvals and decision.status == "OFFER_GENERATED":
                for offer in decision.dc_offers:
                    if offer.status == "OFFER_GENERATED":
                        result = self.agent.act(decision, dispatch_action=DispatchAction(
                            decision_id=decision.decision_id, center_id=offer.dc_id, action="APPROVED"))
                        decision = AgentDecision.model_validate(result["decision"])
            if simulate_approvals and decision.confidence_score >= .75 and decision.fallback and (
                decision.fallback.total_fallback_absorbed_mw+decision.fallback.curtailment_mw > 0
            ):
                result = self.agent.act(decision, dispatch_action=DispatchAction(
                    decision_id=decision.decision_id, center_id="PPC_FALLBACK", action="APPROVED"))
                decision = AgentDecision.model_validate(result["decision"])
            timeline.append(dict(step=step, hour=hour, hour_label=f"{hour:02d}:00",
                timestamp=grid["timestamp"], grid_telemetry=grid, data_centers=centers,
                net_solar_generation_mw=grid["current_surplus_mw"], grid_capacity_limit_mw=grid["grid_capacity_limit_mw"],
                decision=decision.model_dump(), simulation_mode="real_csv_scenario_intervention",
                operator_approvals_simulated=simulate_approvals))
        return timeline

    def calculate_impact(
        self, decisions: List[AgentDecision | Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculează KPI-urile de business și sustenabilitate pe baza listei de decizii:
        - MWh salvați de la curtailment (absorbiți de centrele universitare)
        - Venit suplimentar generat pentru PPC (la 30 EUR/MWh în loc de risipă / preț 0)
        - Economii financiare pentru centrele de cercetare (față de tariful standard 115 EUR/MWh)
        - Tone de CO2 evitate prin rularea calculelor AI pe energie 100% solară din surplus
        """
        parsed_decisions: List[AgentDecision] = [
            d if isinstance(d, AgentDecision) else AgentDecision.model_validate(d)
            for d in decisions
        ]

        total_surplus_mwh = sum(d.surplus_predicted_mw for d in parsed_decisions)
        total_excess = sum(d.grid_excess_mw for d in parsed_decisions)
        mwh_saved = sum(min(max(0., d.grid_excess_mw-d.executed_curtailment_mw), d.executed_absorbed_mw) for d in parsed_decisions)
        planned_mwh = sum(d.allocated_mw for d in parsed_decisions)
        unabsorbed_curtailment_mwh = sum(max(0., d.grid_excess_mw-d.executed_absorbed_mw-d.executed_curtailment_mw) for d in parsed_decisions)

        accepted = [o for d in parsed_decisions for o in d.dc_offers if o.status == "ACCEPTED"]
        ppc_revenue_eur = sum(o.offered_mw*o.offered_price_eur for o in accepted)
        researcher_savings_eur = sum(o.offered_mw*(o.contract_price_eur-o.offered_price_eur) for o in accepted)
        # Stored energy is not credited as avoided emissions before discharge.
        co2_avoided_tons = sum(d.executed_mw+d.executed_ppc_blue_ev_mw for d in parsed_decisions) * self.CO2_TONS_PER_MWH

        offers_count = sum(1 for d in parsed_decisions if d.status == "OFFER_GENERATED")
        escalations_count = sum(
            1 for d in parsed_decisions if d.status == "ESCALATE_TO_HUMAN"
        )

        return {
            "execution_mode": "simulated",
            "predicted_export_mwh": round(sum(d.predicted_export_mw for d in parsed_decisions), 4),
            "demo_injected_surplus_mwh": round(sum(d.demo_injected_surplus_mw for d in parsed_decisions), 4),
            "grid_excess_mwh": round(total_excess, 4),
            "executed_absorbed_mwh": round(sum(d.executed_absorbed_mw for d in parsed_decisions), 4),
            "executed_curtailment_mwh": round(sum(d.executed_curtailment_mw for d in parsed_decisions), 4),
            "remaining_grid_excess_mwh": round(unabsorbed_curtailment_mwh, 4),
            "curtailment_kpi_semantics": "excess_above_grid_limit_only; executed_curtailment_is_not_energy_saved",
            "planned_mwh": round(planned_mwh, 4),
            "planned_revenue_eur": round(sum(o.offered_mw*o.offered_price_eur for d in parsed_decisions for o in d.dc_offers), 2),
            "mwh_saved_from_curtailment": round(mwh_saved, 2),
            "total_surplus_mwh": round(total_surplus_mwh, 2),
            "unabsorbed_curtailment_mwh": round(unabsorbed_curtailment_mwh, 2),
            "ppc_revenue_eur": round(ppc_revenue_eur, 2),
            "researcher_savings_eur": round(researcher_savings_eur, 2),
            "co2_avoided_tons": round(co2_avoided_tons, 2),
            "offers_generated_hours": offers_count,
            "human_escalations_hours": escalations_count,
        }


def generate_24h_scenario(scenario_name: str) -> List[Dict[str, Any]]:
    """Helper la nivel de modul pentru apel rapid din server sau app."""
    return VoltComputeSimulator().generate_24h_scenario(scenario_name)


def calculate_impact(decisions: List[AgentDecision | Dict[str, Any]]) -> Dict[str, Any]:
    """Helper la nivel de modul pentru calculul KPI-urilor de impact."""
    return VoltComputeSimulator().calculate_impact(decisions)

