from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal

import numpy as np

from agent import VoltComputeAgent
from schemas import AgentDecision, GridTelemetry, ResearchCenterTelemetry

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
            raise ValueError(
                f"Scenariu necunoscut '{scenario_name}'. Valori suportate: {list(SCENARIO_DESCRIPTIONS.keys())}"
            )

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
                if 11 <= hour <= 15:
                    # Fluctuații severe de iradianță + volatilitate preț -> confidență < 0.75
                    irr = float(sunny_irradiance[hour] * 0.92)
                    da_price = 110.0
                    cooling_temp = 20.5
                    max_absorb_mw = 6.0
                    pending_jobs = 22
                    stability_idx = 0.58  # Sub pragul de 0.75 -> ESCALATE_TO_HUMAN

            elif scenario_name == "DATACENTER_THERMAL_RISK":
                if 11 <= hour <= 15:
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
                }
            )

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
        mwh_saved = sum(
            d.allocated_mw for d in parsed_decisions if d.status == "OFFER_GENERATED"
        )
        unabsorbed_curtailment_mwh = max(0.0, total_surplus_mwh - mwh_saved)

        ppc_revenue_eur = mwh_saved * self.FIXED_GREEN_PRICE_EUR
        researcher_savings_eur = mwh_saved * (
            self.STANDARD_RETAIL_PRICE_EUR - self.FIXED_GREEN_PRICE_EUR
        )
        co2_avoided_tons = mwh_saved * self.CO2_TONS_PER_MWH

        offers_count = sum(1 for d in parsed_decisions if d.status == "OFFER_GENERATED")
        escalations_count = sum(
            1 for d in parsed_decisions if d.status == "ESCALATE_TO_HUMAN"
        )

        return {
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

