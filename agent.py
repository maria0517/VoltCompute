import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from schemas import (
    AgentDecision,
    DispatchAction,
    GridTelemetry,
    ResearchCenterTelemetry,
)

DEFAULT_MODEL_PATH = Path("models/model_surplus.pkl")


class VoltComputeAgent:
    """
    Agentul autonom decizional VoltCompute care arbitrează în timp real surplusul solar PPC
    și îl redirecționează către centre de calcul academice (baterie virtuală).

    Ciclul agentului:
    1. perceive() -> Validare strictă a telemetriei cu Pydantic v2.
    2. estimate() -> Predicție surplus (MW), scor de confidență și explicații SHAP/atribuire.
    3. decide()   -> Aplicare reguli de siguranță (confidență, protecție termică servere, prag 2.0 MW).
    4. act()      -> Formatare payload dispecerizare pentru integrarea cu SCADA / HPC.
    """

    CONFIDENCE_THRESHOLD: float = 0.75
    THERMAL_LIMIT_CELSIUS: float = 25.0
    MIN_SURPLUS_THRESHOLD_MW: float = 2.0
    FIXED_GREEN_PRICE_EUR: float = 30.0

    def __init__(self, model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.model_artifact: Optional[Dict[str, Any]] = None
        self._load_model()

    def _load_model(self) -> None:
        """Încarcă modelul XGBoost din models/model_surplus.pkl dacă este disponibil."""
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, dict) and "model" in loaded:
                    self.model_artifact = loaded
                else:
                    self.model_artifact = {"model": loaded}
            except Exception:
                self.model_artifact = None

    def perceive(
        self,
        grid_input: Union[GridTelemetry, Dict[str, Any]],
        center_input: Union[ResearchCenterTelemetry, Dict[str, Any]],
    ) -> Tuple[
        Optional[GridTelemetry],
        Optional[ResearchCenterTelemetry],
        Optional[AgentDecision],
    ]:
        """
        Pasul 1: PERCEIVE
        Validează telemetria primită prin contractele Pydantic v2.
        Returnează o decizie cu status="DATA_ERROR" dacă datele sunt corupte sau în afara limitelor fizice.
        """
        try:
            grid = (
                grid_input
                if isinstance(grid_input, GridTelemetry)
                else GridTelemetry.model_validate(grid_input)
            )
            center = (
                center_input
                if isinstance(center_input, ResearchCenterTelemetry)
                else ResearchCenterTelemetry.model_validate(center_input)
            )
            return grid, center, None
        except (ValidationError, TypeError, ValueError) as err:
            error_decision = AgentDecision(
                surplus_predicted_mw=0.0,
                confidence_score=0.0,
                status="DATA_ERROR",
                allocated_mw=0.0,
                fixed_price_eur=self.FIXED_GREEN_PRICE_EUR,
                reasoning=f"Eroare de validare telemetrie (senzori corupți sau valori invalide): {str(err)[:220]}",
                top_shap_features={},
            )
            return None, None, error_decision

    def estimate(
        self,
        grid: GridTelemetry,
        confidence_override: Optional[float] = None,
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Pasul 2: ESTIMATE
        Calculează surplusul estimat (MW), scorul de confidență (0.0-1.0)
        și top contribuții de tip SHAP pentru explicabilitate în fața dispecerului.
        Include fallback fizic determinist dacă modelul .pkl nu este încă antrenat.
        """
        # Estimare fizică de referință (folosită atât ca fallback, cât și pentru verificarea coerenței)
        temp_derating = 1.0 - max(0.0, (grid.ambient_temp_celsius - 25.0) * 0.004)
        gross_solar_mw = (grid.solar_irradiance_wm2 / 1000.0) * 17.8 * temp_derating
        local_load_mw = 1.4 + 0.5 * float(np.sin(np.pi * grid.hour / 12.0))
        net_injection_mw = max(0.0, gross_solar_mw - local_load_mw)
        physics_surplus_mw = max(0.0, net_injection_mw - grid.grid_capacity_limit_mw)

        predicted_surplus_mw = physics_surplus_mw

        if self.model_artifact is not None and "model" in self.model_artifact:
            try:
                model = self.model_artifact["model"]
                features_df = pd.DataFrame(
                    [
                        {
                            "hour": grid.hour,
                            "solar_irradiance_wm2": grid.solar_irradiance_wm2,
                            "ambient_temp_celsius": grid.ambient_temp_celsius,
                            "day_ahead_price_eur": grid.day_ahead_price_eur,
                            "grid_capacity_limit_mw": grid.grid_capacity_limit_mw,
                        }
                    ]
                )
                raw_pred = float(model.predict(features_df)[0])
                # Combinăm predicția ML cu ancora fizică pentru stabilitate pe scenarii extreme
                predicted_surplus_mw = max(0.0, 0.7 * raw_pred + 0.3 * physics_surplus_mw)
            except Exception:
                predicted_surplus_mw = physics_surplus_mw

        # Calcul scor de confidență
        if confidence_override is not None:
            confidence = float(np.clip(confidence_override, 0.0, 1.0))
        elif grid.weather_stability_index is not None:
            confidence = float(np.clip(grid.weather_stability_index, 0.0, 1.0))
        else:
            # Scor de confidență bazat pe coerența dintre iradianța solară, oră și prețul PZU
            base_conf = 0.92
            # Dacă e miezul zilei, dar iradianța e moderată și prețul PZU e foarte volatil -> nebulozitate variabilă
            if 10 <= grid.hour <= 16 and 300 <= grid.solar_irradiance_wm2 <= 620 and grid.day_ahead_price_eur > 95:
                base_conf = 0.64
            elif grid.ambient_temp_celsius > 42.0 or grid.solar_irradiance_wm2 > 1200.0:
                base_conf = 0.70
            confidence = float(np.clip(base_conf, 0.0, 1.0))

        # Calcul atribuire locală explicabilă (stil SHAP values în MW)
        irradiance_contrib = round((grid.solar_irradiance_wm2 - 500.0) / 150.0, 3)
        grid_limit_contrib = round((10.0 - grid.grid_capacity_limit_mw) * 0.85, 3)
        price_signal_contrib = round((50.0 - grid.day_ahead_price_eur) / 45.0, 3)
        temp_contrib = round((25.0 - grid.ambient_temp_celsius) * 0.05, 3)

        top_shap = {
            "solar_irradiance_wm2": irradiance_contrib,
            "day_ahead_price_eur": price_signal_contrib,
            "grid_capacity_limit_mw": grid_limit_contrib,
            "ambient_temp_celsius": temp_contrib,
        }

        return round(predicted_surplus_mw, 3), round(confidence, 4), top_shap

    def decide(
        self,
        grid: GridTelemetry,
        center: ResearchCenterTelemetry,
        surplus_predicted_mw: float,
        confidence_score: float,
        top_shap_features: Dict[str, float],
    ) -> AgentDecision:
        """
        Pasul 3: DECIDE
        Aplică constrângerile stricte de siguranță operațională:
        1. Dacă confidence_score < 0.75 -> ESCALATE_TO_HUMAN (fallback la dispecer).
        2. Dacă cooling_temp_celsius > 25.0 -> NORMAL_GRID_OPERATION (protecție termică servere).
        3. Dacă surplus_predicted_mw > 2.0 -> OFFER_GENERATED la 30 EUR/MWh pentru min(surplus, max_absorb).
        4. Altfel -> NORMAL_GRID_OPERATION.
        """
        # Regula 1: Pragul de confidență (Safety-First AI)
        if confidence_score < self.CONFIDENCE_THRESHOLD:
            return AgentDecision(
                surplus_predicted_mw=surplus_predicted_mw,
                confidence_score=confidence_score,
                status="ESCALATE_TO_HUMAN",
                allocated_mw=0.0,
                fixed_price_eur=self.FIXED_GREEN_PRICE_EUR,
                reasoning=(
                    f"Confidența modelului ({confidence_score:.2f}) este sub pragul minim de siguranță "
                    f"({self.CONFIDENCE_THRESHOLD:.2f}) din cauza instabilității meteo. "
                    f"Decizia de alocare a {surplus_predicted_mw:.2f} MW este escaladată către dispecerul uman PPC."
                ),
                top_shap_features=top_shap_features,
            )

        # Regula 2: Protecția termică a centrului de date universitar
        if center.cooling_temp_celsius > self.THERMAL_LIMIT_CELSIUS:
            return AgentDecision(
                surplus_predicted_mw=surplus_predicted_mw,
                confidence_score=confidence_score,
                status="NORMAL_GRID_OPERATION",
                allocated_mw=0.0,
                fixed_price_eur=self.FIXED_GREEN_PRICE_EUR,
                reasoning=(
                    f"Protecție termică activată pentru {center.center_id}: temperatura de răcire "
                    f"({center.cooling_temp_celsius:.1f}°C) depășește limita de siguranță ({self.THERMAL_LIMIT_CELSIUS:.1f}°C). "
                    f"Transferul energetic a fost blocat pentru a preveni supraîncălzirea serverelor."
                ),
                top_shap_features=top_shap_features,
            )

        # Regula 3: Arbitraj activ al surplusului solar (> 2.0 MW)
        if surplus_predicted_mw > self.MIN_SURPLUS_THRESHOLD_MW and center.max_absorb_capacity_mw > 0:
            allocated_mw = min(surplus_predicted_mw, center.max_absorb_capacity_mw)
            return AgentDecision(
                surplus_predicted_mw=surplus_predicted_mw,
                confidence_score=confidence_score,
                status="OFFER_GENERATED",
                allocated_mw=allocated_mw,
                fixed_price_eur=self.FIXED_GREEN_PRICE_EUR,
                reasoning=(
                    f"Surplus solar critic detectat ({surplus_predicted_mw:.2f} MW peste limita de {grid.grid_capacity_limit_mw:.1f} MW). "
                    f"S-a generat oferta automată de {allocated_mw:.2f} MW la tarif verde fix {self.FIXED_GREEN_PRICE_EUR:.0f} EUR/MWh "
                    f"către {center.center_id} ({center.pending_compute_jobs} joburi AI/HPC în așteptare, răcire optimă la {center.cooling_temp_celsius:.1f}°C)."
                ),
                top_shap_features=top_shap_features,
            )

        # Regula 4: Funcționare normală în parametri (fără risc de congestie / curtailment)
        return AgentDecision(
            surplus_predicted_mw=surplus_predicted_mw,
            confidence_score=confidence_score,
            status="NORMAL_GRID_OPERATION",
            allocated_mw=0.0,
            fixed_price_eur=self.FIXED_GREEN_PRICE_EUR,
            reasoning=(
                f"Rețeaua PPC funcționează în parametri normali (surplus estimat: {surplus_predicted_mw:.2f} MW, "
                f"sub pragul de activare de {self.MIN_SURPLUS_THRESHOLD_MW:.1f} MW). Nu este necesară redirecționarea."
            ),
            top_shap_features=top_shap_features,
        )

    def act(
        self,
        decision: AgentDecision,
        center: Optional[ResearchCenterTelemetry] = None,
        dispatch_action: Optional[DispatchAction] = None,
    ) -> Dict[str, Any]:
        """
        Pasul 4: ACT
        Formatează payload-ul final pentru dispecerizare (SCADA PPC + orchestrator Slurm/Kubernetes la centrul HPC).
        """
        target_center = center.center_id if center else (dispatch_action.center_id if dispatch_action else "UNKNOWN")

        if dispatch_action is not None:
            execution_state = (
                "DISPATCH_EXECUTED" if dispatch_action.action == "APPROVED" else "DISPATCH_ABORTED_BY_OPERATOR"
            )
            effective_mw = decision.allocated_mw if dispatch_action.action == "APPROVED" else 0.0
        else:
            if decision.status == "OFFER_GENERATED":
                execution_state = "READY_FOR_AUTO_DISPATCH"
                effective_mw = decision.allocated_mw
            elif decision.status == "ESCALATE_TO_HUMAN":
                execution_state = "AWAITING_HUMAN_OPERATOR"
                effective_mw = 0.0
            else:
                execution_state = "NO_ACTION_REQUIRED"
                effective_mw = 0.0

        return {
            "decision_id": decision.decision_id,
            "target_center_id": target_center,
            "status": decision.status,
            "execution_state": execution_state,
            "effective_mw": round(effective_mw, 3),
            "tariff_eur_mwh": decision.fixed_price_eur,
            "estimated_hourly_revenue_eur": round(effective_mw * decision.fixed_price_eur, 2),
            "compute_trigger": effective_mw > 0.0,
            "reasoning": decision.reasoning,
        }

    def run_cycle(
        self,
        grid_input: Union[GridTelemetry, Dict[str, Any]],
        center_input: Union[ResearchCenterTelemetry, Dict[str, Any]],
        confidence_override: Optional[float] = None,
    ) -> AgentDecision:
        """Execută ciclul complet perceive -> estimate -> decide și returnează AgentDecision."""
        grid, center, error_decision = self.perceive(grid_input, center_input)
        if error_decision is not None or grid is None or center is None:
            return error_decision  # type: ignore[return-value]

        surplus_mw, confidence, top_shap = self.estimate(
            grid, confidence_override=confidence_override
        )
        return self.decide(grid, center, surplus_mw, confidence, top_shap)

