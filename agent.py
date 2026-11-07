"""Deterministic safety and HITL state machine around the t+2h ML model."""
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional

import joblib
import numpy as np

from schemas import (AgentDecision, DataCenterOffer, DataCenterTelemetry, DispatchAction,
                     FallbackAllocation, GridTelemetry, ResearchCenterTelemetry)
from train_pipeline import MODEL_OUTPUT_PATH, infer_single_step, infer_forecast_window, load_grid_status

DEFAULT_MODEL_PATH = MODEL_OUTPUT_PATH


class VoltComputeAgent:
    CONFIDENCE_THRESHOLD = .75
    THERMAL_LIMIT_CELSIUS = 25.0
    MIN_SURPLUS_THRESHOLD_MW = 2.0
    FIXED_GREEN_PRICE_EUR = 30.0

    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.model_artifact = None
        self._decisions = {}
        self._legacy = set()
        self._lock = RLock()
        self._last_prediction = {}
        self._load_model()

    def _load_model(self):
        try:
            loaded = joblib.load(self.model_path)
            if loaded.get("schema_version") == 2 and loaded.get("feature_names"):
                self.model_artifact = loaded
        except (OSError, ValueError, TypeError, KeyError, EOFError):
            self.model_artifact = None

    @staticmethod
    def _error():
        return AgentDecision(surplus_predicted_mw=0, confidence_score=0, allocated_mw=0,
                             status="DATA_ERROR", reasoning="Telemetrie invalidă sau model indisponibil; alocare blocată.")

    def _remember(self, decision):
        self._decisions[decision.decision_id] = decision.model_copy(deep=True)
        return decision

    def perceive(self, grid_input, center_input=None):
        try:
            grid = GridTelemetry.model_validate(grid_input)
            center = ResearchCenterTelemetry.model_validate(center_input) if center_input is not None else None
            return grid, center, None
        except (ValueError, TypeError):
            return None, None, self._error()

    def estimate(self, grid, confidence_override=None):
        grid = GridTelemetry.model_validate(grid)
        if confidence_override is not None and (not np.isfinite(confidence_override) or not 0 <= confidence_override <= 1):
            raise ValueError("Invalid confidence override")
        if grid.data_source != "legacy" or grid.model_features is not None or grid.current_surplus_mw is not None:
            if self.model_artifact is None:
                raise ValueError("Canonical ML artifact unavailable")
            if grid.model_features is None:
                raise ValueError("Real-data inference requires causal feature history")
            features = dict(grid.model_features)
            # Explicit telemetry edits override their equivalent current model features.
            features.update(hour=grid.hour, hour_sin=float(np.sin(2*np.pi*grid.hour/24)),
                            hour_cos=float(np.cos(2*np.pi*grid.hour/24)),
                            radiatie_solara_wm2=grid.solar_irradiance_wm2,
                            temp_c=grid.ambient_temp_celsius, pret_pzu_eur_mwh=grid.day_ahead_price_eur,
                            viteza_vant_kmh=grid.wind_speed_kmh, avg_battery_soc_pct=grid.prosumer_battery_soc_pct,
                            Surplus_National_MW=grid.current_surplus_mw,
                            total_prosumer_load_mw=grid.total_prosumer_load_mw, is_weekend=grid.is_weekend)
            prediction = infer_single_step(features, self.model_artifact)
            prediction["forecast_window"] = infer_forecast_window(features, self.model_artifact)
            prediction["predicted_export_mw"] = prediction["surplus_predicted_mw"]
            prediction["demo_injected_surplus_mw"] = grid.demo_injected_surplus_mw
            prediction["surplus_predicted_mw"] += grid.demo_injected_surplus_mw
            prediction["grid_excess_mw"] = max(0., prediction["surplus_predicted_mw"]-grid.grid_capacity_limit_mw)
            prediction["surplus_semantics"] = "aggregate_export_plus_explicit_demo_injection"
        else:
            # Legacy synthetic 17.8MW park is a different population from the 100 prosumers.
            # Never pass its inputs to the small-scale CSV model or fabricate SHAP values.
            temp = grid.ambient_temp_celsius if grid.ambient_temp_celsius is not None else 22
            gross = grid.solar_irradiance_wm2 / 1000 * 17.8 * (1-max(0, temp-25)*.004)
            load = 1.4 + .5 * np.sin(np.pi*grid.hour/12)
            surplus = max(0., float(gross-load-grid.grid_capacity_limit_mw))
            prediction = dict(surplus_predicted_mw=surplus, confidence_score=.70,
                              predicted_export_mw=max(0., float(gross-load)), grid_excess_mw=surplus,
                              surplus_semantics="legacy_physical_excess_above_grid_limit",
                              top_shap_features={}, prediction_source="legacy_conservative_physical_fallback",
                              shap_method="unavailable_physical_fallback")
        confidence = prediction["confidence_score"]
        if confidence_override is not None:
            confidence = (float(confidence_override) if prediction["prediction_source"] == "legacy_conservative_physical_fallback"
                          else min(confidence, float(confidence_override)))
        if grid.weather_stability_index is not None:
            confidence = min(confidence, grid.weather_stability_index)
        prediction["confidence_score"] = confidence
        self._last_prediction = prediction
        return prediction["surplus_predicted_mw"], confidence, prediction["top_shap_features"]

    @staticmethod
    def _fallback(remaining):
        remaining = max(0., remaining)
        ev = min(3.5, remaining)
        bess = min(4., max(0., remaining-ev))
        curtail = max(0., remaining-ev-bess)
        tier = "TIER_3_CURTAILMENT" if curtail else "TIER_2_BESS_STORAGE" if bess else "TIER_1_PPC_BLUE_EV" if ev else "NONE"
        return FallbackAllocation(ppc_blue_ev_mw=ev, bess_storage_mw=bess, curtailment_mw=curtail,
                                  total_fallback_absorbed_mw=ev+bess, active_tier=tier,
                                  explanation="Plan de rerutare PPC Blue → BESS → SCADA; neexecutat, necesită aprobare umană.")

    def allocate_and_decide_multi_dc(self, grid, data_centers, surplus_predicted_mw,
                                    confidence_score, top_shap_features=None):
        """Allocate CSV-shaped centers by capacity*contract; returns a registered planned decision."""
        grid = GridTelemetry.model_validate(grid)
        centers = [DataCenterTelemetry.model_validate(dc) for dc in data_centers]
        if len({dc.dc_id for dc in centers}) != len(centers):
            raise ValueError("Duplicate center IDs")
        if not np.isfinite([surplus_predicted_mw, confidence_score]).all() or surplus_predicted_mw < 0 or not 0 <= confidence_score <= 1:
            raise ValueError("Invalid estimate")
        eligible_confidence = confidence_score >= self.CONFIDENCE_THRESHOLD
        remaining = float(surplus_predicted_mw)
        offers = []
        for dc in sorted(centers, key=lambda d: (-d.capacitate_libera_mw*d.pret_contract_eur_mwh, d.dc_id)):
            hot = dc.temp_servere_c >= dc.limita_termica_c-1.5
            mw = min(remaining, dc.capacitate_libera_mw) if eligible_confidence and not hot else 0.
            status = ("REJECTED_THERMAL_SAFETY" if hot else "ESCALATE_TO_HUMAN" if not eligible_confidence
                      else "OFFER_GENERATED" if mw > 0 else "STANDBY")
            offers.append(DataCenterOffer(dc_id=dc.dc_id, dc_name=dc.dc_name or dc.dc_id,
                available_capacity_mw=dc.capacitate_libera_mw, current_temp_c=dc.temp_servere_c,
                thermal_limit_c=dc.limita_termica_c, contract_price_eur=dc.pret_contract_eur_mwh,
                offered_mw=mw, offered_price_eur=30, discount_pct=self._discount(30, dc.pret_contract_eur_mwh),
                priority_score=dc.capacitate_libera_mw*dc.pret_contract_eur_mwh, status=status))
            remaining = max(0., remaining-mw)
        allocated = sum(o.offered_mw for o in offers)
        decision = AgentDecision(surplus_predicted_mw=surplus_predicted_mw, confidence_score=confidence_score,
            predicted_export_mw=surplus_predicted_mw, grid_excess_mw=max(0., surplus_predicted_mw-grid.grid_capacity_limit_mw),
            allocated_mw=allocated, status="ESCALATE_TO_HUMAN" if not eligible_confidence else "OFFER_GENERATED" if allocated else "NORMAL_GRID_OPERATION",
            reasoning="Alocare constrânsă t+2h; protecție termică strictă și aprobare umană obligatorie.",
            top_shap_features=top_shap_features or {}, dc_offers=offers,
            fallback=self._fallback(remaining) if eligible_confidence else self._fallback(0))
        return self._remember(decision)

    @staticmethod
    def _discount(price, contract):
        return 100*(1-price/contract) if contract else 0.

    def decide(self, grid, center, surplus_predicted_mw, confidence_score, top_shap_features):
        hot = center.cooling_temp_celsius > 25
        uncertain = confidence_score < .75
        mw = min(surplus_predicted_mw, center.max_absorb_capacity_mw) if not hot and not uncertain and surplus_predicted_mw > 2 else 0.
        offer = DataCenterOffer(dc_id=center.center_id, dc_name=center.center_id,
            available_capacity_mw=center.max_absorb_capacity_mw, current_temp_c=center.cooling_temp_celsius,
            thermal_limit_c=26.5, contract_price_eur=115, offered_mw=mw, offered_price_eur=30,
            discount_pct=self._discount(30,115), priority_score=center.max_absorb_capacity_mw*115,
            status="REJECTED_THERMAL_SAFETY" if hot else "ESCALATE_TO_HUMAN" if uncertain else "OFFER_GENERATED" if mw else "STANDBY")
        decision = AgentDecision(surplus_predicted_mw=surplus_predicted_mw, confidence_score=confidence_score,
            predicted_export_mw=surplus_predicted_mw+grid.grid_capacity_limit_mw, grid_excess_mw=surplus_predicted_mw,
            surplus_semantics="legacy_physical_excess_above_grid_limit",
            allocated_mw=mw, status="ESCALATE_TO_HUMAN" if uncertain else "OFFER_GENERATED" if mw else "NORMAL_GRID_OPERATION",
            reasoning=f"Profil legacy; temperatură {center.cooling_temp_celsius:.1f}°C; " +
                      ("protecție termică activă." if hot else "aprobare umană obligatorie."),
            top_shap_features=top_shap_features, dc_offers=[offer], fallback=self._fallback(0))
        self._legacy.add(decision.decision_id)
        return self._remember(decision)

    def run_cycle(self, grid_input, center_input=None, confidence_override=None, data_centers=None):
        with self._lock:
            grid, center, error = self.perceive(grid_input, center_input)
            if error is not None:
                return self._remember(error)
            try:
                surplus, confidence, shap = self.estimate(grid, confidence_override)
                if data_centers is not None:
                    decision = self.allocate_and_decide_multi_dc(grid, data_centers, surplus, confidence, shap)
                elif center is not None:
                    decision = self.decide(grid, center, surplus, confidence, shap)
                else:
                    raise ValueError("Center telemetry required")
                for field in ("prediction_source", "prediction_lower_mw", "prediction_upper_mw", "shap_method",
                              "shap_values", "shap_base_value", "raw_prediction_mw", "shap_percentages",
                              "predicted_export_mw", "grid_excess_mw", "demo_injected_surplus_mw", "surplus_semantics", "forecast_window",
                              "weather_estimate_method", "weather_estimate_is_issued_forecast"):
                    if field in self._last_prediction:
                        setattr(decision, field, self._last_prediction[field])
                if decision.prediction_source == "legacy_conservative_physical_fallback":
                    decision.data_quality_warnings.append("Explicit legacy physical fallback; no ML/SHAP or calibrated interval for this synthetic park.")
                return self._remember(decision)
            except (ValueError, TypeError, KeyError, OSError):
                return self._remember(self._error())

    def act(self, decision, center=None, dispatch_action=None):
        """Atomic in-memory demo transaction. All M2M payloads are simulated, never sent."""
        with self._lock:
            stored = self._decisions.get(decision.decision_id)
            if stored is None:
                raise ValueError("Unknown decision ID")
            updated = stored.model_copy(deep=True)
            effective = 0.
            revenue = savings = 0.
            target = center.center_id if center else "UNKNOWN"
            state = "AWAITING_HUMAN_OPERATOR"
            if dispatch_action is not None:
                action = DispatchAction.model_validate(dispatch_action)
                if action.decision_id != updated.decision_id:
                    raise ValueError("Decision ID mismatch")
                target = action.center_id
                if updated.status in ("DATA_ERROR", "ESCALATE_TO_HUMAN") or updated.confidence_score < .75:
                    raise ValueError("Unsafe decision cannot be dispatched")
                if target == "PPC_FALLBACK":
                    return self._approve_fallback(updated, action)
                offer = next((o for o in updated.dc_offers if o.dc_id == target), None)
                if offer is None:
                    raise ValueError("Unknown center ID")
                hot = (offer.current_temp_c > 25 if updated.decision_id in self._legacy
                       else offer.current_temp_c >= offer.thermal_limit_c-1.5)
                if hot or offer.status == "REJECTED_THERMAL_SAFETY":
                    raise ValueError("Thermal safety blocks action")
                if offer.status == "ACCEPTED":
                    if action.action != "APPROVED":
                        raise ValueError("Accepted offers are immutable")
                    state = "ALREADY_SIMULATED"
                elif offer.status != "OFFER_GENERATED" or offer.offered_mw <= 0:
                    raise ValueError("No active offer for this center")
                elif action.action == "APPROVED":
                    offer.status = "ACCEPTED"
                    effective = offer.offered_mw
                    revenue = effective*offer.offered_price_eur
                    savings = effective*(offer.contract_price_eur-offer.offered_price_eur)
                    updated.executed_mw += effective
                    updated.executed_absorbed_mw += effective
                    updated.execution_mode = "simulated"
                    state = "DISPATCH_SIMULATED"
                elif action.action == "DUTCH_AUCTION_STEP":
                    offer.offered_price_eur = max(10., offer.offered_price_eur-10*action.auction_step)
                    offer.auction_round += action.auction_step
                    offer.discount_pct = self._discount(offer.offered_price_eur, offer.contract_price_eur)
                    state = "DUTCH_AUCTION_UPDATED"
                else:
                    quota = offer.offered_mw
                    price = max(10., offer.offered_price_eur-10*action.auction_step)
                    offer.offered_mw = 0
                    offer.status = "REJECTED_BY_OPERATOR"
                    offer.auction_round += action.auction_step
                    offer.offered_price_eur = price
                    offer.discount_pct = self._discount(price, offer.contract_price_eur)
                    if action.trigger_fallback:
                        # Unaccepted eligible offers can absorb spare capacity. Accepted
                        # contracts retain their exact volume and price permanently.
                        for other in sorted(updated.dc_offers, key=lambda o: (-o.priority_score, o.dc_id)):
                            if other.status not in ("STANDBY", "OFFER_GENERATED") or other.current_temp_c >= other.thermal_limit_c-1.5:
                                continue
                            amount = min(quota, max(0., other.available_capacity_mw-other.offered_mw))
                            if amount <= 0:
                                continue
                            other.offered_mw += amount
                            other.offered_price_eur = price
                            other.discount_pct = self._discount(price, other.contract_price_eur)
                            other.auction_round = offer.auction_round
                            other.status = "OFFER_GENERATED"
                            quota = max(0., quota-amount)
                        updated.allocated_mw = sum(o.offered_mw for o in updated.dc_offers)
                        next_fallback = self._fallback(max(0., updated.surplus_predicted_mw-updated.allocated_mw))
                        if updated.fallback is None or any(getattr(next_fallback, key) != getattr(updated.fallback, key)
                            for key in ("ppc_blue_ev_mw", "bess_storage_mw", "curtailment_mw")):
                            updated.fallback_plan_revision += 1
                        updated.fallback = next_fallback
                    else:
                        updated.allocated_mw = sum(o.offered_mw for o in updated.dc_offers)
                    state = "DISPATCH_ABORTED_BY_OPERATOR"
                self._remember(updated)
            return self._execution_result(updated, target, state, effective, revenue, savings)

    def _approve_fallback(self, updated, action):
        """Approve only the unexecuted reservation; stale retries execute zero extra MW."""
        if action.action != "APPROVED":
            raise ValueError("Fallback reservations support explicit approval only")
        # A retry of an earlier approval must not approve a newly enlarged plan.
        # After the first approval, operators identify the displayed plan revision.
        revision = action.fallback_revision
        if revision is None and updated.fallback_approved_revision:
            return self._execution_result(updated, "PPC_FALLBACK", "ALREADY_SIMULATED")
        if revision is not None and revision <= updated.fallback_approved_revision:
            return self._execution_result(updated, "PPC_FALLBACK", "ALREADY_SIMULATED")
        if revision is not None and revision != updated.fallback_plan_revision:
            raise ValueError("Stale or unknown fallback plan revision")
        fb = updated.fallback
        if fb is None or fb.ppc_blue_ev_mw+fb.bess_storage_mw+fb.curtailment_mw <= 0:
            raise ValueError("No fallback reservation to approve")
        ev = max(0., fb.ppc_blue_ev_mw-updated.executed_ppc_blue_ev_mw)
        bess = max(0., fb.bess_storage_mw-updated.executed_bess_storage_mw)
        curtail = max(0., fb.curtailment_mw-updated.executed_curtailment_mw)
        updated.executed_ppc_blue_ev_mw += ev
        updated.executed_bess_storage_mw += bess
        updated.executed_curtailment_mw += curtail
        updated.fallback_approved_revision = updated.fallback_plan_revision
        updated.executed_absorbed_mw = updated.executed_mw+updated.executed_ppc_blue_ev_mw+updated.executed_bess_storage_mw
        updated.execution_mode = "simulated"
        updated.fallback.explanation = "Rezervare cumulativă; execuțiile aprobate sunt simulate și nu se realocă."
        self._remember(updated)
        state = "FALLBACK_DISPATCH_SIMULATED" if ev+bess+curtail > 1e-12 else "ALREADY_SIMULATED"
        return self._execution_result(updated, "PPC_FALLBACK", state, ev=ev, bess=bess, curtail=curtail)

    @staticmethod
    def _execution_result(updated, target, state, effective=0., revenue=0., savings=0., ev=0., bess=0., curtail=0.):
        accepted = [o for o in updated.dc_offers if o.status == "ACCEPTED"]
        tariff = next((o.offered_price_eur for o in updated.dc_offers if o.dc_id == target),
                      None if target == "PPC_FALLBACK" else updated.fixed_price_eur)
        return dict(decision_id=updated.decision_id, target_center_id=target, status=updated.status,
                execution_state=state, execution_mode=updated.execution_mode, simulated=True,
                effective_mw=effective, executed_mw=updated.executed_mw, planned_mw=updated.allocated_mw,
                executed_dc_mw=updated.executed_mw, executed_ppc_blue_ev_mw=updated.executed_ppc_blue_ev_mw,
                executed_bess_storage_mw=updated.executed_bess_storage_mw, executed_curtailment_mw=updated.executed_curtailment_mw,
                executed_absorbed_mw=updated.executed_absorbed_mw,
                effective_ppc_blue_ev_mw=ev, effective_bess_storage_mw=bess, effective_curtailment_mw=curtail,
                effective_absorbed_mw=effective+ev+bess,
                remaining_grid_excess_mw=max(0., updated.grid_excess_mw-updated.executed_absorbed_mw-updated.executed_curtailment_mw),
                fallback_revenue_eur=0.,
                fallback_valuation_status="unpriced_no_revenue_booked",
                tariff_eur_mwh=tariff, estimated_hourly_revenue_eur=revenue,
                executed_revenue_eur=sum(o.offered_mw*o.offered_price_eur for o in accepted),
                executed_savings_eur=sum(o.offered_mw*(o.contract_price_eur-o.offered_price_eur) for o in accepted),
                executed_co2_tons=(updated.executed_mw+updated.executed_ppc_blue_ev_mw)*.28,
                savings_eur=savings, co2_avoided_tons=(effective+ev)*.28,
                compute_trigger=effective > 0, webhook_sent=False,
                reasoning=updated.reasoning, decision=updated.model_dump(),
                fallback=updated.fallback.model_dump() if updated.fallback else None)
