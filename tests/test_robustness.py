import pytest

from agent import VoltComputeAgent
from schemas import GridTelemetry, ResearchCenterTelemetry
from simulator import VoltComputeSimulator


@pytest.fixture
def agent() -> VoltComputeAgent:
    return VoltComputeAgent()


def test_corrupted_telemetry_returns_data_error(agent: VoltComputeAgent) -> None:
    """
    Verifică faptul că datele corupte / în afara limitelor fizice (ex. hour=29, iradianță negativă)
    sunt prinse în perceive() și returnează status='DATA_ERROR' cu 0 MW alocați.
    """
    corrupted_grid = {
        "timestamp": "2026-06-21T13:00:00",
        "hour": 29,  # Invalid (trebuie 0-23)
        "solar_irradiance_wm2": -250.0,  # Invalid (trebuie >= 0)
        "ambient_temp_celsius": 26.0,
        "day_ahead_price_eur": 15.0,
    }
    valid_center = {
        "center_id": "ROEDUNET-UPB",
        "cooling_temp_celsius": 19.0,
        "max_absorb_capacity_mw": 5.0,
        "pending_compute_jobs": 12,
    }

    decision = agent.run_cycle(corrupted_grid, valid_center)
    assert decision.status == "DATA_ERROR"
    assert decision.allocated_mw == 0.0
    assert decision.confidence_score == 0.0


def test_low_confidence_triggers_human_escalation(agent: VoltComputeAgent) -> None:
    """
    Verifică faptul că un scor de confidență sub pragul de 0.75 declanșează
    status='ESCALATE_TO_HUMAN' și blochează alocarea automată (allocated_mw == 0.0).
    """
    grid = GridTelemetry(
        hour=13,
        solar_irradiance_wm2=1050.0,
        ambient_temp_celsius=26.0,
        day_ahead_price_eur=10.0,
        grid_capacity_limit_mw=10.0,
    )
    center = ResearchCenterTelemetry(
        center_id="ROEDUNET-UPB",
        cooling_temp_celsius=19.5,
        max_absorb_capacity_mw=6.0,
        pending_compute_jobs=20,
    )

    decision = agent.run_cycle(grid, center, confidence_override=0.62)
    assert decision.status == "ESCALATE_TO_HUMAN"
    assert decision.allocated_mw == 0.0
    assert decision.confidence_score == pytest.approx(0.62, rel=1e-3)


def test_thermal_limit_blocks_allocation(agent: VoltComputeAgent) -> None:
    """
    Verifică protecția termică a serverelor: dacă cooling_temp_celsius > 26.0°C (ex. 28.5°C),
    agentul returnează status='NORMAL_GRID_OPERATION' și allocated_mw=0.0 chiar dacă există surplus mare.
    """
    grid = GridTelemetry(
        hour=13,
        solar_irradiance_wm2=1090.0,
        ambient_temp_celsius=34.0,
        day_ahead_price_eur=8.0,
        grid_capacity_limit_mw=10.0,
    )
    hot_center = ResearchCenterTelemetry(
        center_id="ROEDUNET-UPB",
        cooling_temp_celsius=28.5,  # Peste pragul HVAC de 26.0°C
        max_absorb_capacity_mw=6.0,
        pending_compute_jobs=30,
    )

    decision = agent.run_cycle(grid, hot_center, confidence_override=0.95)
    assert decision.status == "NORMAL_GRID_OPERATION"
    assert decision.allocated_mw == 0.0
    assert "28.5" in decision.reasoning


def test_high_surplus_generates_offer_at_30_eur(agent: VoltComputeAgent) -> None:
    """
    Verifică generarea ofertei (OFFER_GENERATED) la 30 EUR/MWh atunci când surplusul > 5.0 MW, PZU < 25 EUR/MWh,
    confidența >= 0.75 și temperatura serverelor <= 26.0°C.
    """
    grid = GridTelemetry(
        hour=13,
        solar_irradiance_wm2=1100.0,
        ambient_temp_celsius=25.0,
        day_ahead_price_eur=5.0,
        grid_capacity_limit_mw=10.0,
    )
    cool_center = ResearchCenterTelemetry(
        center_id="ROEDUNET-UPB",
        cooling_temp_celsius=19.0,
        max_absorb_capacity_mw=5.0,
        pending_compute_jobs=25,
    )

    decision = agent.run_cycle(grid, cool_center, confidence_override=0.95)
    assert decision.status == "OFFER_GENERATED"
    assert decision.surplus_predicted_mw > 5.0
    assert 0.0 < decision.allocated_mw <= cool_center.max_absorb_capacity_mw
    assert decision.fixed_price_eur == 30.0


def test_simulator_24h_scenarios_and_impact() -> None:
    """
    Verifică motorul de simulare pe 24h pentru toate cele 3 scenarii de demo și calculul KPI-urilor.
    """
    sim = VoltComputeSimulator()

    sunday = sim.generate_24h_scenario("CRITICAL_SOLAR_SUNDAY")
    assert len(sunday) == 24
    sunday_impact = sim.calculate_impact([step["decision"] for step in sunday])
    assert sunday_impact["mwh_saved_from_curtailment"] > 0.0
    assert sunday_impact["ppc_revenue_eur"] > 0.0

    weather_shock = sim.generate_24h_scenario("UNCERTAIN_WEATHER_SHOCK")
    shock_impact = sim.calculate_impact([step["decision"] for step in weather_shock])
    assert shock_impact["human_escalations_hours"] >= 3

    thermal_risk = sim.generate_24h_scenario("DATACENTER_THERMAL_RISK")
    thermal_impact = sim.calculate_impact([step["decision"] for step in thermal_risk])
    assert thermal_impact["mwh_saved_from_curtailment"] == 0.0



def test_anomalous_sensor_triggers_uncertain(agent: VoltComputeAgent) -> None:
    """Senzor cu valoare la peste 3σ de istoricul de antrenare -> Status UNCERTAIN, 0 MW alocați."""
    grid = GridTelemetry(
        hour=13,
        solar_irradiance_wm2=1050.0,
        ambient_temp_celsius=55.0,  # ~5σ peste media de antrenare
        day_ahead_price_eur=10.0,
    )
    center = ResearchCenterTelemetry(cooling_temp_celsius=20.0, max_absorb_capacity_mw=6.0, pending_compute_jobs=10)
    decision = agent.run_cycle(grid, center, confidence_override=0.95)
    assert decision.status == "UNCERTAIN"
    assert decision.allocated_mw == 0.0
    assert decision.data_quality_warnings


def test_high_pzu_price_blocks_offer(agent: VoltComputeAgent) -> None:
    """Fără arbitraj dacă prețul PZU este >= 25 EUR/MWh, chiar și cu surplus mare."""
    grid = GridTelemetry(hour=13, solar_irradiance_wm2=1100.0, ambient_temp_celsius=25.0, day_ahead_price_eur=60.0)
    center = ResearchCenterTelemetry(cooling_temp_celsius=19.0, max_absorb_capacity_mw=6.0, pending_compute_jobs=10)
    decision = agent.run_cycle(grid, center, confidence_override=0.95)
    assert decision.status == "NORMAL_GRID_OPERATION"
    assert decision.allocated_mw == 0.0


def test_shap_values_come_from_model(agent: VoltComputeAgent) -> None:
    """SHAP-ul returnat provine din TreeSHAP-ul XGBoost (cheile = features ale modelului)."""
    if agent.model_artifact is None or not hasattr(agent.model_artifact["model"], "get_booster"):
        pytest.skip("Modelul XGBoost nu este disponibil")
    grid = GridTelemetry(hour=13, solar_irradiance_wm2=1000.0, ambient_temp_celsius=26.0, day_ahead_price_eur=10.0)
    _, _, shap_vals = agent.estimate(grid)
    assert set(shap_vals) <= set(agent.model_artifact["feature_columns"])
    assert "effective_solar_index" in shap_vals or "solar_irradiance_wm2" in shap_vals
