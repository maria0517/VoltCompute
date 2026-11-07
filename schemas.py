from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4
import math
import re
import unicodedata
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, Field, field_validator, model_validator


def redact_pii(text: str) -> str:
    text = re.sub(r"\b\d{13}\b", "[CNP_REDACTED]", text)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL_REDACTED]", text)
    return re.sub(r"(?<!\w)(?:\+?40[ .-]?)?0?7(?:[ .-]?\d){8}(?!\d)", "[PHONE_REDACTED]", text)


def redact_pii_and_guard_injection(text: str) -> str:
    normalized = ''.join(c for c in unicodedata.normalize('NFKD', text.lower()) if not unicodedata.combining(c))
    normalized = " ".join(normalized.split())
    if re.search(r"(?:ignore|disregard|override|ignora|ignorați|ignorati).{0,60}(?:instruction|instruct|previous|anterioar|system)|system\s*prompt|<\|im_start\|>", normalized):
        raise ValueError("Unsafe instruction-like input blocked")
    return redact_pii(text)


def safe_context(text: str) -> str:
    """Sanitized, structurally delimited untrusted operator context (never instructions)."""
    clean = redact_pii_and_guard_injection(text)
    if "###" in clean or "<|" in clean:
        raise ValueError("Context delimiter injection blocked")
    return f"### Context:\n{clean}\n### End Context"


def _sanitize(value):
    if isinstance(value, str):
        return redact_pii_and_guard_injection(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite telemetry rejected")
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    return value


class BaseModel(PydanticBaseModel):
    model_config = ConfigDict(allow_inf_nan=False, hide_input_in_errors=True, validate_assignment=True)

    @model_validator(mode="before")
    @classmethod
    def guard_input(cls, value):
        return _sanitize(value)


class GridTelemetry(BaseModel):
    """Telemetria primită în timp real de la parcul fotovoltaic și rețeaua PPC."""

    wind_speed_kmh: Optional[float] = Field(default=12.0, ge=0, le=150)
    prosumer_battery_soc_pct: Optional[float] = Field(default=55.0, ge=0, le=100)
    current_surplus_mw: Optional[float] = Field(default=None, ge=0)
    total_prosumer_load_mw: Optional[float] = Field(default=None, ge=0)
    operator_notes: Optional[str] = None
    model_features: Optional[Dict[str, Optional[float]]] = None
    data_source: Literal["legacy", "real_csv", "custom"] = "legacy"
    demo_injected_surplus_mw: float = Field(default=0, ge=0, le=100)
    scenario_name: Optional[str] = None
    demo_market_price_eur_mwh: Optional[float] = None

    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp-ul citirii în format ISO 8601",
    )
    hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Ora curentă din zi (0-23)",
    )
    solar_irradiance_wm2: float = Field(
        ...,
        ge=0.0,
        le=1500.0,
        description="Iradianța solară măsurată în W/m²",
    )
    ambient_temp_celsius: Optional[float] = Field(
        default=None,
        ge=-40.0,
        le=60.0,
        description="Temperatura ambientală la nivelul panourilor (°C). Dacă lipsește (None), agentul o imputează.",
    )
    day_ahead_price_eur: Optional[float] = Field(
        default=None,
        description="Prețul energiei pe piața PZU / Day-Ahead în EUR/MWh. Dacă lipsește (None), agentul îl imputează.",
    )
    grid_capacity_limit_mw: float = Field(
        default=10.0,
        gt=0.0,
        description="Capacitatea maximă de injecție în linia de transport fără risc de supraîncălzire (MW)",
    )
    weather_stability_index: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Indice opțional de stabilitate meteo (0.0 = furtună/fluctuații severe, 1.0 = cer senin stabil)",
    )
    is_weekend: Optional[int] = Field(
        default=0,
        ge=0,
        le=1,
        description="Indicator zi de weekend (1 = Sâmbătă/Duminică cu consum redus în rețea, 0 = Luni-Vineri)",
    )


class ResearchCenterTelemetry(BaseModel):
    """Telemetria primită de la nodul academic / centrul de cercetare (ex. RoEduNet / HPC)."""

    center_id: str = Field(
        default="ROEDUNET-BUC-HPC-01",
        min_length=2,
        description="Identificatorul unic al centrului universitar de date",
    )
    cooling_temp_celsius: float = Field(
        ...,
        ge=5.0,
        le=60.0,
        description="Temperatura curentă în camera serverelor / circuitul de răcire (°C)",
    )
    max_absorb_capacity_mw: float = Field(
        ...,
        ge=0.0,
        description="Puterea maximă suplimentară pe care clusterul HPC o poate absorbi instantaneu (MW)",
    )
    pending_compute_jobs: int = Field(
        ...,
        ge=0,
        description="Numărul de joburi masive în așteptare (antrenare LLM, simulări fizice/climatice)",
    )


DecisionStatus = Literal[
    "OFFER_GENERATED",
    "NORMAL_GRID_OPERATION",
    "ESCALATE_TO_HUMAN",
    "DATA_ERROR",
]


class DataCenterTelemetry(BaseModel):
    timestamp: Optional[str] = None
    dc_id: str = Field(min_length=1)
    dc_name: Optional[str] = None
    capacitate_libera_mw: float = Field(ge=0)
    temp_servere_c: float = Field(ge=5, le=60)
    limita_termica_c: float = Field(ge=10, le=60)
    pret_contract_eur_mwh: float = Field(ge=0)

    @field_validator("dc_id")
    @classmethod
    def reserved_target(cls, value):
        if value == "PPC_FALLBACK":
            raise ValueError("Reserved dispatch target")
        return value


class DataCenterOffer(BaseModel):
    dc_id: str
    dc_name: str
    available_capacity_mw: float = Field(ge=0)
    current_temp_c: float
    thermal_limit_c: float
    contract_price_eur: float = Field(ge=0)
    offered_mw: float = Field(ge=0)
    offered_price_eur: float = Field(ge=10)
    discount_pct: float
    priority_score: float = Field(ge=0)
    status: Literal["OFFER_GENERATED", "ACCEPTED", "REJECTED_BY_OPERATOR", "REJECTED_THERMAL_SAFETY", "STANDBY", "ESCALATE_TO_HUMAN"]
    auction_round: int = Field(default=1, ge=1)


class FallbackAllocation(BaseModel):
    ppc_blue_ev_mw: float = Field(default=0, ge=0, le=3.5)
    bess_storage_mw: float = Field(default=0, ge=0, le=4)
    curtailment_mw: float = Field(default=0, ge=0)
    total_fallback_absorbed_mw: float = Field(default=0, ge=0)
    active_tier: Literal["NONE", "TIER_1_PPC_BLUE_EV", "TIER_2_BESS_STORAGE", "TIER_3_CURTAILMENT"] = "NONE"
    explanation: str = ""


class ForecastPoint(BaseModel):
    prediction_horizon_hours: int = Field(ge=1, le=4)
    predicted_export_mw: float = Field(ge=0)
    prediction_lower_mw: float = Field(ge=0)
    prediction_upper_mw: float = Field(ge=0)
    confidence_score: float = Field(ge=0, le=1)
    prediction_source: str = "xgboost"
    weather_estimate_method: str = "seasonal_observation_t_minus_22"
    weather_estimate_is_issued_forecast: bool = False


class AgentDecision(BaseModel):
    """Decizia structurată produsă de agentul VoltCompute."""

    prediction_horizon_hours: int = 2
    predicted_export_mw: float = Field(default=0, ge=0)
    grid_excess_mw: float = Field(default=0, ge=0)
    demo_injected_surplus_mw: float = Field(default=0, ge=0)
    surplus_semantics: str = "aggregate_export_available_for_arbitrage"
    forecast_window: List[ForecastPoint] = Field(default_factory=list)
    weather_estimate_method: str = "unavailable"
    weather_estimate_is_issued_forecast: bool = False
    requires_human_approval: bool = True
    dc_offers: List[DataCenterOffer] = Field(default_factory=list)
    fallback: Optional[FallbackAllocation] = None
    shap_percentages: Dict[str, float] = Field(default_factory=dict)
    shap_values: Dict[str, float] = Field(default_factory=dict)
    shap_base_value: Optional[float] = None
    raw_prediction_mw: Optional[float] = None
    shap_method: str = "unavailable"
    prediction_source: str = "unavailable"
    prediction_lower_mw: Optional[float] = None
    prediction_upper_mw: Optional[float] = None
    executed_mw: float = Field(default=0, ge=0)
    executed_ppc_blue_ev_mw: float = Field(default=0, ge=0)
    executed_bess_storage_mw: float = Field(default=0, ge=0)
    executed_curtailment_mw: float = Field(default=0, ge=0)
    executed_absorbed_mw: float = Field(default=0, ge=0)
    fallback_plan_revision: int = Field(default=1, ge=1)
    fallback_approved_revision: int = Field(default=0, ge=0)
    execution_mode: Literal["planned", "simulated"] = "planned"

    decision_id: str = Field(
        default_factory=lambda: f"dec-{uuid4().hex[:8]}",
        description="ID unic al deciziei pentru audit și trasabilitate",
    )
    surplus_predicted_mw: float = Field(
        ...,
        ge=0.0,
        description="Export disponibil pentru arbitraj (plus injecția demo); în profil legacy, exces fizic peste limita liniei. Vezi surplus_semantics.",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Scorul de confidență al predicției (0.0 - 1.0)",
    )
    status: DecisionStatus = Field(
        ...,
        description="Starea operațională decisă de agent",
    )
    allocated_mw: float = Field(
        ...,
        ge=0.0,
        description="Cantitatea de energie redirecționată către centrul de cercetare (MW)",
    )
    fixed_price_eur: float = Field(
        default=30.0,
        description="Tariful verde fix preferențial pentru nodurile academice (EUR/MWh)",
    )
    reasoning: str = Field(
        ...,
        description="Explicația deciziei în limbaj natural pentru dispecerul PPC",
    )
    top_shap_features: Dict[str, float] = Field(
        default_factory=dict,
        description="Contribuția principalelor variabile la predicție (explicabilitate XAI / SHAP)",
    )
    similar_historical_cases: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Cele mai similare cazuri din istoric folosite pentru justificarea deciziei",
    )
    data_quality_warnings: List[str] = Field(
        default_factory=list,
        description="Avertismente privind senzori secundari lipsă care au fost imputați automat",
    )

    @field_validator("confidence_score")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(float(v), 4)

    @field_validator("surplus_predicted_mw", "allocated_mw")
    @classmethod
    def round_mw(cls, v: float) -> float:
        return float(v)


class DispatchAction(BaseModel):
    """Comanda de confirmare sau respingere (Human-in-the-Loop / Auto-Dispatch)."""

    decision_id: str = Field(
        ...,
        min_length=1,
        description="ID-ul deciziei care este aprobată sau respinsă",
    )
    center_id: str = Field(
        ...,
        min_length=1,
        description="ID-ul centrului de cercetare țintă",
    )
    trigger_fallback: bool = True
    fallback_revision: Optional[int] = Field(default=None, ge=1)
    auction_step: int = Field(default=1, ge=1, le=100)
    action: Literal["APPROVED", "REJECTED", "DUTCH_AUCTION_STEP"] = Field(
        ...,
        description="Acțiunea finală a operatorului sau a sistemului automat",
    )


class EvaluateRequest(BaseModel):
    grid: Dict[str, Any]
    center: Optional[Dict[str, Any]] = None
    data_centers: Optional[List[Dict[str, Any]]] = None
    confidence_override: Optional[float] = Field(default=None, ge=0, le=1)


class GridStatusResponse(BaseModel):
    step: int
    total_steps: int
    timestamp: str
    grid: GridTelemetry
    data_centers: List[DataCenterTelemetry]
    price_pzu_eur_mwh: float
    avg_battery_soc_pct: float
    data_source: str


class HealthResponse(BaseModel):
    status: str
    service: str = "VoltCompute API"
    timestamp: str
    ml_model_loaded: bool
    metrics: Dict[str, Any] = Field(default_factory=dict)
    available_scenarios: List[str]


class ExecutionResult(BaseModel):
    decision_id: str
    target_center_id: str
    status: DecisionStatus
    execution_state: str
    execution_mode: str
    simulated: bool
    effective_mw: float = Field(ge=0)
    executed_mw: float = Field(ge=0)
    executed_dc_mw: float = Field(ge=0)
    executed_ppc_blue_ev_mw: float = Field(ge=0)
    executed_bess_storage_mw: float = Field(ge=0)
    executed_curtailment_mw: float = Field(ge=0)
    executed_absorbed_mw: float = Field(ge=0)
    effective_ppc_blue_ev_mw: float = Field(ge=0)
    effective_bess_storage_mw: float = Field(ge=0)
    effective_curtailment_mw: float = Field(ge=0)
    effective_absorbed_mw: float = Field(ge=0)
    remaining_grid_excess_mw: float = Field(ge=0)
    fallback_revenue_eur: float = 0
    planned_mw: float = Field(ge=0)
    tariff_eur_mwh: Optional[float]
    fallback_valuation_status: str = "unpriced_no_revenue_booked"
    estimated_hourly_revenue_eur: float
    executed_revenue_eur: float
    executed_savings_eur: float
    executed_co2_tons: float
    savings_eur: float
    co2_avoided_tons: float
    compute_trigger: bool
    webhook_sent: bool
    reasoning: str
    decision: AgentDecision
    fallback: Optional[FallbackAllocation]


class DispatchResponse(ExecutionResult):
    timestamp: str
    dispatch_action: DispatchAction
    execution_result: ExecutionResult


class SimulateScenarioRequest(BaseModel):
    scenario_name: Literal["CRITICAL_SOLAR_SUNDAY", "UNCERTAIN_WEATHER_SHOCK", "DATACENTER_THERMAL_RISK"] = "CRITICAL_SOLAR_SUNDAY"
    custom: Optional[Dict[str, Any]] = None
    start_step: int = Field(default=48, ge=24, le=696)
    simulate_approvals: bool = False


class SimulationStep(BaseModel):
    step: int
    hour: int
    hour_label: str
    timestamp: str
    grid_telemetry: GridTelemetry
    data_centers: List[DataCenterTelemetry]
    net_solar_generation_mw: float
    grid_capacity_limit_mw: float
    decision: AgentDecision
    simulation_mode: str
    operator_approvals_simulated: bool


class SimulationResponse(BaseModel):
    scenario_name: str
    description: str
    impact_kpis: Dict[str, Any]
    timeline: List[SimulationStep]
