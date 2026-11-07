from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4
from pydantic import BaseModel, Field, field_validator


class GridTelemetry(BaseModel):
    """Telemetria primită în timp real de la parcul fotovoltaic și rețeaua PPC."""

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


class AgentDecision(BaseModel):
    """Decizia structurată produsă de agentul VoltCompute."""

    decision_id: str = Field(
        default_factory=lambda: f"dec-{uuid4().hex[:8]}",
        description="ID unic al deciziei pentru audit și trasabilitate",
    )
    surplus_predicted_mw: float = Field(
        ...,
        ge=0.0,
        description="Surplusul energetic estimat peste capacitatea de siguranță a liniei (MW)",
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
        return round(float(v), 3)


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
    action: Literal["APPROVED", "REJECTED"] = Field(
        ...,
        description="Acțiunea finală a operatorului sau a sistemului automat",
    )
