from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import VoltComputeAgent
from schemas import (
    AgentDecision,
    DispatchAction,
    GridTelemetry,
    ResearchCenterTelemetry,
)
from services.router import FIXED_TARIFF_EUR_MWH, OfferPool, load_profiles, rank_datacenters
from simulator import SCENARIO_DESCRIPTIONS, VoltComputeSimulator

app = FastAPI(
    title="VoltCompute API — 24h to Power PPC",
    description=(
        "Backend API pentru agentul AI de arbitraj al surplusului solar PPC "
        "către centre universitare de calcul (RoEduNet / HPC)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = VoltComputeAgent()
simulator = VoltComputeSimulator(agent=agent)

# Registru in-memory pentru auditul deciziilor și acțiunilor Human-in-the-Loop
DECISION_STORE: Dict[str, AgentDecision] = {}
DISPATCH_LOG: List[Dict[str, Any]] = []


class DCTelemetry(BaseModel):
    """Telemetria curentă a unui Data Center pentru clasare."""

    id: str
    free_capacity_mw: float
    temp_c: float


class RankRequest(BaseModel):
    """Payload pentru POST /api/router/rank."""

    surplus_mw: float
    datacenters: List[DCTelemetry]
    rejected_ids: List[str] = []


class EvaluateRequest(BaseModel):
    """Payload compus pentru endpoint-ul POST /api/evaluate."""

    grid: Dict[str, Any]
    center: Dict[str, Any]
    confidence_override: Optional[float] = None


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Verifică starea serviciului și încărcarea modelului ML."""
    return {
        "status": "healthy",
        "service": "VoltCompute API",
        "timestamp": datetime.now().isoformat(),
        "ml_model_loaded": agent.model_artifact is not None,
        "available_scenarios": list(SCENARIO_DESCRIPTIONS.keys()),
    }


@app.post("/api/evaluate", response_model=AgentDecision)
def evaluate_telemetry(payload: EvaluateRequest) -> AgentDecision:
    """
    Evaluează telemetria curentă (rețea PPC + centru universitar) și returnează
    decizia structurată a agentului (AgentDecision).
    """
    decision = agent.run_cycle(
        grid_input=payload.grid,
        center_input=payload.center,
        confidence_override=payload.confidence_override,
    )
    DECISION_STORE[decision.decision_id] = decision
    return decision


@app.post("/api/dispatch")
def dispatch_decision(action: DispatchAction) -> Dict[str, Any]:
    """
    Înregistrează confirmarea sau respingerea unei decizii (Human-in-the-Loop)
    și declanșează execuția prin agent.act().
    """
    stored_decision = DECISION_STORE.get(action.decision_id)
    if stored_decision is None:
        # Dacă decizia provine dintr-o sesiune externă, construim un obiect de bază
        stored_decision = AgentDecision(
            decision_id=action.decision_id,
            surplus_predicted_mw=4.2,
            confidence_score=0.85,
            status="OFFER_GENERATED" if action.action == "APPROVED" else "NORMAL_GRID_OPERATION",
            allocated_mw=4.2 if action.action == "APPROVED" else 0.0,
            fixed_price_eur=30.0,
            reasoning=f"Comandă manuală dispecer ({action.action}) pentru {action.center_id}.",
            top_shap_features={},
        )

    act_payload = agent.act(
        decision=stored_decision,
        dispatch_action=action,
    )
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "dispatch_action": action.model_dump(),
        "execution_result": act_payload,
    }
    DISPATCH_LOG.append(log_entry)
    return log_entry


@app.get("/api/simulation/{scenario_name}")
def run_simulation(scenario_name: str) -> Dict[str, Any]:
    """
    Rulează simularea pe 24 de ore pentru unul dintre cele 3 scenarii de demo:
    - CRITICAL_SOLAR_SUNDAY
    - UNCERTAIN_WEATHER_SHOCK
    - DATACENTER_THERMAL_RISK
    """
    if scenario_name not in SCENARIO_DESCRIPTIONS:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Scenariul '{scenario_name}' nu există.",
                "valid_scenarios": list(SCENARIO_DESCRIPTIONS.keys()),
            },
        )

    timeline = simulator.generate_24h_scenario(scenario_name)
    decisions = [step["decision"] for step in timeline]
    impact = simulator.calculate_impact(decisions)

    return {
        "scenario_name": scenario_name,
        "description": SCENARIO_DESCRIPTIONS[scenario_name],
        "impact_kpis": impact,
        "timeline": timeline,
    }



@app.post("/api/router/rank")
def rank_and_route(payload: RankRequest) -> Dict[str, Any]:
    """
    DECIDE + ACT: clasează Data Center-ele după Match Score (istoric din dc_profiles_and_history.json)
    și întoarce oferta curentă la tarif fix. Nodurile din `rejected_ids` au refuzat deja
    (prețul nu scade; oferta sare la următorul nod) -> ALL_DCS_EXHAUSTED dacă nu mai rămâne niciunul.
    """
    history = {dc["id"]: dc["history"] for dc in load_profiles()["datacenters"]}
    candidates = [
        {
            **dc.model_dump(),
            "offers_sent": history.get(dc.id, {}).get("offers_sent", 0),
            "offers_accepted": history.get(dc.id, {}).get("offers_accepted", 0),
        }
        for dc in payload.datacenters
    ]
    pool = OfferPool(payload.surplus_mw, rank_datacenters(payload.surplus_mw, candidates))
    pool.rejected = [i for i in payload.rejected_ids if any(c["id"] == i for c in candidates)]
    return {
        "tariff_eur_mwh": FIXED_TARIFF_EUR_MWH,
        "pool_status": pool.status,
        "current_offer": pool.current_offer(),
        "ranking": pool.ranking,
    }


@app.get("/api/weather/bucharest")
def get_bucharest_weather_endpoint(
    format_type: str = "macro",
    forecast_days: int = 7,
    past_days: int = 0,
) -> Dict[str, Any]:
    """
    Endpoint pentru date meteo pe București (locația panourilor solare PPC).
    Redenumește automat câmpurile din API în numele din headerele CSV existente:
      - format_type="macro": conform 1_macro_vreme_piata.csv
      - format_type="energy": conform energy_datacenter_data.csv
    Suportă date din trecut (past_days) și din viitor (forecast_days).
    """
    from services.weather_service import (
        BUCHAREST_LAT,
        BUCHAREST_LON,
        get_energy_datacenter_dataframe,
        get_macro_weather_dataframe,
    )

    if format_type == "energy":
        df = get_energy_datacenter_dataframe(forecast_days=forecast_days, past_days=past_days)
    else:
        df = get_macro_weather_dataframe(forecast_days=forecast_days, past_days=past_days)

    return {
        "location": "București",
        "latitude": BUCHAREST_LAT,
        "longitude": BUCHAREST_LON,
        "format": format_type,
        "past_days": past_days,
        "forecast_days": forecast_days,
        "records_count": len(df),
        "columns": list(df.columns),
        "data": df.to_dict(orient="records"),
    }


@app.get("/api/weather/bucharest/lookahead")
def get_bucharest_weather_lookahead_endpoint(
    hours_ahead: int = 2,
) -> Dict[str, Any]:
    """
    Endpoint pentru prognoza meteo în avans cu 2 ore (sau N ore) peste București,
    folosită pentru generarea anticipată a ofertelor de arbitraj energetic.
    """
    from services.weather_service import get_weather_lookahead

    return get_weather_lookahead(hours_ahead=hours_ahead)
