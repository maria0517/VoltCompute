"""FastAPI aliases share one validated, stateful HITL execution core."""
from datetime import datetime
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from agent import VoltComputeAgent
from schemas import (AgentDecision, DispatchAction, DispatchResponse, EvaluateRequest,
                     GridStatusResponse, HealthResponse, SimulateScenarioRequest, SimulationResponse)
from simulator import SCENARIO_DESCRIPTIONS, VoltComputeSimulator
from train_pipeline import load_grid_status

app = FastAPI(title="VoltCompute API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])
agent = VoltComputeAgent()
simulator = VoltComputeSimulator(agent)
DECISION_STORE: Dict[str, AgentDecision] = {}
DISPATCH_LOG: List[Dict[str, Any]] = []


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    # Pydantic/FastAPI normally echo raw invalid inputs (including PII). Never do so.
    # Transport/schema failures (including non-JSON NaN/Infinity) are HTTP 422.
    # Valid envelopes with invalid telemetry reaching run_cycle return DATA_ERROR.
    return JSONResponse(status_code=422, content={"detail": "Invalid request schema; no agent decision or execution was performed."})


@app.get("/health", response_model=HealthResponse)
def health_check():
    loaded = agent.model_artifact is not None
    return dict(status="healthy" if loaded else "degraded", timestamp=datetime.now().isoformat(),
                ml_model_loaded=loaded, metrics=agent.model_artifact["metrics"] if loaded else {},
                available_scenarios=list(SCENARIO_DESCRIPTIONS))


@app.get("/api/grid-status", response_model=GridStatusResponse)
def grid_status(step: int = Query(default=37, ge=0, le=719)):
    try:
        return load_grid_status(step)
    except (ValueError, OSError):
        raise HTTPException(503, "CSV telemetry unavailable") from None


@app.post("/api/evaluate", response_model=AgentDecision)
@app.post("/api/evaluate-arbitrage", response_model=AgentDecision)
def evaluate_telemetry(payload: EvaluateRequest):
    decision = agent.run_cycle(payload.grid, payload.center, payload.confidence_override, payload.data_centers)
    DECISION_STORE[decision.decision_id] = decision
    return decision


@app.post("/api/dispatch", response_model=DispatchResponse)
@app.post("/api/actions/execute", response_model=DispatchResponse)
def dispatch_decision(action: DispatchAction):
    decision = DECISION_STORE.get(action.decision_id)
    if decision is None:
        raise HTTPException(404, "Unknown decision ID")
    try:
        result = agent.act(decision, dispatch_action=action)
    except ValueError:
        raise HTTPException(409, "Action blocked: unknown center, inactive offer or safety constraint") from None
    DECISION_STORE[decision.decision_id] = AgentDecision.model_validate(result["decision"])
    response = dict(**result, timestamp=datetime.now().isoformat(), dispatch_action=action.model_dump(), execution_result=result)
    DISPATCH_LOG.append(response)
    return response


def _simulate(payload):
    try:
        timeline = simulator.generate_real_24h_scenario(payload.scenario_name, payload.custom,
                                                       payload.start_step, payload.simulate_approvals)
        # Validate before storing or returning, including user-supplied custom fields.
        result = SimulationResponse.model_validate(dict(scenario_name=payload.scenario_name,
            description="Real CSV scenario; interventions and optional operator approvals are simulated.",
            impact_kpis=simulator.calculate_impact([s["decision"] for s in timeline]), timeline=timeline))
    except (ValueError, TypeError, KeyError):
        raise HTTPException(422, "Invalid scenario telemetry") from None
    for step in result.timeline:
        DECISION_STORE[step.decision.decision_id] = step.decision
    return result


@app.get("/api/simulation/{scenario_name}", response_model=SimulationResponse)
def run_simulation(scenario_name: str):
    if scenario_name not in SCENARIO_DESCRIPTIONS:
        raise HTTPException(404, "Unknown scenario")
    return _simulate(SimulateScenarioRequest(scenario_name=scenario_name))


@app.post("/api/simulate-scenario", response_model=SimulationResponse)
def simulate_scenario(payload: SimulateScenarioRequest):
    return _simulate(payload)
