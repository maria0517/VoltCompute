import calendar
import ctypes
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agent import VoltComputeAgent
from schemas import GridTelemetry
from services.router import (
    FIXED_TARIFF_EUR_MWH,
    HVAC_MAX_TEMP_C,
    POOL_ALL_DCS_EXHAUSTED,
    load_profiles,
    rank_datacenters,
)
from simulator import VoltComputeSimulator
from train import (
    LIVE_TEST_DATA_PATH,
    MODEL_PATH,
    TRAIN_DATA_PATH,
    ensure_split_datasets,
    train_pipeline,
)

FIXED_PRICE_EUR = FIXED_TARIFF_EUR_MWH
LOOKAHEAD_HOURS = 2
HISTORY_HOURS = 24
REFERENCE_TARIFF_EUR = VoltComputeSimulator.STANDARD_RETAIL_PRICE_EUR
DATACENTER_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "3_data_centers.csv"
TEMP_RISE_PER_MW_C = 0.5

PPC_SOLAR_PARK = {
    "id": "PPC_SOLAR_MUNTENIA",
    "name": "Parc Fotovoltaic PPC Renewables (Muntenia Sud)",
    "short": "Parc Solar PPC (18.5 MWp)",
    "lat": 44.235,
    "lon": 27.280,
}

DC_GEO_METADATA: Dict[str, Dict[str, Any]] = {
    "DC_AI_BUCURESTI": {
        "city": "București (UPB / RoEduNet)",
        "lat": 44.4385,
        "lon": 26.0495,
        "waypoints": [(44.235, 27.280), (44.360, 26.650), (44.4385, 26.0495)],
        "workload": "LLM Training · RoEduNet Cluster",
    },
    "DC_CLOUD_CLUJ": {
        "city": "Cluj-Napoca (UTCN Cloud)",
        "lat": 46.7712,
        "lon": 23.6236,
        "waypoints": [
            (44.235, 27.280),
            (44.950, 26.020),
            (45.650, 25.580),
            (46.300, 24.500),
            (46.7712, 23.6236),
        ],
        "workload": "Distributed Cloud Batch · UTCN",
    },
    "DC_EDGE_TIMISOARA": {
        "city": "Timișoara (UVT HPC Edge)",
        "lat": 45.7473,
        "lon": 21.2314,
        "waypoints": [
            (44.235, 27.280),
            (44.320, 23.800),
            (44.630, 22.660),
            (45.200, 21.900),
            (45.7473, 21.2314),
        ],
        "workload": "HPC Scientific Simulation · UVT",
    },
}

GRID_400KV_BACKBONE: List[Tuple[float, float]] = [
    (44.18, 28.63),
    (44.33, 28.03),
    (44.235, 27.280),
    (44.4385, 26.0495),
    (44.85, 24.87),
    (44.32, 23.80),
    (44.63, 22.66),
    (45.7473, 21.2314),
    (46.18, 21.31),
    (47.05, 21.92),
    (46.7712, 23.6236),
    (46.45, 24.23),
    (45.65, 25.58),
    (44.95, 26.02),
    (44.4385, 26.0495),
]


class _SYSTEMTIME(ctypes.Structure):
    _fields_ = [
        ("wYear", ctypes.c_uint16),
        ("wMonth", ctypes.c_uint16),
        ("wDayOfWeek", ctypes.c_uint16),
        ("wDay", ctypes.c_uint16),
        ("wHour", ctypes.c_uint16),
        ("wMinute", ctypes.c_uint16),
        ("wSecond", ctypes.c_uint16),
        ("wMilliseconds", ctypes.c_uint16),
    ]


def get_windows_month_start() -> datetime:
    """Preia anul și luna de start printr-un apel API nativ către Windows (`kernel32.GetLocalTime`)."""
    try:
        if hasattr(ctypes, "windll") and hasattr(ctypes.windll, "kernel32"):
            sys_time = _SYSTEMTIME()
            ctypes.windll.kernel32.GetLocalTime(ctypes.byref(sys_time))
            if 2000 <= int(sys_time.wYear) <= 2100 and 1 <= int(sys_time.wMonth) <= 12:
                return datetime(int(sys_time.wYear), int(sys_time.wMonth), 1, 0, 0, 0)
    except Exception:
        pass
    now = datetime.now()
    return datetime(now.year, now.month, 1, 0, 0, 0)


class DashboardStateService:
    """
    Serviciu stateful pentru consola SCADA și harta națională 400 kV,
    expus prin REST API către frontend-ul TypeScript.
    """

    def __init__(self, agent: Optional[VoltComputeAgent] = None) -> None:
        train_df, live_df = ensure_split_datasets()
        live_df = live_df.reset_index(drop=True)
        live_df["dt"] = pd.to_datetime(live_df["timestamp"])
        if "gross_solar_mw" not in live_df.columns:
            derating = 1.0 - np.maximum(0.0, (live_df["ambient_temp_celsius"] - 25.0) * 0.004)
            live_df["gross_solar_mw"] = live_df["solar_irradiance_wm2"] / 1000.0 * 17.8 * derating
        if "net_injection_mw" not in live_df.columns:
            local_load = 1.4 + 0.5 * np.sin(np.pi * live_df["hour"] / 12.0)
            live_df["net_injection_mw"] = np.maximum(0.0, live_df["gross_solar_mw"] - local_load)

        self.feed: pd.DataFrame = live_df
        self.n_train_rows: int = len(train_df)
        self.n_feed: int = len(live_df)
        self.base_live_dt: datetime = get_windows_month_start()
        self.y_axis_max: float = round(float(live_df["gross_solar_mw"].max()) * 1.08 + 0.5, 2)

        dc_df = pd.read_csv(DATACENTER_DATA_PATH)
        dc_df["dt"] = pd.to_datetime(dc_df["timestamp"])
        if "hour" not in dc_df.columns:
            dc_df["hour"] = dc_df["dt"].dt.hour
        self.dc_lookup = dc_df.set_index(["dt", "dc_id"]).to_dict("index")
        self.dc_hour_lookup = {
            (int(row["hour"]), row["dc_id"]): row.to_dict() for _, row in dc_df.iterrows()
        }

        self.dc_profiles = load_profiles()
        self.hvac_limit_c = float(self.dc_profiles.get("hvac_max_temp_c", HVAC_MAX_TEMP_C))
        self.datacenters: List[Dict[str, Any]] = []
        for dc in self.dc_profiles["datacenters"]:
            geo = DC_GEO_METADATA.get(dc["id"], {})
            self.datacenters.append({**dc, **geo})
        self.dc_by_id: Dict[str, Dict[str, Any]] = {n["id"]: n for n in self.datacenters}

        self.agent = agent or VoltComputeAgent(model_path=MODEL_PATH)
        if (
            self.agent.model_artifact is None
            or "model" not in self.agent.model_artifact
            or "injection_model" not in self.agent.model_artifact
        ):
            train_pipeline()
            self.agent._load_model()

        self.pred_cache: Dict[int, Tuple[float, float, Dict[str, float], float]] = {}
        self.deals: Dict[int, Dict[str, float]] = {}
        self.refused: Dict[int, set] = {}
        self.dc_history: Dict[str, Dict[str, int]] = {
            n["id"]: dict(n.get("history", {"offers_sent": 0, "offers_accepted": 0}))
            for n in self.datacenters
        }

        # Pasul inițial (o oră de vârf solar la prânz pentru demo)
        candidates = (
            self.feed[(self.feed["hour"] == 13) & (self.feed.get("surplus_mw", 0) > 5.2)]
            if "surplus_mw" in self.feed.columns
            else self.feed.iloc[0:0]
        )
        if len(candidates) > 1:
            self.default_step = int(candidates.index[1])
        elif len(candidates) > 0:
            self.default_step = int(candidates.index[0])
        else:
            self.default_step = 37

    def month_window(self, cycle: int) -> Tuple[datetime, int, int]:
        total_month_idx = (self.base_live_dt.month - 1) + int(cycle)
        year = self.base_live_dt.year + (total_month_idx // 12)
        month = (total_month_idx % 12) + 1
        days_in_month = calendar.monthrange(year, month)[1]
        month_start = datetime(year, month, 1, 0, 0, 0)
        offset_hours = int(round((month_start - self.base_live_dt).total_seconds() / 3600.0))
        return month_start, days_in_month * 24, offset_hours

    def ring_row(self, v_idx: int) -> pd.Series:
        return self.feed.iloc[v_idx % self.n_feed]

    def ring_dt(self, v_idx: int) -> datetime:
        return self.base_live_dt + timedelta(hours=int(v_idx))

    def _cycle_variation(self, v_idx: int) -> Tuple[float, float, float]:
        v_cycle = v_idx // self.n_feed
        if v_cycle == 0:
            return 1.0, 0.0, 0.0
        rng = np.random.default_rng(seed=100_003 + int(v_idx) * 31)
        irr_factor = float(np.clip(1.0 + rng.normal(0.0, 0.045), 0.88, 1.12))
        temp_offset = float(rng.normal(0.0, 0.6))
        inj_noise = float(rng.normal(0.0, 0.28))
        return irr_factor, temp_offset, inj_noise

    def net_injection_mw(self, v_idx: int) -> float:
        row = self.ring_row(v_idx)
        irr_factor, _, inj_noise = self._cycle_variation(v_idx)
        base_real_inj = float(row["net_injection_mw"])
        if int(row["hour"]) < 6 or int(row["hour"]) > 19 or base_real_inj <= 0.01:
            return 0.0
        return round(max(0.0, base_real_inj * irr_factor + inj_noise), 3)

    def grid_telemetry(self, v_idx: int) -> GridTelemetry:
        row = self.ring_row(v_idx)
        irr_factor, temp_offset, _ = self._cycle_variation(v_idx)
        dt_val = self.ring_dt(v_idx)
        irr_val = float(np.clip(float(row["solar_irradiance_wm2"]) * irr_factor, 0.0, 1400.0))
        temp_val = float(np.clip(float(row["ambient_temp_celsius"]) + temp_offset, -20.0, 55.0))
        return GridTelemetry(
            timestamp=dt_val.isoformat(),
            hour=int(row["hour"]),
            solar_irradiance_wm2=round(irr_val, 2),
            ambient_temp_celsius=round(temp_val, 2),
            day_ahead_price_eur=float(row["day_ahead_price_eur"]),
            grid_capacity_limit_mw=float(row.get("grid_capacity_limit_mw", 10.0)),
            is_weekend=int(dt_val.weekday() >= 5),
        )

    def step_anomalies(self, v_idx: int) -> List[str]:
        return self.agent.detect_anomalies(self.grid_telemetry(v_idx))

    def step_price(self, v_idx: int) -> float:
        return float(self.ring_row(v_idx)["day_ahead_price_eur"])

    def predict_step(self, v_idx: int) -> Tuple[float, float, Dict[str, float], float]:
        if v_idx not in self.pred_cache:
            grid = self.grid_telemetry(v_idx)
            surplus_pred, conf, shap_dict = self.agent.estimate(grid)
            inj_pred = self.agent.estimate_injection(grid)
            self.pred_cache[v_idx] = (surplus_pred, conf, shap_dict, inj_pred)
        return self.pred_cache[v_idx]

    def _surplus_trigger(self, step: int) -> bool:
        return (
            self.predict_step(step)[0] > self.agent.MIN_SURPLUS_THRESHOLD_MW
            and self.step_price(step) < self.agent.MAX_PZU_PRICE_EUR
        )

    def step_uncertain(self, step: int) -> bool:
        return self._surplus_trigger(step) and bool(self.step_anomalies(step))

    def step_qualifies(self, step: int) -> bool:
        return (
            self._surplus_trigger(step)
            and not self.step_anomalies(step)
            and self.predict_step(step)[1] >= self.agent.CONFIDENCE_THRESHOLD
        )

    def step_escalated(self, step: int) -> bool:
        return (
            self._surplus_trigger(step)
            and not self.step_anomalies(step)
            and self.predict_step(step)[1] < self.agent.CONFIDENCE_THRESHOLD
        )

    def delivered_mw(self, step: int) -> float:
        return round(sum(self.deals.get(step, {}).values()), 3)

    def remaining_surplus(self, step: int) -> float:
        return max(0.0, self.predict_step(step)[0] - self.delivered_mw(step))

    def node_open_for_step(self, step: int, node_id: str) -> bool:
        return (
            node_id not in self.deals.get(step, {})
            and node_id not in self.refused.get(step, set())
        )

    def datacenter_snapshot(self, step: int, node_id: str) -> Optional[Dict[str, Any]]:
        feed_row = self.ring_row(step)
        timestamp = pd.Timestamp(feed_row["dt"])
        snapshot = self.dc_lookup.get((timestamp, node_id))
        if snapshot is None:
            snapshot = self.dc_hour_lookup.get((int(feed_row["hour"]), node_id))
        return dict(snapshot) if snapshot is not None else None

    def delivered_to_node(self, step: int, node_id: str) -> float:
        return float(self.deals.get(step, {}).get(node_id, 0.0))

    def delivered_temperature(self, step: int, node_id: str) -> float:
        snapshot = self.datacenter_snapshot(step, node_id)
        if snapshot is None:
            return 22.0
        return float(snapshot["temp_servere_c"]) + self.delivered_to_node(step, node_id) * TEMP_RISE_PER_MW_C

    def node_thermal_limit(self, snapshot: Dict[str, Any]) -> float:
        return min(float(snapshot["limita_termica_c"]), self.hvac_limit_c)

    def thermal_rejected(self, step: int, node_id: str) -> bool:
        snapshot = self.datacenter_snapshot(step, node_id)
        if snapshot is None:
            return False
        return self.delivered_temperature(step, node_id) > self.node_thermal_limit(snapshot)

    def node_free_capacity(self, step: int, node_id: str) -> float:
        snapshot = self.datacenter_snapshot(step, node_id)
        if snapshot is None:
            return 0.0
        temp = self.delivered_temperature(step, node_id)
        free_cap = max(0.0, float(snapshot["capacitate_libera_mw"]) - self.delivered_to_node(step, node_id))
        thermal_cap = max(0.0, (self.node_thermal_limit(snapshot) - temp) / TEMP_RISE_PER_MW_C)
        return min(free_cap, thermal_cap)

    def offer_capacity(self, step: int, node_id: str, remaining_mw: float) -> float:
        if self.thermal_rejected(step, node_id):
            return 0.0
        return max(0.0, min(float(remaining_mw), self.node_free_capacity(step, node_id)))

    def node_ranking(self, step: int) -> List[Dict[str, Any]]:
        basis_mw = self.remaining_surplus(step) if self.remaining_surplus(step) > 0.05 else self.predict_step(step)[0]
        candidates = []
        for node in self.datacenters:
            snapshot = self.datacenter_snapshot(step, node["id"])
            if snapshot is None:
                continue
            hist = self.dc_history[node["id"]]
            candidates.append(
                {
                    "id": node["id"],
                    "free_capacity_mw": self.node_free_capacity(step, node["id"]),
                    "temp_c": self.delivered_temperature(step, node["id"]),
                    "thermal_limit_c": self.node_thermal_limit(snapshot),
                    "offers_sent": hist["offers_sent"],
                    "offers_accepted": hist["offers_accepted"],
                }
            )
        return rank_datacenters(basis_mw, candidates, hvac_max_temp_c=self.hvac_limit_c)

    def build_offer(self, steps: List[int]) -> Tuple[Optional[Dict[str, Any]], bool]:
        pending_steps = [s for s in steps if self.step_qualifies(s) and self.remaining_surplus(s) > 0.05]
        if not pending_steps:
            return None, False
        for ranked in self.node_ranking(pending_steps[0]):
            if not ranked["eligible"]:
                continue
            node = self.dc_by_id[ranked["id"]]
            per_step: Dict[int, float] = {}
            for s in pending_steps:
                if self.node_open_for_step(s, node["id"]):
                    mw = round(self.offer_capacity(s, node["id"], self.remaining_surplus(s)), 2)
                    if mw >= 0.05:
                        per_step[s] = mw
            if per_step:
                return {
                    "node_id": node["id"],
                    "node_name": node["name"],
                    "node_short": node["short"],
                    "node_color": node["color"],
                    "per_step": {str(k): v for k, v in per_step.items()},
                    "steps": list(per_step.keys()),
                    "total_mwh": round(sum(per_step.values()), 2),
                    "match_score": ranked["match_score"],
                    "price_eur_mwh": FIXED_PRICE_EUR,
                    "reference_price_eur_mwh": REFERENCE_TARIFF_EUR,
                    "buyer_savings_eur": round(sum(per_step.values()) * (REFERENCE_TARIFF_EUR - FIXED_PRICE_EUR), 2),
                    "ppc_revenue_eur": round(sum(per_step.values()) * FIXED_PRICE_EUR, 2),
                }, False
        return None, True

    def accept_offer(self, node_id: str, per_step: Dict[int, float]) -> str:
        for step, mw in per_step.items():
            self.deals.setdefault(int(step), {})[node_id] = float(mw)
        hist = self.dc_history[node_id]
        hist["offers_sent"] += 1
        hist["offers_accepted"] += 1
        total = sum(per_step.values())
        node_label = self.dc_by_id.get(node_id, {}).get("short", node_id)
        return f"Tranzacție aprobată: {node_label} trece în COMPUTING cu {total:.2f} MWh la {FIXED_PRICE_EUR:.0f} €/MWh."

    def refuse_offer(self, node_id: str, steps: List[int]) -> str:
        for step in steps:
            self.refused.setdefault(int(step), set()).add(node_id)
        self.dc_history[node_id]["offers_sent"] += 1
        node_label = self.dc_by_id.get(node_id, {}).get("short", node_id)
        return f"{node_label} a refuzat. Tariful rămâne {FIXED_PRICE_EUR:.0f} €/MWh — oferta sare la următorul nod."

    def reset_offers(self, steps: List[int]) -> str:
        for step in steps:
            s_int = int(step)
            for node_id in self.deals.pop(s_int, {}):
                hist = self.dc_history[node_id]
                hist["offers_sent"] = max(0, hist["offers_sent"] - 1)
                hist["offers_accepted"] = max(0, hist["offers_accepted"] - 1)
            for node_id in self.refused.pop(s_int, set()):
                hist = self.dc_history[node_id]
                hist["offers_sent"] = max(0, hist["offers_sent"] - 1)
        return "Ofertele pentru interval au fost resetate (istoricul DC restaurat)."

    def get_state(self, step: Optional[int] = None, cycle: int = 0) -> Dict[str, Any]:
        cycle_start_dt, n_rows, cycle_offset_hours = self.month_window(cycle)
        cur_step = self.default_step if step is None else int(step)
        cur_step = int(min(max(0, cur_step), n_rows - 1))
        T = cycle_offset_hours + cur_step

        row_T = self.ring_row(T)
        dt_T = self.ring_dt(T)
        surplus_T, conf_T, shap_T, pred_inj_T = self.predict_step(T)

        delivered_T = self.delivered_mw(T)
        revenue_T = round(delivered_T * FIXED_PRICE_EUR, 2)
        is_dispatched = delivered_T > 0.01

        cumulative_profit = round(
            sum(self.delivered_mw(s) * FIXED_PRICE_EUR for s in self.deals if s <= T), 2
        )
        cumulative_mwh = round(
            sum(self.delivered_mw(s) for s in self.deals if s <= T), 2
        )

        raw_injection_T = self.net_injection_mw(T)
        actual_injection_T = round(max(0.0, raw_injection_T - delivered_T), 3)
        grid_limit_mw = float(row_T.get("grid_capacity_limit_mw", 10.0))
        actual_surplus_T = round(
            max(0.0, actual_injection_T - grid_limit_mw) if is_dispatched else surplus_T, 3
        )

        day_hist_errors = [
            abs(self.net_injection_mw(i) - self.predict_step(i)[3])
            for i in range(T - HISTORY_HOURS, T + 1)
            if 7 <= int(self.ring_row(i)["hour"]) <= 18
        ]
        live_window_mae = round(float(np.mean(day_hist_errors)) if day_hist_errors else 0.42, 2)

        future_idx = [T + k for k in range(1, LOOKAHEAD_HOURS + 1)]
        upcoming_spikes = [i for i in future_idx if self.step_qualifies(i)]
        has_upcoming_spike = len(upcoming_spikes) > 0

        if self.step_qualifies(T):
            offer_steps = [T]
            offer_mode = "ACUM"
            lead_hours = 0
        elif has_upcoming_spike:
            offer_steps = upcoming_spikes
            offer_mode = "ÎN AVANS"
            lead_hours = upcoming_spikes[0] - T
        else:
            offer_steps = []
            offer_mode = "FĂRĂ SURPLUS"
            lead_hours = 0

        offer, offers_exhausted = self.build_offer(offer_steps)
        table_steps = offer_steps if offer_steps else [T]
        pool_status = POOL_ALL_DCS_EXHAUSTED if (offer_steps and offers_exhausted) else "ACTIVE"

        uncertain_steps = [i for i in [T] + future_idx if self.step_uncertain(i)]
        escalated_steps = [i for i in [T] + future_idx if self.step_escalated(i)]

        # Banner de stare SCADA
        if uncertain_steps:
            u_idx = uncertain_steps[0]
            anom = ", ".join(self.step_anomalies(u_idx))
            banner = {
                "level": "amber",
                "text": f"STATUS: UNCERTAIN ({self.ring_dt(u_idx).strftime('%H:%M')}) — senzorii raportează valori > 3σ ({anom}). Ofertarea automată este oprită; dispecerul uman este notificat.",
            }
        elif escalated_steps and not offer_steps:
            e_idx = escalated_steps[0]
            _, c_e, _, _ = self.predict_step(e_idx)
            banner = {
                "level": "amber",
                "text": f"ESCALATE_TO_HUMAN ({self.ring_dt(e_idx).strftime('%H:%M')}) — încrederea modelului este {c_e * 100:.0f}% (sub pragul de {self.agent.CONFIDENCE_THRESHOLD * 100:.0f}%). Decizia este transferată operatorului PPC.",
            }
        elif is_dispatched:
            banner = {
                "level": "green",
                "text": f"SURPLUS DISPECERIZAT: {delivered_T:.2f} MW redirecționați către Data Centre la {FIXED_PRICE_EUR:.0f} €/MWh · Curentul a scăzut de la {raw_injection_T:.2f} MW la {actual_injection_T:.2f} MW · Profit PPC: +{revenue_T:,.2f} €",
            }
        elif pool_status == POOL_ALL_DCS_EXHAUSTED:
            banner = {
                "level": "red",
                "text": f"ALL_DCS_EXHAUSTED: toate data centerele eligibile au refuzat oferta la {FIXED_PRICE_EUR:.0f} €/MWh. Se solicită intervenția dispeceratului; riscul de curtailment rămâne.",
            }
        elif has_upcoming_spike and not self.step_qualifies(T):
            start_str = self.ring_dt(upcoming_spikes[0]).strftime("%H:%M")
            end_str = self.ring_dt(upcoming_spikes[-1] + 1).strftime("%H:%M")
            win_mwh = round(sum(self.predict_step(si)[0] for si in upcoming_spikes), 2)
            banner = {
                "level": "amber",
                "text": f"ALERTĂ TIMPURIE AI (T+{upcoming_spikes[0] - T}h): Modelul anticipează un vârf de +{win_mwh:.1f} MW peste limita liniei în intervalul {start_str} – {end_str}. Oferta poate fi contractată în avans.",
            }
        elif self.step_qualifies(T):
            banner = {
                "level": "red",
                "text": f"CONGESTIE ÎN CURS ({dt_T.strftime('%H:%M')}): Producția depășește limita de {grid_limit_mw:.0f} MW cu +{surplus_T:.2f} MW (PZU {self.step_price(T):.1f} €/MWh). Trimite oferta către Data Centre!",
            }
        else:
            banner = {
                "level": "green",
                "text": f"REȚEA ÎN PARAMETRI NORMALI: Producția este sub limita de {grid_limit_mw:.0f} MW în următoarele {LOOKAHEAD_HOURS} ore.",
            }

        # Clasament și telemetrie noduri Data Center
        ranking_list = self.node_ranking(table_steps[0])
        ranking_by_id = {r["id"]: (idx + 1, r) for idx, r in enumerate(ranking_list)}

        dc_cards: List[Dict[str, Any]] = []
        for rank_pos, ranked in enumerate(ranking_list, start=1):
            node = self.dc_by_id[ranked["id"]]
            nid = node["id"]
            snap = self.datacenter_snapshot(table_steps[0], nid) or {}
            temp_now = round(self.delivered_temperature(table_steps[0], nid), 1)
            temp_limit = round(self.node_thermal_limit(snap), 1) if snap else 26.0
            free_cap = round(self.node_free_capacity(table_steps[0], nid), 2)
            alloc_mwh = round(sum(self.deals.get(s, {}).get(nid, 0.0) for s in table_steps), 2)

            if any(nid in self.deals.get(s, {}) for s in table_steps):
                status = "ACCEPTED"
            elif any(self.thermal_rejected(s, nid) for s in table_steps):
                status = "REJECTED_THERMAL_SAFETY"
            elif any(nid in self.refused.get(s, set()) for s in table_steps):
                status = "REFUSED"
            elif offer is not None and offer["node_id"] == nid:
                status = "OFFER_PENDING"
            elif any(self.step_uncertain(s) for s in table_steps):
                status = "UNCERTAIN"
            elif any(self.step_escalated(s) for s in table_steps):
                status = "ESCALATE_TO_HUMAN"
            elif any(self.step_qualifies(s) and self.remaining_surplus(s) > 0.05 for s in table_steps):
                status = "IN_QUEUE"
            else:
                status = "STANDBY"

            hist = self.dc_history[nid]
            dc_cards.append(
                {
                    "id": nid,
                    "name": node["name"],
                    "short": node["short"],
                    "city": node.get("city", node["short"]),
                    "color": node["color"],
                    "lat": node.get("lat", 44.43),
                    "lon": node.get("lon", 26.05),
                    "waypoints": node.get("waypoints", []),
                    "workload": node.get("workload", "AI Compute Cluster"),
                    "rank": rank_pos,
                    "match_score": ranked["match_score"],
                    "volume_fit": ranked["volume_fit"],
                    "acceptance_rate": ranked["acceptance_rate"],
                    "thermal_penalty": ranked["thermal_penalty"],
                    "eligible": ranked["eligible"],
                    "temp_c": temp_now,
                    "thermal_limit_c": temp_limit,
                    "free_capacity_mw": free_cap,
                    "base_load_mw": round(float(snap.get("putere_baza_mw", 4.0)), 2),
                    "max_capacity_mw": round(float(snap.get("capacitate_max_mw", 10.0)), 2),
                    "allocated_mwh": alloc_mwh,
                    "offers_sent": hist["offers_sent"],
                    "offers_accepted": hist["offers_accepted"],
                    "status": status,
                }
            )

        # Seria de timp 24h istoric + 2h predicție pentru graficul SCADA
        hist_start = T - HISTORY_HOURS
        chart_points: List[Dict[str, Any]] = []
        for idx in range(hist_start, T + LOOKAHEAD_HOURS + 1):
            dt_pt = self.ring_dt(idx)
            s_pred, c_pred, _, inj_pred = self.predict_step(idx)
            deliv = self.delivered_mw(idx)
            is_future = idx > T
            real_inj = None if is_future else self.net_injection_mw(idx)
            after_dispatch = (
                max(0.0, (real_inj if real_inj is not None else inj_pred) - deliv)
                if deliv > 0.01
                else None
            )
            chart_points.append(
                {
                    "v_idx": idx,
                    "timestamp": dt_pt.strftime("%Y-%m-%d %H:%M"),
                    "short_time": dt_pt.strftime("%H:%M"),
                    "is_current": idx == T,
                    "is_future": is_future,
                    "real_injection_mw": real_inj,
                    "predicted_injection_mw": round(inj_pred, 3),
                    "after_dispatch_mw": round(after_dispatch, 3) if after_dispatch is not None else None,
                    "predicted_surplus_mw": round(s_pred, 3),
                    "delivered_mw": deliv,
                    "grid_limit_mw": grid_limit_mw,
                    "confidence": round(c_pred, 3),
                }
            )

        # Explicație în limbaj uman (AI Copilot Reasoning)
        if offer is not None:
            copilot_reasoning = (
                f"Recomandare activă: {offer['node_short']} (Match Score {offer['match_score'] * 100:.0f}%). "
                f"Alocarea a {offer['total_mwh']:.2f} MWh la {FIXED_PRICE_EUR:.0f} €/MWh menține linia PPC sub "
                f"limita de {grid_limit_mw:.0f} MW și economisește {offer['buyer_savings_eur']:,.0f} € față de tariful standard."
            )
        elif is_dispatched:
            copilot_reasoning = (
                f"Tranzacție activă: {delivered_T:.2f} MW sunt absorbiți de centrele de date. "
                f"Curentul în linia PPC a scăzut la {actual_injection_T:.2f} MW (sub plafonul de {grid_limit_mw:.0f} MW)."
            )
        else:
            copilot_reasoning = (
                f"Monitorizare continuă: la ora {dt_T.strftime('%H:%M')} injecția este {raw_injection_T:.2f} MW "
                f"(plafon linie {grid_limit_mw:.0f} MW, PZU {self.step_price(T):.1f} €/MWh)."
            )

        return {
            "timeline": {
                "current_step": cur_step,
                "ring_cycle": cycle,
                "virtual_index": T,
                "hours_in_month": n_rows,
                "month_label": cycle_start_dt.strftime("%Y-%m"),
                "current_timestamp": dt_T.strftime("%Y-%m-%d %H:%M"),
                "current_hour_str": dt_T.strftime("%H:%M"),
                "slider_start_label": self.ring_dt(cycle_offset_hours).strftime("%Y-%m-%d %H:%M"),
                "slider_end_label": self.ring_dt(cycle_offset_hours + n_rows - 1).strftime("%Y-%m-%d %H:%M"),
                "train_rows": self.n_train_rows,
            },
            "kpis": {
                "raw_injection_mw": raw_injection_T,
                "actual_injection_mw": actual_injection_T,
                "predicted_injection_mw": round(pred_inj_T, 2),
                "surplus_mw": round(surplus_T, 2),
                "actual_surplus_mw": round(actual_surplus_T, 2),
                "grid_limit_mw": grid_limit_mw,
                "confidence": round(conf_T, 3),
                "day_ahead_price_eur": round(self.step_price(T), 2),
                "solar_irradiance_wm2": round(float(row_T["solar_irradiance_wm2"]), 1),
                "ambient_temp_c": round(float(row_T["ambient_temp_celsius"]), 1),
                "delivered_mw": delivered_T,
                "revenue_eur": revenue_T,
                "cumulative_profit_eur": cumulative_profit,
                "cumulative_mwh": cumulative_mwh,
                "live_window_mae": live_window_mae,
                "is_dispatched": is_dispatched,
                "ai_tokens_millions": round(max(delivered_T, cumulative_mwh) * 4.1, 1),
                "co2_saved_tons": round(max(delivered_T, cumulative_mwh) * 0.42, 2),
            },
            "banner": banner,
            "explainability": {
                "top_shap_features": shap_T,
                "anomalies": self.step_anomalies(T),
                "copilot_reasoning": copilot_reasoning,
                "min_surplus_threshold_mw": self.agent.MIN_SURPLUS_THRESHOLD_MW,
                "max_pzu_price_eur": self.agent.MAX_PZU_PRICE_EUR,
                "confidence_threshold": self.agent.CONFIDENCE_THRESHOLD,
            },
            "offer_state": {
                "mode": offer_mode,
                "lead_hours": lead_hours,
                "offer_steps": offer_steps,
                "table_steps": table_steps,
                "offer": offer,
                "offers_exhausted": offers_exhausted,
                "pool_status": pool_status,
                "fixed_price_eur": FIXED_PRICE_EUR,
                "reference_tariff_eur": REFERENCE_TARIFF_EUR,
            },
            "datacenters": dc_cards,
            "chart_points": chart_points,
            "y_axis_max": self.y_axis_max,
            "map_topology": {
                "solar_park": PPC_SOLAR_PARK,
                "backbone_400kv": GRID_400KV_BACKBONE,
            },
        }
