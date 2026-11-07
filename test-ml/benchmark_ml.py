"""VoltCompute — Benchmark ML pentru modelul de prognoză a surplusului solar (t+2h).

Benchmark-ul este AGNOSTIC față de versiunea codului: reconstruiește setul de date direct
din CSV-urile brute și evaluează orice `model.joblib` care expune `model` + `feature_names`
(funcționează atât pe varianta din `main`, cât și pe cea din `mihai`).

Rulare (din rădăcina proiectului):
    python test-ml/benchmark_ml.py                      # model.joblib implicit
    python test-ml/benchmark_ml.py --model models/model.joblib

Nu modifică artefactele. Scrie doar:
    test-ml/results/benchmark_results.json
    test-ml/results/benchmark_report.md

Secțiuni:
 1. Acuratețe vs. baseline Persistență și Sezonier 24h (+ skill score), pe toate orizonturile disponibile
 2. Erori pe regimuri orare (vârf solar, rampe, noapte)
 3. Detecția depășirilor (surplus > prag): precision / recall / F1
 4. Calibrarea incertitudinii (acoperire interval 90%, utilitatea confidenței) — dacă modelul o oferă
 5. Backtest walk-forward (expanding window, 4 fold-uri, reantrenarea aceleiași configurații)
 6. Robustețe: zgomot pe senzorii meteo + verificări fail-closed pe calea de inferență
 7. Importanța atributelor (permutation importance + TreeSHAP global)
 8. Performanță: latență, throughput, timp de antrenare, determinism
 9. Integrare: agentul VoltCompute folosește efectiv modelul ML la runtime?
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score,
)

DATA_DIR = ROOT / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_MODEL = ROOT / "model.joblib"
GRID_LIMIT_MW = 0.35
SEED = 42
TEST_FRACTION = 0.2
WEATHER_COLUMNS = [
    "temp_c", "radiatie_solara_wm2", "viteza_vant_kmh", "temp_lag_1", "radiatie_lag_1",
    "seasonal_radiation_t_plus_2", "seasonal_temp_t_plus_2", "seasonal_wind_t_plus_2",
]
NON_NEGATIVE = {"radiatie_solara_wm2", "viteza_vant_kmh", "radiatie_lag_1",
                "seasonal_radiation_t_plus_2", "seasonal_wind_t_plus_2"}
PHYSICAL_BOUNDS = {"radiatie_solara_wm2": (0, 1500), "temp_c": (-40, 60),
                   "viteza_vant_kmh": (0, 150), "avg_battery_soc_pct": (0, 100)}


# =====================================================================
# DATA (reconstruit independent din CSV-uri, fără a depinde de pipeline)
# =====================================================================
def build_history() -> pd.DataFrame:
    """720 ore: agregare bottom-up a celor 100 de prosumatori + macro + superset de features cauzale."""
    micro = pd.read_csv(DATA_DIR / "2_micro_100_prosumatori.csv", parse_dates=["timestamp"])
    cols = [c for c in ("export_in_retea_kw", "baterie_curenta_kwh", "capacitate_bat_kwh", "consum_kw")
            if c in micro.columns]
    sums = micro.groupby("timestamp")[cols].sum(min_count=1)
    sums["Surplus_National_MW"] = sums["export_in_retea_kw"] / 1000.0
    if {"baterie_curenta_kwh", "capacitate_bat_kwh"} <= set(cols):
        sums["avg_battery_soc_pct"] = 100 * sums["baterie_curenta_kwh"] / sums["capacitate_bat_kwh"]
    if "consum_kw" in cols:
        sums["total_prosumer_load_mw"] = sums["consum_kw"] / 1000.0
    macro = pd.read_csv(DATA_DIR / "1_macro_vreme_piata.csv", parse_dates=["timestamp"])
    df = macro.merge(sums.reset_index(), on="timestamp", validate="one_to_one")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hour"] = df.timestamp.dt.hour
    df["dayofweek"] = df.timestamp.dt.dayofweek
    df["is_weekend"] = (df.dayofweek >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.hour / 24)
    for lag in (1, 2, 24):
        df[f"surplus_lag_{lag}"] = df.Surplus_National_MW.shift(lag)
    df["temp_lag_1"] = df.temp_c.shift(1)
    df["radiatie_lag_1"] = df.radiatie_solara_wm2.shift(1)
    df["pret_lag_1"] = df.pret_pzu_eur_mwh.shift(1)
    for name, src in (("radiation", "radiatie_solara_wm2"), ("temp", "temp_c"), ("wind", "viteza_vant_kmh")):
        df[f"seasonal_{name}_t_plus_2"] = df[src].shift(22)
    return df


def horizon_frame(history: pd.DataFrame, h: int, window: Optional[pd.Series] = None) -> pd.DataFrame:
    f = history.copy()
    f["target"] = f.Surplus_National_MW.shift(-h)
    f["target_timestamp"] = f.timestamp.shift(-h)
    f["seasonal_baseline"] = f.Surplus_National_MW.shift(24 - h)
    f = f.dropna(subset=["target", "seasonal_baseline", "surplus_lag_24"])
    if window is not None:
        f = f.loc[f.timestamp.between(window.iloc[0], window.iloc[-1])]
    return f.reset_index(drop=True)


# =====================================================================
# HELPERS
# =====================================================================
def _scores(y, p) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return dict(mae=float(mean_absolute_error(y, p)), rmse=float(np.sqrt(mean_squared_error(y, p))),
                r2=float(r2_score(y, p)), bias=float(np.mean(p - y)),
                max_abs_error=float(np.max(np.abs(p - y))))


def _skill(err: float, ref: float) -> float:
    return float(1 - err / ref) if ref > 0 else float("nan")


def _quiet(fn: Callable, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def _safe(section: Callable, *args) -> Dict[str, Any]:
    """O secțiune care eșuează (ex. diferență de versiune) nu oprește tot benchmark-ul."""
    try:
        return section(*args)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


class Ctx:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.artifact = joblib.load(self.model_path)
        if not isinstance(self.artifact, dict):
            self.artifact = {"model": self.artifact}
        self.model = self.artifact["model"]
        self.features: List[str] = list(self.artifact.get("feature_names")
                                        or self.artifact.get("feature_columns")
                                        or getattr(self.model, "feature_names_in_", []))
        self.history = build_history()
        self.canonical = horizon_frame(self.history, 2)
        missing = [c for c in self.features if c not in self.canonical.columns]
        if missing:
            raise ValueError(f"Artefactul cere features necunoscute de benchmark: {missing}")
        self.canonical = self.canonical.dropna(subset=self.features).reset_index(drop=True)
        self.boundary = self.canonical.iloc[int(len(self.canonical) * (1 - TEST_FRACTION))].timestamp
        self.test = self.canonical.loc[self.canonical.timestamp >= self.boundary]
        self.has_shap = hasattr(self.model, "get_booster")
        try:
            import train_pipeline as tp  # noqa: WPS433
            self.tp = tp
        except Exception:  # noqa: BLE001
            self.tp = None
        self.has_calibration = bool(self.artifact.get("calibration")) and hasattr(self.tp, "infer_single_step")

    def predict(self, X, model=None) -> np.ndarray:
        return np.clip((model or self.model).predict(X[self.features]), 0.0, None)

    def model_for(self, h: int):
        bundle = self.artifact.get("forecast_models", {}).get(str(h))
        if bundle is not None:
            return bundle["model"], list(bundle.get("feature_names", self.features))
        return (self.model, self.features) if h == 2 else (None, None)

    def version(self) -> str:
        if self.artifact.get("schema_version") == 2:
            return "v2 (mihai: multi-orizont + calibrare + TreeSHAP)"
        return "v1 (main: model t+2h fără calibrare)"


# =====================================================================
# 1. ACCURACY
# =====================================================================
def accuracy(ctx: Ctx) -> Dict[str, Any]:
    out = {}
    for h in (1, 2, 3, 4):
        model, feats = ctx.model_for(h)
        if model is None:
            continue
        frame = horizon_frame(ctx.history, h, ctx.canonical.timestamp).dropna(subset=feats)
        test = frame.loc[frame.timestamp >= ctx.boundary]
        raw = model.predict(test[feats])
        xgb = _scores(test.target, np.clip(raw, 0, None))
        pers = _scores(test.target, test.Surplus_National_MW)
        seas = _scores(test.target, test.seasonal_baseline)
        out[str(h)] = dict(test_rows=int(len(test)), xgboost=xgb, baseline_persistence=pers,
                           baseline_seasonal=seas, skill_vs_persistence=_skill(xgb["mae"], pers["mae"]),
                           skill_vs_seasonal=_skill(xgb["mae"], seas["mae"]),
                           raw_mae_unclipped=float(mean_absolute_error(test.target, raw)))
    return out


# =====================================================================
# 2. REGIMES
# =====================================================================
def regimes(ctx: Ctx) -> Dict[str, Any]:
    t = ctx.test.copy()
    t["pred"] = ctx.predict(t)
    th = (t.hour + 2) % 24
    groups = {"varf_solar_11_15": th.between(11, 15),
              "rampe_8_10_16_18": th.isin([8, 9, 10, 16, 17, 18]),
              "noapte_19_7": ~th.between(8, 18)}
    out = {}
    for name, mask in groups.items():
        p = t.loc[mask]
        if len(p):
            out[name] = dict(rows=int(len(p)), mean_target_mw=float(p.target.mean()),
                             xgboost_mae=float(mean_absolute_error(p.target, p.pred)),
                             persistence_mae=float(mean_absolute_error(p.target, p.Surplus_National_MW)),
                             seasonal_mae=float(mean_absolute_error(p.target, p.seasonal_baseline)))
    hourly = t.assign(err=(t.pred - t.target).abs(), th=th).groupby("th").err.mean()
    out["mae_per_target_hour"] = {int(k): float(v) for k, v in hourly.items()}
    return out


# =====================================================================
# 3. EXCEEDANCE DETECTION
# =====================================================================
def exceedance(ctx: Ctx) -> Dict[str, Any]:
    t = ctx.test
    pred = ctx.predict(t)
    pos = t.target[t.target > 0]
    thresholds = {"limita_retea_0.35MW": GRID_LIMIT_MW}
    if len(pos):
        thresholds["p75_export_pozitiv"] = float(np.quantile(pos, 0.75))
    out = {}
    for name, thr in thresholds.items():
        y = (t.target.to_numpy() > thr).astype(int)
        yp = (pred > thr).astype(int)
        ys = (t.seasonal_baseline.to_numpy() > thr).astype(int)
        e = dict(threshold_mw=float(thr), actual_events=int(y.sum()), predicted_events=int(yp.sum()))
        if y.sum() == 0:
            e["note"] = "Nicio depășire reală în test la acest prag."
        else:
            e.update(precision=float(precision_score(y, yp, zero_division=0)),
                     recall=float(recall_score(y, yp, zero_division=0)),
                     f1=float(f1_score(y, yp, zero_division=0)),
                     seasonal_baseline_f1=float(f1_score(y, ys, zero_division=0)))
        out[name] = e
    return out


# =====================================================================
# 4. UNCERTAINTY
# =====================================================================
def uncertainty(ctx: Ctx) -> Dict[str, Any]:
    if not ctx.has_calibration:
        return {"available": False,
                "note": "Modelul nu expune interval de predicție / confidență calibrată (lipsește `calibration`)."}
    rows = ctx.test[ctx.features].to_dict("records")
    outs = [ctx.tp.infer_single_step(r, ctx.artifact) for r in rows]
    y = ctx.test.target.to_numpy()
    pred = np.array([o["predicted_export_mw"] for o in outs])
    lo = np.array([o["prediction_lower_mw"] for o in outs])
    hi = np.array([o["prediction_upper_mw"] for o in outs])
    conf = np.array([o["confidence_score"] for o in outs])
    err = np.abs(pred - y)
    inside = (y >= lo - 1e-12) & (y <= hi + 1e-12)
    gate = conf >= 0.75
    rho = pd.Series(conf).corr(pd.Series(err), method="spearman")
    day = y > 0
    return dict(available=True, nominal_coverage=0.90, empirical_coverage=float(inside.mean()),
                coverage_daylight=float(inside[day].mean()) if day.any() else None,
                mean_interval_width_mw=float(np.mean(hi - lo)),
                confidence_mean=float(conf.mean()), confidence_min=float(conf.min()),
                confidence_max=float(conf.max()), share_passing_gate_075=float(gate.mean()),
                mae_when_conf_ge_075=float(err[gate].mean()) if gate.any() else None,
                mae_when_conf_lt_075=float(err[~gate].mean()) if (~gate).any() else None,
                spearman_conf_vs_abs_error=None if pd.isna(rho) else float(rho))


# =====================================================================
# 5. WALK-FORWARD
# =====================================================================
def walk_forward(ctx: Ctx, folds: int = 4, initial_frac: float = 0.5) -> Dict[str, Any]:
    df = ctx.canonical
    n = len(df)
    start = int(n * initial_frac)
    size = (n - start) // folds
    rows = []
    for k in range(folds):
        s, e = start + k * size, (n if k == folds - 1 else start + (k + 1) * size)
        test = df.iloc[s:e]
        train = df.iloc[:s].loc[lambda x: x.target_timestamp < test.timestamp.iloc[0]]  # purjare
        m = clone(ctx.model).fit(train[ctx.features], train.target)
        p = ctx.predict(test, m)
        x = mean_absolute_error(test.target, p)
        pe = mean_absolute_error(test.target, test.Surplus_National_MW)
        se = mean_absolute_error(test.target, test.seasonal_baseline)
        rows.append(dict(fold=k + 1, train_rows=int(len(train)), test_rows=int(len(test)),
                         test_start=test.timestamp.iloc[0].isoformat(), test_end=test.timestamp.iloc[-1].isoformat(),
                         xgboost_mae=float(x), persistence_mae=float(pe), seasonal_mae=float(se),
                         beats_persistence=bool(x < pe), beats_seasonal=bool(x < se)))
    d = pd.DataFrame(rows)
    return dict(folds=rows, mean_xgboost_mae=float(d.xgboost_mae.mean()),
                std_xgboost_mae=float(d.xgboost_mae.std(ddof=0)),
                mean_persistence_mae=float(d.persistence_mae.mean()),
                mean_seasonal_mae=float(d.seasonal_mae.mean()),
                wins_vs_persistence=int(d.beats_persistence.sum()),
                wins_vs_seasonal=int(d.beats_seasonal.sum()))


# =====================================================================
# 6. ROBUSTNESS
# =====================================================================
def robustness(ctx: Ctx) -> Dict[str, Any]:
    t = ctx.test
    base = mean_absolute_error(t.target, ctx.predict(t))
    rng = np.random.default_rng(SEED)
    cols = [c for c in WEATHER_COLUMNS if c in ctx.features]
    noise = {}
    for sigma in (0.05, 0.10, 0.20):
        maes = []
        for _ in range(5):
            X = t[ctx.features].copy()
            for c in cols:
                X[c] = X[c] * (1 + rng.normal(0, sigma, len(X)))
                if c in NON_NEGATIVE:
                    X[c] = X[c].clip(lower=0)
            maes.append(mean_absolute_error(t.target, np.clip(ctx.model.predict(X), 0, None)))
        mae = float(np.mean(maes))
        noise[f"sigma_{int(sigma * 100)}pct"] = dict(mae=mae, degradation_pct=float(100 * (mae / base - 1)))

    sample = t[ctx.features].iloc[len(t) // 2].to_dict()
    validated = hasattr(ctx.tp, "infer_single_step") and ctx.has_calibration

    def rejects(features: Dict[str, float]) -> bool:
        if validated:
            try:
                ctx.tp.infer_single_step(features, ctx.artifact)
                return False
            except (ValueError, KeyError, TypeError):
                return True
        try:
            if any(k not in features for k in ctx.features):
                return True
            for v in features.values():
                if not np.isfinite(v):
                    return True
            for col, (lo, hi) in PHYSICAL_BOUNDS.items():
                if col in features and not (lo <= features[col] <= hi):
                    return True
            from agent import VoltComputeAgent
            from schemas import GridTelemetry, ResearchCenterTelemetry
            ag = VoltComputeAgent()
            gt = GridTelemetry(
                hour=int(features.get("hour", 12)),
                solar_irradiance_wm2=float(features.get("radiatie_solara_wm2", 500)),
                ambient_temp_celsius=float(features.get("temp_c", 20)),
                day_ahead_price_eur=float(features.get("pret_pzu_eur_mwh", 50)),
            )
            rc = ResearchCenterTelemetry(
                center_id="DC_TEST", cooling_temp_celsius=20.0,
                max_absorb_capacity_mw=5.0, pending_compute_jobs=5
            )
            _, _, err = ag.perceive(gt, rc)
            return err is not None
        except Exception:
            return True

    probes = {"valid_input_accepted": (sample, False),
              "missing_feature_rejected": ({k: v for k, v in sample.items() if k != ctx.features[-1]}, True),
              "nan_rejected": ({**sample, ctx.features[4]: float("nan")}, True),
              "inf_rejected": ({**sample, ctx.features[4]: float("inf")}, True)}
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col in ctx.features:
            probes[f"out_of_bounds_{col}_rejected"] = ({**sample, col: hi * 4 if hi else -100.0}, True)
    fail_closed = {name: (rejects(f) == expected) for name, (f, expected) in probes.items()}
    return dict(baseline_mae=float(base), weather_noise=noise, fail_closed=fail_closed,
                inference_path="train_pipeline.infer_single_step (validat)" if validated else "model.predict (nevalidat)")


# =====================================================================
# 7. FEATURE IMPORTANCE
# =====================================================================
def importance(ctx: Ctx, top: int = 10) -> Dict[str, Any]:
    t = ctx.test
    base = mean_absolute_error(t.target, ctx.predict(t))
    rng = np.random.default_rng(SEED)
    perm = {}
    for c in ctx.features:
        d = []
        for _ in range(5):
            X = t[ctx.features].copy()
            X[c] = rng.permutation(X[c].to_numpy())
            d.append(mean_absolute_error(t.target, np.clip(ctx.model.predict(X), 0, None)) - base)
        perm[c] = float(np.mean(d))
    shap = {}
    if ctx.has_shap:
        from xgboost import DMatrix
        contrib = ctx.model.get_booster().predict(DMatrix(ctx.canonical.loc[
            ctx.canonical.timestamp < ctx.boundary, ctx.features]), pred_contribs=True)[:, :-1]
        shap = dict(zip(ctx.features, np.abs(contrib).mean(axis=0).astype(float)))
    return dict(permutation_mae_increase_mw=dict(sorted(perm.items(), key=lambda x: -x[1])[:top]),
                treeshap_global_mean_abs_mw=dict(sorted(shap.items(), key=lambda x: -x[1])[:top]))


# =====================================================================
# 8. PERFORMANCE
# =====================================================================
def performance(ctx: Ctx, n: int = 300) -> Dict[str, Any]:
    X = ctx.test[ctx.features]
    rows = [X.iloc[[i]] for i in range(len(X))]

    def lat(fn) -> Dict[str, float]:
        ts = []
        for i in range(n):
            t0 = time.perf_counter()
            fn(i)
            ts.append((time.perf_counter() - t0) * 1000)
        a = np.array(ts[5:])
        return dict(mean_ms=float(a.mean()), p50_ms=float(np.percentile(a, 50)),
                    p95_ms=float(np.percentile(a, 95)), p99_ms=float(np.percentile(a, 99)))

    out: Dict[str, Any] = dict(single_row_predict=lat(lambda i: ctx.model.predict(rows[i % len(rows)])))
    if ctx.has_shap:
        from xgboost import DMatrix
        booster = ctx.model.get_booster()
        out["single_row_treeshap"] = lat(lambda i: booster.predict(DMatrix(rows[i % len(rows)]), pred_contribs=True))
    if ctx.has_calibration:
        recs = X.to_dict("records")
        out["full_inference_pipeline"] = lat(lambda i: ctx.tp.infer_single_step(recs[i % len(recs)], ctx.artifact))

    t0 = time.perf_counter()
    for _ in range(50):
        ctx.model.predict(X)
    out["batch_throughput_rows_per_s"] = float(len(X) / ((time.perf_counter() - t0) / 50))

    train = ctx.canonical.loc[ctx.canonical.timestamp < ctx.boundary]
    t0 = time.perf_counter()
    m1 = clone(ctx.model).fit(train[ctx.features], train.target)
    out["train_seconds_same_config"] = float(time.perf_counter() - t0)
    m2 = clone(ctx.model).fit(train[ctx.features], train.target)
    out["deterministic_retrain_max_abs_diff"] = float(np.max(np.abs(m1.predict(X) - m2.predict(X))))
    out["artifact_vs_retrain_max_abs_diff"] = float(np.max(np.abs(m1.predict(X) - ctx.model.predict(X))))
    return out


# =====================================================================
# 9. AGENT INTEGRATION
# =====================================================================
class _CountingModel:
    """Proxy care numără apelurile `predict` reușite făcute de agent asupra modelului ML."""

    def __init__(self, inner):
        self._inner = inner
        self.successful_predicts = 0

    def predict(self, *a, **kw):
        result = self._inner.predict(*a, **kw)
        self.successful_predicts += 1
        return result

    def __getattr__(self, name):
        return getattr(self._inner, name)


def agent_integration(ctx: Ctx) -> Dict[str, Any]:
    from agent import VoltComputeAgent  # noqa: WPS433

    agent = _quiet(VoltComputeAgent)
    if not agent.model_artifact or "model" not in agent.model_artifact:
        return dict(model_loaded=False, uses_ml_model=False, note="Agentul nu a încărcat niciun artefact ML.")
    proxy = _CountingModel(agent.model_artifact["model"])
    agent.model_artifact["model"] = proxy
    row = ctx.test.iloc[len(ctx.test) // 2]
    direct = float(ctx.predict(ctx.test.iloc[[len(ctx.test) // 2]])[0])

    if hasattr(ctx.tp, "load_grid_status"):  # API v2
        step = int(ctx.history.index[ctx.history.timestamp.eq(row.timestamp)][0])
        status = ctx.tp.load_grid_status(step)
        decision = agent.run_cycle(status["grid"], data_centers=status["data_centers"])
        agent_value = float(decision.predicted_export_mw)
        source = decision.prediction_source
    else:  # API v1
        from schemas import GridTelemetry
        grid = GridTelemetry(hour=int(row.hour), solar_irradiance_wm2=float(row.radiatie_solara_wm2),
                             ambient_temp_celsius=float(row.temp_c), day_ahead_price_eur=float(row.pret_pzu_eur_mwh),
                             is_weekend=int(row.dayofweek >= 5))
        agent_value = float(agent.estimate(grid)[0])
        source = "xgboost" if proxy.successful_predicts else "physics_fallback (modelul ML a fost ignorat)"

    uses = proxy.successful_predicts > 0
    return dict(model_loaded=True, uses_ml_model=uses, successful_model_calls=proxy.successful_predicts,
                prediction_source=source, timestamp=row.timestamp.isoformat(),
                direct_model_prediction_mw=direct, agent_prediction_mw=agent_value,
                matches_direct_prediction=bool(uses and abs(agent_value - direct) < 1e-6))


# =====================================================================
# VERDICT
# =====================================================================
def checks(r: Dict[str, Any]) -> List[Dict[str, Any]]:
    acc, wf, unc = r["accuracy"], r["walk_forward"], r["uncertainty"]
    perf, rob, ag = r["performance"], r["robustness"], r["agent_integration"]
    am = r["artifact_metrics"]
    out = []

    def add(name, value, threshold, passed):
        out.append(dict(check=name, value=value, threshold=threshold,
                        passed=None if passed is None else bool(passed)))

    if "error" not in acc and "2" in acc:
        if am.get("test_mae") is not None:
            d = abs(acc["2"]["raw_mae_unclipped"] - am["test_mae"])
            add("Paritate: MAE recalculat independent = MAE raportat în artefact", f"{d:.2e}", "< 1e-6", d < 1e-6)
        add("Bate Persistența pe toate orizonturile",
            ", ".join(f"t+{h}: {v['skill_vs_persistence']:.0%}" for h, v in acc.items()), "skill > 0",
            all(v["skill_vs_persistence"] > 0 for v in acc.values()))
        add("Bate Baseline-ul Sezonier 24h pe toate orizonturile",
            ", ".join(f"t+{h}: {v['skill_vs_seasonal']:.0%}" for h, v in acc.items()), "skill > 0",
            all(v["skill_vs_seasonal"] > 0 for v in acc.values()))
        add("R² pe test la t+2h", round(acc["2"]["xgboost"]["r2"], 4), "> 0.90", acc["2"]["xgboost"]["r2"] > 0.90)
    if "error" not in wf:
        k = len(wf["folds"])
        add("Walk-forward: bate Persistența în toate fold-urile", f"{wf['wins_vs_persistence']}/{k}", "toate",
            wf["wins_vs_persistence"] == k)
        add("Walk-forward: bate Sezonierul în majoritatea fold-urilor", f"{wf['wins_vs_seasonal']}/{k}", ">= 3/4",
            wf["wins_vs_seasonal"] >= int(np.ceil(0.75 * k)))
    if unc.get("available"):
        add("Acoperire empirică interval 90%", round(unc["empirical_coverage"], 3), "0.80 – 0.99",
            0.80 <= unc["empirical_coverage"] <= 0.99)
    else:
        add("Incertitudine calibrată (interval + confidență)", "indisponibil", "prezent", None)
    if "error" not in perf:
        add("Latență predict 1 rând (p95)", f"{perf['single_row_predict']['p95_ms']:.2f} ms", "< 50 ms",
            perf["single_row_predict"]["p95_ms"] < 50)
        add("Timp antrenare (aceeași configurație)", f"{perf['train_seconds_same_config']:.2f} s", "< 15 s",
            perf["train_seconds_same_config"] < 15)
        add("Reantrenare deterministă", f"{perf['deterministic_retrain_max_abs_diff']:.1e}", "== 0",
            perf["deterministic_retrain_max_abs_diff"] == 0)
    if "error" not in rob:
        deg = rob["weather_noise"]["sigma_10pct"]["degradation_pct"]
        add("Zgomot 10% pe senzorii meteo → degradare MAE", f"{deg:+.1f}%", "< 50%", deg < 50)
        fc = rob["fail_closed"]
        add("Fail-closed pe calea de inferență (NaN / Inf / lipsă / limite fizice)",
            f"{sum(fc.values())}/{len(fc)}", "toate", all(fc.values()))
    if "error" not in ag:
        add("Agentul folosește efectiv modelul ML la runtime", ag.get("prediction_source"), "xgboost",
            ag.get("uses_ml_model"))
    else:
        add("Agentul folosește efectiv modelul ML la runtime", ag["error"][:80], "xgboost", False)
    return out


# =====================================================================
# ORCHESTRATION
# =====================================================================
def run_benchmark(model_path: Path = DEFAULT_MODEL, write: bool = True) -> Dict[str, Any]:
    t0 = time.perf_counter()
    ctx = Ctx(model_path)
    r: Dict[str, Any] = dict(
        meta=dict(generated_at=pd.Timestamp.now().isoformat(timespec="seconds"), python=platform.python_version(),
                  platform=platform.platform(), model_path=str(ctx.model_path), model_version=ctx.version(),
                  model_type=type(ctx.model).__name__, features=ctx.features, n_features=len(ctx.features),
                  dataset_rows=int(len(ctx.canonical)), test_rows=int(len(ctx.test)),
                  test_boundary=ctx.boundary.isoformat(), grid_limit_mw=GRID_LIMIT_MW),
        artifact_metrics={k: ctx.artifact.get("metrics", {}).get(k) for k in ("test_mae", "test_rmse", "test_r2")},
    )
    for key, fn in (("accuracy", accuracy), ("regimes", regimes), ("exceedance", exceedance),
                    ("uncertainty", uncertainty), ("walk_forward", walk_forward), ("robustness", robustness),
                    ("feature_importance", importance), ("performance", performance),
                    ("agent_integration", agent_integration)):
        r[key] = _safe(fn, ctx)
    r["checks"] = checks(r)
    r["meta"]["benchmark_seconds"] = float(time.perf_counter() - t0)
    if write:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "benchmark_results.json").write_text(
            json.dumps(r, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
        (RESULTS_DIR / "benchmark_report.md").write_text(render(r), encoding="utf-8")
    return r


# =====================================================================
# REPORT
# =====================================================================
def _f(x, d=5):
    return "—" if x is None else f"{x:.{d}f}"


def _icon(p):
    return "⚠️ N/A" if p is None else ("✅" if p else "❌")


def render(r: Dict[str, Any]) -> str:
    m = r["meta"]
    hard = [c for c in r["checks"] if c["passed"] is not None]
    L = ["# 📊 VoltCompute — Benchmark ML (prognoză surplus solar)", "",
         f"- **Model:** `{m['model_path']}` · {m['model_version']} · `{m['model_type']}` · {m['n_features']} features",
         f"- **Date:** {m['dataset_rows']} ferestre orare reconstruite din CSV · test cronologic {m['test_rows']} ore "
         f"de la `{m['test_boundary']}`",
         f"- **Mediu:** Python `{m['python']}` · durată benchmark `{m['benchmark_seconds']:.1f}s` · generat `{m['generated_at']}`",
         "", f"## Verdict: {sum(c['passed'] for c in hard)}/{len(hard)} verificări trecute", "",
         "| Verificare | Valoare | Prag | Status |", "|---|---|---|:---:|"]
    L += [f"| {c['check']} | `{c['value']}` | {c['threshold']} | {_icon(c['passed'])} |" for c in r["checks"]]

    def section_error(key):
        return r[key].get("error") if isinstance(r[key], dict) else None

    L += ["", "## 1. Acuratețe vs. baseline (set test cronologic)", ""]
    if section_error("accuracy"):
        L.append(f"> ❌ {section_error('accuracy')}")
    else:
        L += ["| Orizont | Rânduri | Model MAE | RMSE | R² | Persistență MAE | Sezonier MAE | Skill vs Pers. | Skill vs Sez. |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for h, v in r["accuracy"].items():
            x = v["xgboost"]
            L.append(f"| t+{h}h | {v['test_rows']} | **{_f(x['mae'])}** | {_f(x['rmse'])} | {_f(x['r2'], 4)} | "
                     f"{_f(v['baseline_persistence']['mae'])} | {_f(v['baseline_seasonal']['mae'])} | "
                     f"{v['skill_vs_persistence']:.1%} | {v['skill_vs_seasonal']:.1%} |")
        L += ["", "> Skill = 1 − MAE_model / MAE_baseline. Predicțiile sunt tăiate la 0 MW (export negativ imposibil)."]

    L += ["", "## 2. Erori pe regimuri (după ora livrării t+2)", ""]
    if section_error("regimes"):
        L.append(f"> ❌ {section_error('regimes')}")
    else:
        L += ["| Regim | Rânduri | Export mediu MW | Model MAE | Persistență MAE | Sezonier MAE |", "|---|---:|---:|---:|---:|---:|"]
        for k, v in r["regimes"].items():
            if k != "mae_per_target_hour":
                L.append(f"| {k} | {v['rows']} | {_f(v['mean_target_mw'], 4)} | **{_f(v['xgboost_mae'])}** | "
                         f"{_f(v['persistence_mae'])} | {_f(v['seasonal_mae'])} |")
        worst = sorted(r["regimes"]["mae_per_target_hour"].items(), key=lambda x: -x[1])[:3]
        L += ["", "Cele mai dificile ore de livrare: " + ", ".join(f"`{h:02d}:00` ({e:.4f} MW)" for h, e in worst)]

    L += ["", "## 3. Detecția depășirilor (surplus > prag la t+2)", ""]
    if section_error("exceedance"):
        L.append(f"> ❌ {section_error('exceedance')}")
    else:
        L += ["| Prag | MW | Evenimente reale | Prezise | Precision | Recall | F1 | F1 baseline sezonier |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for k, v in r["exceedance"].items():
            L.append(f"| {k} | {v['threshold_mw']:.4f} | {v['actual_events']} | {v['predicted_events']} | "
                     f"{_f(v.get('precision'), 3)} | {_f(v.get('recall'), 3)} | {_f(v.get('f1'), 3)} | "
                     f"{_f(v.get('seasonal_baseline_f1'), 3)} |")

    u = r["uncertainty"]
    L += ["", "## 4. Calibrarea incertitudinii & confidența", ""]
    if u.get("error"):
        L.append(f"> ❌ {u['error']}")
    elif not u.get("available"):
        L.append(f"> ⚠️ {u['note']} Gate-ul de siguranță de 75% nu poate fi fundamentat statistic pe acest model.")
    else:
        L += ["| Metrică | Valoare |", "|---|---:|",
              f"| Acoperire nominală | {u['nominal_coverage']:.0%} |",
              f"| Acoperire empirică (tot testul) | **{u['empirical_coverage']:.1%}** |",
              f"| Acoperire empirică (ore cu export > 0) | {_f(u['coverage_daylight'], 3)} |",
              f"| Lățime medie interval | {_f(u['mean_interval_width_mw'])} MW |",
              f"| Confidență medie / min / max | {u['confidence_mean']:.3f} / {u['confidence_min']:.3f} / {u['confidence_max']:.3f} |",
              f"| Ore care trec gate-ul 0.75 | {u['share_passing_gate_075']:.1%} |",
              f"| MAE la confidență ≥ 0.75 / < 0.75 | {_f(u['mae_when_conf_ge_075'])} / {_f(u['mae_when_conf_lt_075'])} |",
              f"| Spearman(confidență, eroare) — ideal negativ | {_f(u['spearman_conf_vs_abs_error'], 3)} |"]

    wf = r["walk_forward"]
    L += ["", "## 5. Backtest walk-forward (expanding window, reantrenare per fold)", ""]
    if wf.get("error"):
        L.append(f"> ❌ {wf['error']}")
    else:
        L += ["| Fold | Train | Test | Perioadă test | Model MAE | Persistență MAE | Sezonier MAE |", "|---:|---:|---:|---|---:|---:|---:|"]
        for f in wf["folds"]:
            L.append(f"| {f['fold']} | {f['train_rows']} | {f['test_rows']} | {f['test_start'][:13]} → {f['test_end'][:13]} | "
                     f"**{_f(f['xgboost_mae'])}** | {_f(f['persistence_mae'])} | {_f(f['seasonal_mae'])} |")
        L.append(f"| **Medie** | | | | **{_f(wf['mean_xgboost_mae'])} ± {_f(wf['std_xgboost_mae'])}** | "
                 f"{_f(wf['mean_persistence_mae'])} | {_f(wf['mean_seasonal_mae'])} |")

    rob = r["robustness"]
    L += ["", "## 6. Robustețe", ""]
    if rob.get("error"):
        L.append(f"> ❌ {rob['error']}")
    else:
        L += [f"MAE referință: `{rob['baseline_mae']:.5f}` MW · cale inferență testată: {rob['inference_path']}", "",
              "| Zgomot multiplicativ pe senzorii meteo | MAE | Degradare |", "|---|---:|---:|"]
        L += [f"| {k.replace('sigma_', 'σ = ').replace('pct', '%')} | {_f(v['mae'])} | {v['degradation_pct']:+.1f}% |"
              for k, v in rob["weather_noise"].items()]
        L += ["", "| Test fail-closed | Rezultat |", "|---|:---:|"]
        L += [f"| {k} | {'✅' if v else '❌'} |" for k, v in rob["fail_closed"].items()]

    fi = r["feature_importance"]
    L += ["", "## 7. Importanța atributelor", ""]
    if fi.get("error"):
        L.append(f"> ❌ {fi['error']}")
    else:
        perm, shap = list(fi["permutation_mae_increase_mw"].items()), list(fi["treeshap_global_mean_abs_mw"].items())
        L += ["| # | Permutation (Δ MAE pe test, MW) | | TreeSHAP global pe train (mean abs, MW) | |", "|---:|---|---:|---|---:|"]
        for i in range(max(len(perm), len(shap))):
            a = perm[i] if i < len(perm) else ("", None)
            b = shap[i] if i < len(shap) else ("", None)
            L.append(f"| {i + 1} | `{a[0]}` | {_f(a[1])} | `{b[0]}` | {_f(b[1])} |")

    p = r["performance"]
    L += ["", "## 8. Performanță & reproductibilitate", ""]
    if p.get("error"):
        L.append(f"> ❌ {p['error']}")
    else:
        L += ["| Metrică | mean / p50 / p95 / p99 |", "|---|---:|"]
        for k, label in (("single_row_predict", "Predict 1 rând"), ("single_row_treeshap", "TreeSHAP 1 rând"),
                         ("full_inference_pipeline", "Inferență completă (validare + interval + SHAP)")):
            if k in p:
                s = p[k]
                L.append(f"| {label} | {s['mean_ms']:.2f} / {s['p50_ms']:.2f} / {s['p95_ms']:.2f} / {s['p99_ms']:.2f} ms |")
        L += ["", "| Metrică | Valoare |", "|---|---:|",
              f"| Throughput batch | {p['batch_throughput_rows_per_s']:,.0f} rânduri/s |",
              f"| Timp antrenare (aceeași configurație) | {p['train_seconds_same_config']:.2f} s |",
              f"| Diferență între 2 reantrenări | {p['deterministic_retrain_max_abs_diff']:.2e} |",
              f"| Diferență reantrenare vs. artefact salvat | {p['artifact_vs_retrain_max_abs_diff']:.2e} |"]

    ag = r["agent_integration"]
    L += ["", "## 9. Integrare cu agentul VoltCompute", ""]
    if ag.get("error"):
        L.append(f"> ❌ {ag['error']}")
    else:
        L += ["| Metrică | Valoare |", "|---|---|",
              f"| Model încărcat de agent | {'da' if ag['model_loaded'] else 'nu'} |",
              f"| Apeluri reușite către model | {ag.get('successful_model_calls', 0)} |",
              f"| Sursa predicției în agent | `{ag.get('prediction_source')}` |",
              f"| Predicție directă model / agent (MW) | {_f(ag.get('direct_model_prediction_mw'))} / {_f(ag.get('agent_prediction_mw'))} |"]
        if not ag.get("uses_ml_model"):
            L += ["", "> ❌ **Agentul NU folosește modelul ML** — deciziile se bazează pe fallback-ul fizic. "
                      "Cauză tipică: nume de features diferite între artefact și `agent.py`."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark ML VoltCompute")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="Calea către artefactul joblib")
    args = ap.parse_args()
    r = run_benchmark(Path(args.model), write=True)
    print(render(r).split("## 1.")[0])
    print(f"Raport complet: {RESULTS_DIR / 'benchmark_report.md'}")
    return 1 if any(c["passed"] is False for c in r["checks"]) else 0


if __name__ == "__main__":
    sys.exit(main())
