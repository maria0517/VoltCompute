"""Canonical causal t+2h XGBoost pipeline and shared CSV/inference helpers."""
from functools import lru_cache
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import DMatrix, XGBRegressor

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MACRO_FILE = DATA_DIR / "1_macro_vreme_piata.csv"
PROSUMERS_FILE = DATA_DIR / "2_micro_100_prosumatori.csv"
MODEL_OUTPUT_PATH = ROOT / "model.joblib"
FEATURE_NAMES = [
    "hour", "dayofweek", "is_weekend", "hour_sin", "hour_cos",
    "temp_c", "radiatie_solara_wm2", "viteza_vant_kmh", "pret_pzu_eur_mwh",
    "Surplus_National_MW", "avg_battery_soc_pct", "total_prosumer_load_mw",
    "surplus_lag_1", "surplus_lag_2", "surplus_lag_24",
    "temp_lag_1", "radiatie_lag_1", "pret_lag_1",
    "seasonal_radiation_t_plus_2", "seasonal_temp_t_plus_2", "seasonal_wind_t_plus_2",
]


def load_hourly_history(macro_path=MACRO_FILE, prosumers_path=PROSUMERS_FILE):
    """All 720 observed hours, with causal features; warm-up lags remain NaN."""
    micro = pd.read_csv(prosumers_path, parse_dates=["timestamp"])
    sums = micro.groupby("timestamp")[
        ["export_in_retea_kw", "baterie_curenta_kwh", "capacitate_bat_kwh", "consum_kw"]
    ].sum(min_count=1)
    sums["Surplus_National_MW"] = sums["export_in_retea_kw"] / 1000
    sums["avg_battery_soc_pct"] = 100 * sums["baterie_curenta_kwh"] / sums["capacitate_bat_kwh"]
    sums["total_prosumer_load_mw"] = sums["consum_kw"] / 1000
    df = pd.read_csv(macro_path, parse_dates=["timestamp"]).merge(
        sums.reset_index(), on="timestamp", validate="one_to_one"
    ).sort_values("timestamp").reset_index(drop=True)
    if not df.timestamp.diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("CSV history must contain consecutive hourly observations")
    df["hour"] = df.timestamp.dt.hour
    df["dayofweek"] = df.timestamp.dt.dayofweek
    df["is_weekend"] = (df.dayofweek >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.hour / 24)
    for lag in (1, 2, 24):
        df[f"surplus_lag_{lag}"] = df.Surplus_National_MW.shift(lag)
    for dest, source in [("temp", "temp_c"), ("radiatie", "radiatie_solara_wm2"), ("pret", "pret_pzu_eur_mwh")]:
        df[f"{dest}_lag_1"] = df[source].shift(1)
    # No issued-weather-forecast dataset exists: explicitly label the seasonal estimate.
    # Weather at t+2 is estimated using the observed weather at t+2-24 = t-22.
    for name, source in [("radiation", "radiatie_solara_wm2"), ("temp", "temp_c"), ("wind", "viteza_vant_kmh")]:
        df[f"seasonal_{name}_t_plus_2"] = df[source].shift(22)
    return df


def load_and_preprocess_data(macro_path=MACRO_FILE, prosumers_path=PROSUMERS_FILE):
    df = load_hourly_history(macro_path, prosumers_path)
    df["target"] = df.Surplus_National_MW.shift(-2)
    df["target_timestamp"] = df.timestamp.shift(-2)
    df["seasonal_baseline"] = df.Surplus_National_MW.shift(22)
    return df.dropna().reset_index(drop=True)


@lru_cache(maxsize=1)
def _csv_data():
    return load_hourly_history(), pd.read_csv(DATA_DIR / "3_data_centers.csv", parse_dates=["timestamp"])


def load_grid_status(step=37):
    """CSV index 0..719; steps <24 expose incomplete features and fail closed in ML."""
    history, centers = _csv_data()
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < len(history):
        raise ValueError("Step outside hourly history")
    row = history.iloc[step]
    features = {k: (float(row[k]) if pd.notna(row[k]) else None) for k in FEATURE_NAMES}
    grid = dict(timestamp=row.timestamp.isoformat(), hour=int(row.hour),
                solar_irradiance_wm2=float(row.radiatie_solara_wm2), ambient_temp_celsius=float(row.temp_c),
                day_ahead_price_eur=float(row.pret_pzu_eur_mwh), wind_speed_kmh=float(row.viteza_vant_kmh),
                prosumer_battery_soc_pct=float(row.avg_battery_soc_pct), current_surplus_mw=float(row.Surplus_National_MW),
                total_prosumer_load_mw=float(row.total_prosumer_load_mw), is_weekend=int(row.is_weekend),
                grid_capacity_limit_mw=0.35, model_features=features, data_source="real_csv")
    dcs = centers.loc[centers.timestamp.eq(row.timestamp)].copy()
    dcs["timestamp"] = dcs.timestamp.map(lambda x: x.isoformat())
    return dict(step=step, total_steps=len(history), timestamp=grid["timestamp"], grid=grid,
                data_centers=dcs.to_dict("records"), price_pzu_eur_mwh=grid["day_ahead_price_eur"],
                avg_battery_soc_pct=grid["prosumer_battery_soc_pct"], data_source="real_csv")


def _model():
    return XGBRegressor(n_estimators=240, max_depth=3, learning_rate=0.045,
                        objective="reg:squarederror", tree_method="hist", n_jobs=2,
                        subsample=1.0, colsample_bytree=1.0, random_state=42)


def _scores(y, prediction):
    return dict(mae=float(mean_absolute_error(y, prediction)),
                rmse=float(np.sqrt(mean_squared_error(y, prediction))), r2=float(r2_score(y, prediction)))


def train_model(df, horizon=2, test_start=None):
    split = int(len(df) * 0.8)
    if test_start is not None:
        split = int((df.timestamp < test_start).sum())
    test = df.iloc[split:]
    # Purge labels at/after the first test observation (including the boundary).
    train = df.iloc[:split].loc[lambda x: x.target_timestamp < test.timestamp.iloc[0]]
    calibration_start = int(len(train) * 0.75)
    calibration = train.iloc[calibration_start:]
    fit = train.iloc[:calibration_start].loc[lambda x: x.target_timestamp < calibration.timestamp.iloc[0]]
    pilot = _model().fit(fit[FEATURE_NAMES], fit.target)
    errors = np.abs(calibration.target.to_numpy() - pilot.predict(calibration[FEATURE_NAMES]))
    # Held-out residuals exclusively inside the training period; never test residuals.
    q = float(np.quantile(errors, 0.90, method="higher"))
    hourly = {str(h): float(np.quantile(errors[calibration.hour.to_numpy() == h], .9, method="higher"))
              for h in range(24) if (calibration.hour == h).any()}
    model = _model().fit(train[FEATURE_NAMES], train.target)
    prediction = model.predict(test[FEATURE_NAMES])
    scores = {"xgboost": _scores(test.target, prediction),
              "baseline_persistence": _scores(test.target, test.Surplus_National_MW),
              "baseline_seasonal": _scores(test.target, test.seasonal_baseline)}
    metrics = dict(scores, test_mae=scores["xgboost"]["mae"], test_rmse=scores["xgboost"]["rmse"],
                   test_r2=scores["xgboost"]["r2"], baseline_persistence_mae=scores["baseline_persistence"]["mae"],
                   baseline_seasonal_mae=scores["baseline_seasonal"]["mae"],
                   train_mae=float(mean_absolute_error(train.target, model.predict(train[FEATURE_NAMES]))),
                   train_rows=len(train), test_rows=len(test), purged_rows=split-len(train),
                   prediction_horizon_hours=horizon,
                   train_last_target=train.target_timestamp.iloc[-1].isoformat(),
                   test_first_timestamp=test.timestamp.iloc[0].isoformat())
    model.volt_metadata = dict(
        calibration=dict(method="train_only_chronological_holdout_absolute_residual_90pct", quantile=q,
                         hourly_quantiles=hourly, confidence_scale_mw=float(train.target.max()),
                         calibration_rows=len(calibration), last_target=calibration.target_timestamp.iloc[-1].isoformat()),
        feature_bounds={k: [float(train[k].min()), float(train[k].max())] for k in FEATURE_NAMES},
        shap=dict(method="xgboost_native_exact_treeshap", units="MW", approximate=False,
                  global_mean_abs=dict(zip(FEATURE_NAMES, np.abs(model.get_booster().predict(
                      DMatrix(train[FEATURE_NAMES]), pred_contribs=True)[:, :-1]).mean(axis=0).astype(float)))))
    print("Model                       MAE          RMSE         R2")
    for name, values in scores.items():
        print(f"{name:27} {values['mae']:.6f}     {values['rmse']:.6f}     {values['r2']:.6f}")
    return model, FEATURE_NAMES.copy(), metrics


def _artifact(model, feature_cols, metrics, horizon=2):
    return dict(model=model, feature_names=feature_cols, feature_columns=feature_cols,
                    feature_schema=[dict(name=k, dtype="float64", available_at="t_or_earlier") for k in feature_cols],
                    weather_estimate=dict(method="seasonal_observation_t_minus_22", target_horizon_hours=2,
                                          issued_forecast=False, source_available_at="t_minus_22"),
                    metrics=metrics, target_description=f"Surplus_National_MW_t_plus_{horizon}h",
                    prediction_horizon_hours=horizon, schema_version=2, **model.volt_metadata)


def train_forecast_models(canonical_df, canonical_model, canonical_metrics):
    """Additional horizons use the frozen t+2 test boundary, never future targets in train."""
    boundary = canonical_df.iloc[int(len(canonical_df)*.8)].timestamp
    models = {"2": _artifact(canonical_model, FEATURE_NAMES.copy(), dict(canonical_metrics))}
    history = load_hourly_history()
    for horizon in (1, 3, 4):
        frame = history.copy()
        frame["target"] = frame.Surplus_National_MW.shift(-horizon)
        frame["target_timestamp"] = frame.timestamp.shift(-horizon)
        frame["seasonal_baseline"] = frame.Surplus_National_MW.shift(24-horizon)
        # Same observation window/boundary; only unavailable terminal labels differ.
        frame = frame.loc[frame.timestamp.between(canonical_df.timestamp.iloc[0], canonical_df.timestamp.iloc[-1])].dropna().reset_index(drop=True)
        print(f"Forecast horizon t+{horizon}h; frozen test boundary {boundary}")
        model, features, metrics = train_model(frame, horizon=horizon, test_start=boundary)
        models[str(horizon)] = _artifact(model, features, metrics, horizon)
    return models


def save_pipeline(model, feature_cols, metrics, output_path=MODEL_OUTPUT_PATH, forecast_models=None):
    artifact = _artifact(model, feature_cols, metrics)
    if forecast_models is not None:
        artifact["forecast_models"] = forecast_models
    joblib.dump(artifact, output_path)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(artifact, ROOT / "models/model.joblib")
    (ROOT / "models/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _load_artifact.cache_clear()


@lru_cache(maxsize=1)
def _load_artifact():
    return joblib.load(MODEL_OUTPUT_PATH)


def infer_single_step(input_dict, artifact=None):
    """Strict ordered feature inference. No imputation, future lookups or physical fallback."""
    artifact = _load_artifact() if artifact is None else artifact
    source = input_dict.get("model_features", input_dict)
    names = artifact["feature_names"]
    try:
        values = [float(source[k]) for k in names]
    except (KeyError, TypeError, ValueError):
        raise ValueError("Missing or invalid model features") from None
    if not np.isfinite(values).all():
        raise ValueError("Non-finite model features")
    features = dict(zip(names, values))
    bounds = {"hour": (0, 23), "dayofweek": (0, 6), "is_weekend": (0, 1),
              "hour_sin": (-1, 1), "hour_cos": (-1, 1), "temp_c": (-40, 60),
              "temp_lag_1": (-40, 60), "radiatie_solara_wm2": (0, 1500),
              "radiatie_lag_1": (0, 1500), "viteza_vant_kmh": (0, 150),
              "avg_battery_soc_pct": (0, 100)}
    bounds.update(seasonal_radiation_t_plus_2=(0, 1500), seasonal_temp_t_plus_2=(-40, 60),
                  seasonal_wind_t_plus_2=(0, 150))
    for name in ("Surplus_National_MW", "total_prosumer_load_mw", "surplus_lag_1", "surplus_lag_2", "surplus_lag_24"):
        bounds[name] = (0, float("inf"))
    if any(not lo <= features[k] <= hi for k, (lo, hi) in bounds.items()):
        raise ValueError("Model features outside physical bounds")
    frame = pd.DataFrame([values], columns=names)
    contributions = artifact["model"].get_booster().predict(DMatrix(frame), pred_contribs=True)[0]
    raw = float(artifact["model"].predict(frame)[0])
    prediction = max(0.0, raw)
    calibration = artifact["calibration"]
    radius = max(calibration["quantile"] * .25,
                 calibration["hourly_quantiles"].get(str(int(frame.hour.iloc[0])), calibration["quantile"]))
    scale = max(calibration["confidence_scale_mw"], 1e-6)
    ood = sum(max(0, lo-v, v-hi) / max(hi-lo, 1) for v, (lo, hi) in
              zip(values, [artifact["feature_bounds"][k] for k in names]))
    confidence = float(np.clip(1 - radius / scale - min(.5, ood * .1), 0, 1))
    shap_values = dict(zip(names, map(float, contributions[:-1])))
    top = dict(sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:3])
    denominator = sum(abs(v) for v in top.values())
    return dict(predicted_surplus_mw=prediction, surplus_predicted_mw=prediction, confidence_score=confidence,
                prediction_lower_mw=max(0., prediction-radius), prediction_upper_mw=prediction+radius,
                top_shap_features=top, shap_percentages={k: 100*abs(v)/denominator if denominator else 0 for k,v in top.items()},
                shap_values=shap_values, shap_base_value=float(contributions[-1]), raw_prediction_mw=raw,
                predicted_export_mw=prediction,
                weather_estimate_method="seasonal_observation_t_minus_22",
                weather_estimate_is_issued_forecast=False,
                shap_method="xgboost_native_exact_treeshap", prediction_source="xgboost",
                prediction_horizon_hours=artifact.get("prediction_horizon_hours", 2))


def infer_forecast_window(features, artifact=None):
    """Four independent causal XGB forecasts, with empirical intervals, ordered t+1..t+4.

    Accepts the same dict/grid.model_features as infer_single_step. All values are
    base CSV export forecasts; demo injections are intentionally not ML predictions.
    """
    artifact = _load_artifact() if artifact is None else artifact
    models = artifact.get("forecast_models", {})
    if any(str(h) not in models for h in (1, 2, 3, 4)):
        raise ValueError("Multi-horizon artifacts unavailable; run canonical pipeline")
    return [infer_single_step(features, models[str(h)]) for h in (1, 2, 3, 4)]


def main():
    started = time.perf_counter()
    df = load_and_preprocess_data()
    model, features, metrics = train_model(df)
    forecast_models = train_forecast_models(df, model, metrics)
    metrics["forecast_horizons"] = {h: bundle["metrics"] for h, bundle in forecast_models.items()}
    metrics["pipeline_seconds"] = time.perf_counter() - started
    save_pipeline(model, features, metrics, forecast_models=forecast_models)
    print(f"Pipeline completed in {time.perf_counter()-started:.3f}s; artifacts saved.")


if __name__ == "__main__":
    main()
