"""Backward-compatible training helpers + canonical real-data ML pipeline."""
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

from train_pipeline import main as train_canonical_pipeline

DATA_PATH = Path("data/energy_datacenter_data.csv")
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "model_surplus.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

BASE_FEATURE_COLUMNS: List[str] = [
    "hour",
    "solar_irradiance_wm2",
    "ambient_temp_celsius",
    "day_ahead_price_eur",
    "grid_capacity_limit_mw",
]

ENGINEERED_FEATURE_COLUMNS: List[str] = [
    "hour",
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "solar_irradiance_wm2",
    "ambient_temp_celsius",
    "day_ahead_price_eur",
    "grid_capacity_limit_mw",
    "effective_solar_index",
    "price_surplus_signal",
]

FEATURE_COLUMNS = ENGINEERED_FEATURE_COLUMNS
TARGET_COLUMN = "surplus_mw"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour_sin"] = np.sin(2.0 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * out["hour"] / 24.0)
    if "is_weekend" not in out.columns:
        if "timestamp" in out.columns:
            dt_series = pd.to_datetime(out["timestamp"], errors="coerce")
            out["is_weekend"] = dt_series.dt.dayofweek.isin([5, 6]).astype(int).fillna(0)
        else:
            out["is_weekend"] = 0
    temp_derating = 1.0 - np.maximum(0.0, (out["ambient_temp_celsius"] - 25.0) * 0.004)
    out["effective_solar_index"] = (out["solar_irradiance_wm2"] / 1000.0) * temp_derating
    out["price_surplus_signal"] = np.clip((60.0 - out["day_ahead_price_eur"]) / 60.0, -1.5, 2.0)
    return out


def generate_synthetic_dataset(n_hours: int = 24 * 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_time = datetime(2026, 5, 1, 0, 0, 0)
    records = []
    for i in range(n_hours):
        ts = start_time + timedelta(hours=i)
        hour = ts.hour
        day_of_year = ts.timetuple().tm_yday
        is_weekend = 1 if ts.weekday() in (5, 6) else 0
        if 6 <= hour <= 19:
            solar_angle = np.sin(np.pi * (hour - 6) / 13.0)
            cloud_factor = rng.uniform(0.55, 1.05)
            irradiance = max(0.0, 1050.0 * solar_angle * cloud_factor + rng.normal(0, 25))
        else:
            irradiance = 0.0
        base_temp = 18.0 + 9.5 * np.sin(np.pi * (hour - 5) / 15.0)
        ambient_temp = base_temp + (irradiance / 350.0) + rng.normal(0, 1.2)
        if 11 <= hour <= 15 and irradiance > 650:
            da_price = max(-10.0, 25.0 - 0.03 * (irradiance - 600) + rng.normal(0, 4.0))
        elif 18 <= hour <= 21:
            da_price = 125.0 + rng.normal(0, 12.0)
        else:
            da_price = 75.0 + rng.normal(0, 8.0)
        grid_limit = 10.0
        temp_derating = 1.0 - max(0.0, (ambient_temp - 25.0) * 0.004)
        gross_solar_mw = (irradiance / 1000.0) * 17.8 * temp_derating
        local_load_mw = 1.4 + 0.5 * np.sin(np.pi * hour / 12.0) + rng.normal(0, 0.15)
        net_injection_mw = max(0.0, gross_solar_mw - local_load_mw)
        surplus_mw = max(0.0, net_injection_mw - grid_limit)
        records.append(
            {
                "timestamp": ts.isoformat(),
                "hour": hour,
                "day_of_year": day_of_year,
                "is_weekend": is_weekend,
                "solar_irradiance_wm2": round(float(irradiance), 2),
                "ambient_temp_celsius": round(float(ambient_temp), 2),
                "day_ahead_price_eur": round(float(da_price), 2),
                "grid_capacity_limit_mw": round(float(grid_limit), 2),
                "gross_solar_mw": round(float(gross_solar_mw), 3),
                "surplus_mw": round(float(surplus_mw), 3),
            }
        )
    return pd.DataFrame(records)


def main():
    return train_canonical_pipeline()


if __name__ == "__main__":
    main()
