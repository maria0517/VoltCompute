import json
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

DATA_PATH = Path("data/energy_datacenter_data.csv")
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "model_surplus.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

FEATURE_COLUMNS = [
    "hour",
    "solar_irradiance_wm2",
    "ambient_temp_celsius",
    "day_ahead_price_eur",
    "grid_capacity_limit_mw",
]
TARGET_COLUMN = "surplus_mw"


def generate_synthetic_dataset(n_hours: int = 24 * 60, seed: int = 42) -> pd.DataFrame:
    """
    Generează un set de date sintetic realist pe 60 de zile (1440 ore)
    pentru producția parcului fotovoltaic PPC și limitele rețelei.
    """
    rng = np.random.default_rng(seed)
    start_time = datetime(2026, 5, 1, 0, 0, 0)

    records = []
    for i in range(n_hours):
        ts = start_time + timedelta(hours=i)
        hour = ts.hour
        day_of_year = ts.timetuple().tm_yday

        # Profil solar zilnic (răsărit ~6:00, apus ~20:00, maxim la 13:00)
        if 6 <= hour <= 19:
            solar_angle = np.sin(np.pi * (hour - 6) / 13.0)
            cloud_factor = rng.uniform(0.55, 1.05)
            irradiance = max(0.0, 1050.0 * solar_angle * cloud_factor + rng.normal(0, 25))
        else:
            irradiance = 0.0

        # Temperatură ambientală corelată cu ora și iradianța
        base_temp = 18.0 + 9.5 * np.sin(np.pi * (hour - 5) / 15.0)
        ambient_temp = base_temp + (irradiance / 350.0) + rng.normal(0, 1.2)

        # Preț PZU (scade puternic la prânz când e supraproducție solară - "duck curve")
        if 11 <= hour <= 15 and irradiance > 650:
            da_price = max(-10.0, 25.0 - 0.03 * (irradiance - 600) + rng.normal(0, 4.0))
        elif 18 <= hour <= 21:
            da_price = 125.0 + rng.normal(0, 12.0)
        else:
            da_price = 75.0 + rng.normal(0, 8.0)

        grid_limit = 10.0

        # Puterea brută produsă de parc (capacitate instalată ~18.5 MWp)
        # Eficiența panourilor scade ușor la temperaturi foarte mari (>25°C)
        temp_derating = 1.0 - max(0.0, (ambient_temp - 25.0) * 0.004)
        gross_solar_mw = (irradiance / 1000.0) * 17.8 * temp_derating

        # Consumul local de bază pe linie (variază între 1.0 și 2.4 MW)
        local_load_mw = 1.4 + 0.5 * np.sin(np.pi * hour / 12.0) + rng.normal(0, 0.15)

        net_injection_mw = max(0.0, gross_solar_mw - local_load_mw)
        surplus_mw = max(0.0, net_injection_mw - grid_limit)

        records.append(
            {
                "timestamp": ts.isoformat(),
                "hour": hour,
                "day_of_year": day_of_year,
                "solar_irradiance_wm2": round(float(irradiance), 2),
                "ambient_temp_celsius": round(float(ambient_temp), 2),
                "day_ahead_price_eur": round(float(da_price), 2),
                "grid_capacity_limit_mw": round(float(grid_limit), 2),
                "gross_solar_mw": round(float(gross_solar_mw), 3),
                "surplus_mw": round(float(surplus_mw), 3),
            }
        )

    return pd.DataFrame(records)


def ensure_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Încarcă CSV-ul existent sau generează automat date sintetice de start."""
    if not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Nu s-a găsit {data_path}. Generăm set de date sintetic (1440 ore)...")
        df = generate_synthetic_dataset()
        df.to_csv(data_path, index=False)
        print(f"[INFO] Set de date salvat în {data_path} ({len(df)} rânduri).")
    else:
        print(f"[INFO] Încărcăm setul de date existent din {data_path}...")
        df = pd.read_csv(data_path)
    return df


def train_pipeline() -> dict:
    """
    Antrenează și compară DummyRegressor (Baseline) cu XGBRegressor pe date ordonate temporal
    (shuffle=False), exportând metricile și modelul final.
    """
    df = ensure_dataset()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Split strict temporal (fără data leakage din viitor)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # 1. Baseline: DummyRegressor (media istorică)
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    y_pred_base = baseline.predict(X_test)
    base_mae = float(mean_absolute_error(y_test, y_pred_base))
    base_rmse = float(root_mean_squared_error(y_test, y_pred_base))

    # 2. Model avansat: XGBRegressor
    xgb_model = XGBRegressor(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = np.clip(xgb_model.predict(X_test), 0.0, None)

    xgb_mae = float(mean_absolute_error(y_test, y_pred_xgb))
    xgb_rmse = float(root_mean_squared_error(y_test, y_pred_xgb))

    mae_improvement_pct = (
        ((base_mae - xgb_mae) / base_mae) * 100.0 if base_mae > 0 else 0.0
    )
    rmse_improvement_pct = (
        ((base_rmse - xgb_rmse) / base_rmse) * 100.0 if base_rmse > 0 else 0.0
    )

    feature_importances = {
        col: round(float(imp), 4)
        for col, imp in zip(FEATURE_COLUMNS, xgb_model.feature_importances_)
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Salvăm pachetul model + metadate în models/model_surplus.pkl
    artifact = {
        "model": xgb_model,
        "feature_columns": FEATURE_COLUMNS,
        "feature_importances": feature_importances,
        "trained_at": datetime.now().isoformat(),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    metrics = {
        "trained_at": datetime.now().isoformat(),
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "split_strategy": "train_test_split(shuffle=False, test_size=0.2)",
        "baseline_dummy": {
            "mae": round(base_mae, 4),
            "rmse": round(base_rmse, 4),
        },
        "xgboost_regressor": {
            "mae": round(xgb_mae, 4),
            "rmse": round(xgb_rmse, 4),
        },
        "improvement_pct": {
            "mae_reduction_pct": round(mae_improvement_pct, 2),
            "rmse_reduction_pct": round(rmse_improvement_pct, 2),
        },
        "feature_importances": feature_importances,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n=== Rezultate Antrenare VoltCompute ===")
    print(f"Baseline (Dummy)  -> MAE: {base_mae:.4f} MW | RMSE: {base_rmse:.4f} MW")
    print(f"XGBRegressor      -> MAE: {xgb_mae:.4f} MW | RMSE: {xgb_rmse:.4f} MW")
    print(f"Îmbunătățire MAE  -> {mae_improvement_pct:.2f}%")
    print(f"Îmbunătățire RMSE -> {rmse_improvement_pct:.2f}%")
    print(f"Model salvat în   -> {MODEL_PATH}")
    print(f"Metrici salvate în-> {METRICS_PATH}\n")

    return metrics


if __name__ == "__main__":
    train_pipeline()

