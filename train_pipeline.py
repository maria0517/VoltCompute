#!/usr/bin/env python3
"""
Green Compute Orchestrator - ML Pipeline
Pasul 1: Antrenare model predicție surplus prosumatori (Surplus_National_MW) la t+2 ore.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

if sys.platform == 'win32':
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Încercăm importul XGBoost, cu fallback la HistGradientBoostingRegressor dacă e cazul
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_XGB = False

# Căi implicite
DATA_DIR = Path("data")
MACRO_FILE = DATA_DIR / "1_macro_vreme_piata.csv"
PROSUMERS_FILE = DATA_DIR / "2_micro_100_prosumatori.csv"
MODEL_OUTPUT_PATH = Path("model.joblib")


def load_and_preprocess_data(
    macro_path: Path = MACRO_FILE,
    prosumers_path: Path = PROSUMERS_FILE,
) -> pd.DataFrame:
    """
    1. Citește datele de prosumatori și calculează surplusul național (MW).
    2. Fuzionează cu datele macro (vreme și piață PZU).
    3. Construiește features temporale și lag-uri.
    4. Tratează valorile lipsă (NaN).
    """
    print(f"[*] Încărcare date prosumatori din: {prosumers_path}")
    prosumers_df = pd.read_csv(prosumers_path)
    prosumers_df["timestamp"] = pd.to_datetime(prosumers_df["timestamp"])

    # Groupby după timestamp și suma coloanei export_in_retea_kw
    surplus_df = (
        prosumers_df.groupby("timestamp", as_index=False)["export_in_retea_kw"]
        .sum()
        .rename(columns={"export_in_retea_kw": "Surplus_National_kW"})
    )

    # Conversie în MW
    surplus_df["Surplus_National_MW"] = surplus_df["Surplus_National_kW"] / 1000.0
    print(f"[+] Prosumatori agregați: {len(surplus_df)} intervale orare.")

    # Încărcare macro vreme și piață
    print(f"[*] Încărcare date macro vreme & piață din: {macro_path}")
    macro_df = pd.read_csv(macro_path)
    macro_df["timestamp"] = pd.to_datetime(macro_df["timestamp"])

    # Merge pe timestamp
    df = pd.merge(surplus_df, macro_df, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"[+] Set integrat inițial: {len(df)} rânduri.")

    # Creare features temporale
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2.0 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * df["hour"] / 24.0)

    # Creare lag-uri (lag 1h, lag 2h, lag 24h pentru surplus și vreme)
    df["surplus_lag_1"] = df["Surplus_National_MW"].shift(1)
    df["surplus_lag_2"] = df["Surplus_National_MW"].shift(2)
    df["surplus_lag_24"] = df["Surplus_National_MW"].shift(24)  # surplusul de acum 24h
    df["temp_lag_1"] = df["temp_c"].shift(1)
    df["radiatie_lag_1"] = df["radiatie_solara_wm2"].shift(1)
    df["pret_lag_1"] = df["pret_pzu_eur_mwh"].shift(1)

    # Target: Surplus_National_MW la t+2 (peste 2 ore)
    df["target"] = df["Surplus_National_MW"].shift(-2)

    # Tratare valori NaN generate de lag-uri (primele 24 rânduri) și shift(-2) (ultimele 2)
    initial_rows = len(df)
    clean_df = df.dropna().reset_index(drop=True)
    dropped_rows = initial_rows - len(clean_df)
    print(f"[+] Tratare NaN finalizată: {dropped_rows} rânduri eliminate. Rămase: {len(clean_df)} rânduri curate.")

    return clean_df


def train_model(
    df: pd.DataFrame,
) -> Tuple[Any, List[str], Dict[str, float]]:
    """
    Împarte setul de date 80/20 secvențial (fără random split pentru serii temporale)
    și antrenează XGBoostRegressor.
    """
    feature_cols = [
        "hour",
        "dayofweek",
        "hour_sin",
        "hour_cos",
        "temp_c",
        "radiatie_solara_wm2",
        "viteza_vant_kmh",
        "pret_pzu_eur_mwh",
        "Surplus_National_MW",
        "surplus_lag_1",
        "surplus_lag_2",
        "surplus_lag_24",
        "temp_lag_1",
        "radiatie_lag_1",
        "pret_lag_1",
    ]

    target_col = "target"

    # Split secvențial 80/20 (păstrând ordinea cronologică)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    print(f"[*] Train set: {len(X_train)} mostre | Test set: {len(X_test)} mostre (split 80/20 secvențial)")

    if HAS_XGB:
        print("[*] Antrenare model XGBoostRegressor...")
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            objective="reg:squarederror",
        )
    else:
        print("[!] XGBoost indisponibil, utilizare HistGradientBoostingRegressor fallback...")
        model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
        )

    model.fit(X_train, y_train)

    # Evaluare pe train și test
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, train_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    train_rmse = root_mean_squared_error(y_train, train_preds)
    test_rmse = root_mean_squared_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)

    metrics = {
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "test_r2": float(test_r2),
    }

    print("\n================== METRICI MODEL (t+2 ore) ==================")
    print(f"Train MAE : {train_mae:.4f} MW | Train RMSE: {train_rmse:.4f} MW")
    print(f"Test MAE  : {test_mae:.4f} MW | Test RMSE : {test_rmse:.4f} MW")
    print(f"Test R²   : {test_r2:.4f}")
    print("============================================================\n")

    return model, feature_cols, metrics


def save_pipeline(
    model: Any,
    feature_cols: List[str],
    metrics: Dict[str, float],
    output_path: Path = MODEL_OUTPUT_PATH,
) -> None:
    """
    Salvează modelul antrenat și metadatele într-un pachet joblib.
    Asigură acces direct la .predict() dacă cineva folosește bundle['model'] sau obiectul direct.
    """
    payload = {
        "model": model,
        "feature_names": feature_cols,
        "metrics": metrics,
        "target_description": "Surplus_National_MW_t_plus_2h",
    }

    # Salvare în locația cerută expres: model.joblib
    joblib.dump(payload, output_path)
    print(f"[+] Model salvat cu succes în: {output_path.resolve()}")

    # Salvăm și o copie în models/ pentru consistență
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, models_dir / "model.joblib")
    print(f"[+] Copie de siguranță salvată în: {(models_dir / 'model.joblib').resolve()}")


def main():
    df = load_and_preprocess_data()
    model, features, metrics = train_model(df)
    save_pipeline(model, features, metrics)
    print("[✓] Pipeline de antrenare finalizat cu succes!")


if __name__ == "__main__":
    main()
