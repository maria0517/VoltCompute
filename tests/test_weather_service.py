import re
from pathlib import Path

import pandas as pd
import pytest

from services.weather_service import (
    BUCHAREST_LAT,
    BUCHAREST_LON,
    ENERGY_COLUMNS,
    MACRO_COLUMNS,
    calculate_pzu_price,
    fetch_weather_forecast,
    get_energy_datacenter_dataframe,
    get_macro_weather_dataframe,
    load_macro_weather,
    sync_weather_to_csv,
)


def test_fetch_weather_forecast_live() -> None:
    """Verifică interogarea API Open-Meteo pentru București."""
    data = fetch_weather_forecast(forecast_days=2)
    assert "hourly" in data
    hourly = data["hourly"]
    for col in ["time", "temperature_2m", "shortwave_radiation", "wind_speed_10m"]:
        assert col in hourly
    assert len(hourly["time"]) == 48  # 2 zile x 24h


def test_macro_dataframe_structure_and_types() -> None:
    """
    Verifică că DataFrame-ul returnat respectă cu strictețe schema din:
    1_macro_vreme_piata.csv:
    timestamp,temp_c,radiatie_solara_wm2,viteza_vant_kmh,pret_pzu_eur_mwh
    """
    df = get_macro_weather_dataframe(forecast_days=3)

    # 1. Coloane exacte în ordinea cerută
    assert list(df.columns) == MACRO_COLUMNS

    # 2. Nu există valori NaN
    assert not df.isnull().values.any()
    assert len(df) == 72

    # 3. Format timestamp YYYY-MM-DD HH:MM:SS
    sample_ts = str(df["timestamp"].iloc[0])
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", sample_ts)

    # 4. Limite fizice valide
    assert df["temp_c"].between(-40.0, 50.0).all()
    assert (df["radiatie_solara_wm2"] >= 0.0).all()
    assert (df["viteza_vant_kmh"] >= 0.0).all()


def test_energy_datacenter_dataframe_structure_and_types() -> None:
    """
    Verifică că DataFrame-ul returnat respectă schema din:
    energy_datacenter_data.csv:
    timestamp,hour,day_of_year,solar_irradiance_wm2,ambient_temp_celsius,day_ahead_price_eur,grid_capacity_limit_mw,gross_solar_mw,surplus_mw
    """
    df = get_energy_datacenter_dataframe(forecast_days=2)

    assert list(df.columns) == ENERGY_COLUMNS
    assert not df.isnull().values.any()
    assert len(df) == 48

    # Format ISO 8601
    sample_ts = str(df["timestamp"].iloc[0])
    assert "T" in sample_ts

    assert df["hour"].between(0, 23).all()
    assert df["day_of_year"].between(1, 366).all()
    assert (df["solar_irradiance_wm2"] >= 0.0).all()
    assert (df["grid_capacity_limit_mw"] == 10.0).all()
    assert (df["gross_solar_mw"] >= 0.0).all()
    assert (df["surplus_mw"] >= 0.0).all()


def test_calculate_pzu_price_duck_curve() -> None:
    """Verifică comportamentul curbei duck la prânz vs noapte/seară."""
    price_night = calculate_pzu_price(hour=2, irradiance_wm2=0.0, seed=42)
    price_noon_sunny = calculate_pzu_price(hour=13, irradiance_wm2=900.0, seed=42)
    price_evening_peak = calculate_pzu_price(hour=20, irradiance_wm2=0.0, seed=42)

    # La prânz cu iradianță mare, prețul trebuie să scadă puternic sub cel de seară și de noapte
    assert price_noon_sunny < price_night
    assert price_evening_peak > price_night


def test_sync_weather_to_csv_in_temp_dir(tmp_path: Path) -> None:
    """Verifică scrierea corectă în CSV-uri cu generare de backup."""
    macro_file = tmp_path / "1_macro_vreme_piata.csv"
    energy_file = tmp_path / "energy_datacenter_data.csv"

    # Creăm un fișier inițial pentru testarea backup-ului
    macro_file.write_text("dummy content")

    results = sync_weather_to_csv(
        macro_csv_path=macro_file,
        energy_csv_path=energy_file,
        forecast_days=2,
        backup=True,
    )

    assert "macro_csv" in results
    assert "energy_csv" in results
    assert macro_file.exists()
    assert energy_file.exists()
    assert (tmp_path / "1_macro_vreme_piata.csv.bak").exists()

    # Citire înapoi cu pandas
    read_macro = pd.read_csv(macro_file)
    assert list(read_macro.columns) == MACRO_COLUMNS
    assert len(read_macro) == 48

    read_energy = pd.read_csv(energy_file)
    assert list(read_energy.columns) == ENERGY_COLUMNS
    assert len(read_energy) == 48


def test_load_macro_weather_fallback(tmp_path: Path) -> None:
    """Verifică fallback-ul pe CSV când use_api=False."""
    dummy_file = tmp_path / "test_macro.csv"
    dummy_df = pd.DataFrame(
        {
            "timestamp": ["2026-11-08 12:00:00"],
            "temp_c": [15.2],
            "radiatie_solara_wm2": [450.0],
            "viteza_vant_kmh": [8.5],
            "pret_pzu_eur_mwh": [25.0],
        }
    )
    dummy_df.to_csv(dummy_file, index=False)

    loaded, source = load_macro_weather(macro_csv_path=dummy_file, use_api=False)
    assert len(loaded) == 1
    assert "Local CSV" in source


def test_fastapi_weather_endpoints() -> None:
    """Verifică integrarea endpoint-urilor FastAPI /api/weather/bucharest."""
    from fastapi.testclient import TestClient
    from server import app

    client = TestClient(app)
    resp = client.get("/api/weather/bucharest?format_type=macro&forecast_days=1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["location"] == "București"
    assert body["format"] == "macro"
    assert body["records_count"] == 24
    assert list(body["data"][0].keys()) == MACRO_COLUMNS


def test_weather_api_renaming_mappings() -> None:
    """Verifică redenumirea conformă a cheilor din API în headerele CSV existente."""
    from services.weather_service import API_TO_ENERGY_RENAME, API_TO_MACRO_RENAME

    # În 1_macro_vreme_piata.csv
    assert API_TO_MACRO_RENAME["time"] == "timestamp"
    assert API_TO_MACRO_RENAME["temperature_2m"] == "temp_c"
    assert API_TO_MACRO_RENAME["shortwave_radiation"] == "radiatie_solara_wm2"
    assert API_TO_MACRO_RENAME["wind_speed_10m"] == "viteza_vant_kmh"

    # În energy_datacenter_data.csv
    assert API_TO_ENERGY_RENAME["time"] == "timestamp"
    assert API_TO_ENERGY_RENAME["temperature_2m"] == "ambient_temp_celsius"
    assert API_TO_ENERGY_RENAME["shortwave_radiation"] == "solar_irradiance_wm2"


def test_fetch_bucharest_weather_past_and_future() -> None:
    """Verifică extragerea combinată a datelor din trecut (past) și viitor (forecast)."""
    from services.weather_service import fetch_bucharest_weather

    # 1 zi trecut (24h) + 2 zile prognoză (48h) = 72h
    df = fetch_bucharest_weather(forecast_days=2, past_days=1, format_type="macro")
    assert len(df) == 72
    assert list(df.columns) == MACRO_COLUMNS
    assert not df.isnull().values.any()


def test_weather_lookahead_2_hours() -> None:
    """Verifică furnizarea datelor meteo cu 2 ore în avans pentru ofertare solară PPC."""
    from services.weather_service import get_weather_lookahead

    lookahead = get_weather_lookahead(hours_ahead=2)
    assert lookahead["lookahead_hours"] == 2
    assert "location" in lookahead
    assert "timestamp" in lookahead
    # Câmpurile obligatorii din CSV-ul de vreme sunt prezente ca float/numere valide
    for col in ["temp_c", "radiatie_solara_wm2", "viteza_vant_kmh", "pret_pzu_eur_mwh"]:
        assert col in lookahead
        assert isinstance(lookahead[col], (int, float))

    assert "gross_solar_mw" in lookahead
    assert "surplus_mw" in lookahead
    assert "offer_recommended" in lookahead
    assert isinstance(lookahead["offer_recommended"], bool)


def test_fastapi_weather_lookahead_endpoint() -> None:
    """Verifică endpoint-ul GET /api/weather/bucharest/lookahead."""
    from fastapi.testclient import TestClient
    from server import app

    client = TestClient(app)
    resp = client.get("/api/weather/bucharest/lookahead?hours_ahead=2")
    assert resp.status_code == 200
    body = resp.json()
    assert body["lookahead_hours"] == 2
    assert "timestamp" in body
    assert "temp_c" in body
    assert "radiatie_solara_wm2" in body
    assert "offer_recommended" in body

