"""
Weather Service - Green Compute Orchestrator / VoltCompute
Integrare API meteo live (Open-Meteo) pentru prognoza meteo în zona București.
Transformă datele orare din API în formatul exact al fișierelor CSV utilizate de algoritmi:
  - 1_macro_vreme_piata.csv: [timestamp, temp_c, radiatie_solara_wm2, viteza_vant_kmh, pret_pzu_eur_mwh]
  - energy_datacenter_data.csv: [timestamp, hour, day_of_year, solar_irradiance_wm2, ambient_temp_celsius,
                                day_ahead_price_eur, grid_capacity_limit_mw, gross_solar_mw, surplus_mw]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Coordonate implicite pentru zona București (România)
BUCHAREST_LAT: float = 44.4323
BUCHAREST_LON: float = 26.1063
BUCHAREST_TIMEZONE: str = "Europe/Bucharest"

# Căi implicite către date
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MACRO_CSV_PATH = DATA_DIR / "1_macro_vreme_piata.csv"
DEFAULT_ENERGY_CSV_PATH = DATA_DIR / "energy_datacenter_data.csv"

# Structura sablon a campurilor din CSV
MACRO_COLUMNS = [
    "timestamp",
    "temp_c",
    "radiatie_solara_wm2",
    "viteza_vant_kmh",
    "pret_pzu_eur_mwh",
]

ENERGY_COLUMNS = [
    "timestamp",
    "hour",
    "day_of_year",
    "solar_irradiance_wm2",
    "ambient_temp_celsius",
    "day_ahead_price_eur",
    "grid_capacity_limit_mw",
    "gross_solar_mw",
    "surplus_mw",
]

# Mapare explicită: denumiri API Open-Meteo -> denumiri header CSV 1_macro_vreme_piata.csv
API_TO_MACRO_RENAME: Dict[str, str] = {
    "time": "timestamp",
    "temperature_2m": "temp_c",
    "shortwave_radiation": "radiatie_solara_wm2",
    "wind_speed_10m": "viteza_vant_kmh",
}

# Mapare explicită: denumiri API Open-Meteo -> denumiri header CSV energy_datacenter_data.csv
API_TO_ENERGY_RENAME: Dict[str, str] = {
    "time": "timestamp",
    "temperature_2m": "ambient_temp_celsius",
    "shortwave_radiation": "solar_irradiance_wm2",
}

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def calculate_pzu_price(
    hour: int,
    irradiance_wm2: float,
    seed: Optional[int] = None,
) -> float:
    """
    Estimează prețul energiei pe piața pentru ziua următoare (PZU / Day-Ahead în EUR/MWh)
    pe baza profilului orar și a curbei 'duck curve' românești (scădere puternică la prânz când
    iradianța solară generează supraofertă în rețea, și vârfuri de consum seara).
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    noise = float(rng.normal(0, 2.5))

    if 11 <= hour <= 15:
        if irradiance_wm2 > 650.0:
            # Prețuri foarte mici sau negative în momente de producție solară masivă
            base = max(-15.0, 25.0 - 0.045 * (irradiance_wm2 - 600.0))
        elif irradiance_wm2 > 300.0:
            base = 45.0 - 0.02 * (irradiance_wm2 - 300.0)
        else:
            base = 60.0
    elif 18 <= hour <= 21:
        # Vârful de sarcină de seară (peak evening)
        base = 120.0
    elif 7 <= hour <= 10:
        # Vârf de dimineață
        base = 80.0
    elif 0 <= hour <= 5:
        # Noaptea - consum redus
        base = 65.0
    else:
        base = 72.0

    return round(float(max(-20.0, base + noise)), 2)


def fetch_weather_forecast(
    latitude: float = BUCHAREST_LAT,
    longitude: float = BUCHAREST_LON,
    forecast_days: int = 7,
    past_days: int = 0,
    timezone: str = BUCHAREST_TIMEZONE,
    timeout: float = 12.0,
) -> Dict[str, Any]:
    """
    Interoghează API-ul Open-Meteo pentru prognoza orară pe coordonatele date (implicit București).
    Returnează răspunsul JSON complet conținând temperatură, radiație solară și viteza vântului.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "temperature_2m",
            "shortwave_radiation",
            "direct_normal_irradiance",
            "wind_speed_10m",
        ],
        "timezone": timezone,
        "forecast_days": max(1, min(16, int(forecast_days))),
    }
    if past_days > 0:
        params["past_days"] = min(92, int(past_days))

    logger.info(
        "Interogare API Open-Meteo pentru București (lat=%s, lon=%s, forecast_days=%s, past_days=%s)",
        latitude,
        longitude,
        forecast_days,
        past_days,
    )

    try:
        # Folosim urllib din biblioteca standard pentru compatibilitate totală
        full_url = f"{OPEN_METEO_FORECAST_URL}?{urllib.parse.urlencode(params, doseq=True)}"
        req = urllib.request.Request(
            full_url,
            headers={"User-Agent": "VoltCompute-WeatherService/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
            if "hourly" not in data or "time" not in data["hourly"]:
                raise ValueError("Răspunsul API Open-Meteo nu conține secțiunea 'hourly'.")
            return data
    except Exception as exc:
        logger.error("Eroare la apelarea API Open-Meteo: %s", exc)
        raise RuntimeError(f"Nu s-au putut prelua datele meteo de la Open-Meteo: {exc}") from exc


def fetch_weather_historical(
    start_date: str,
    end_date: str,
    latitude: float = BUCHAREST_LAT,
    longitude: float = BUCHAREST_LON,
    timezone: str = BUCHAREST_TIMEZONE,
    timeout: float = 15.0,
) -> Dict[str, Any]:
    """
    Interoghează API-ul Open-Meteo Archive pentru date meteo istorice pe un interval de date (YYYY-MM-DD).
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "shortwave_radiation",
            "direct_normal_irradiance",
            "wind_speed_10m",
        ],
        "timezone": timezone,
    }
    try:
        full_url = f"{OPEN_METEO_ARCHIVE_URL}?{urllib.parse.urlencode(params, doseq=True)}"
        req = urllib.request.Request(
            full_url,
            headers={"User-Agent": "VoltCompute-WeatherService/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Nu s-au putut prelua datele meteo de arhivă: {exc}") from exc


def parse_hourly_payload(
    api_data: Dict[str, Any],
) -> pd.DataFrame:
    """
    Parsează și normalizează secțiunea 'hourly' din payload-ul JSON primit de la Open-Meteo.
    Redenumește explicit cheile din API în numele din headerele CSV existente (1_macro_vreme_piata.csv),
    pentru a minimiza modificările de cod și a păstra compatibilitatea totală.
    """
    hourly = api_data.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise ValueError("Payload-ul nu conține câmpul 'hourly.time'.")

    # Mapăm și redenumim explicit cheile din API în numele din CSV
    df = pd.DataFrame(
        {
            "dt": pd.to_datetime(hourly["time"]),
            API_TO_MACRO_RENAME["time"]: hourly["time"],
            API_TO_MACRO_RENAME["temperature_2m"]: pd.to_numeric(
                hourly.get("temperature_2m", []), errors="coerce"
            ),
            API_TO_MACRO_RENAME["shortwave_radiation"]: pd.to_numeric(
                hourly.get("shortwave_radiation", []), errors="coerce"
            ),
            API_TO_MACRO_RENAME["wind_speed_10m"]: pd.to_numeric(
                hourly.get("wind_speed_10m", []), errors="coerce"
            ),
            "direct_normal_irradiance": pd.to_numeric(
                hourly.get("direct_normal_irradiance", []), errors="coerce"
            ),
        }
    )

    # Curățare și interpolare pentru eventuale valori lipsă
    df["temp_c"] = df["temp_c"].ffill().bfill().fillna(15.0)
    df["radiatie_solara_wm2"] = (
        df["radiatie_solara_wm2"].fillna(0.0).clip(lower=0.0)
    )
    df["viteza_vant_kmh"] = df["viteza_vant_kmh"].fillna(10.0).clip(lower=0.0)

    # Aliasuri pentru compatibilitate cu ambele formate (macro și energy)
    df["temperature_2m"] = df["temp_c"]
    df["shortwave_radiation"] = df["radiatie_solara_wm2"]
    df["wind_speed_10m"] = df["viteza_vant_kmh"]
    df["ambient_temp_celsius"] = df["temp_c"]
    df["solar_irradiance_wm2"] = df["radiatie_solara_wm2"]

    return df


def get_macro_weather_dataframe(
    api_data: Optional[Dict[str, Any]] = None,
    forecast_days: int = 7,
    past_days: int = 0,
    latitude: float = BUCHAREST_LAT,
    longitude: float = BUCHAREST_LON,
) -> pd.DataFrame:
    """
    Mulează răspunsul API meteo pe structura exactă din:
    data/1_macro_vreme_piata.csv
    
    Coloane returnate:
      - timestamp: string format 'YYYY-MM-DD HH:MM:SS'
      - temp_c: float (1 zecimală)
      - radiatie_solara_wm2: float (1 zecimală)
      - viteza_vant_kmh: float (1 zecimală)
      - pret_pzu_eur_mwh: float (2 zecimale)
    """
    if api_data is None:
        api_data = fetch_weather_forecast(
            latitude=latitude,
            longitude=longitude,
            forecast_days=forecast_days,
            past_days=past_days,
        )

    parsed = parse_hourly_payload(api_data)

    timestamps: List[str] = []
    temps: List[float] = []
    rads: List[float] = []
    winds: List[float] = []
    prices: List[float] = []

    for _, row in parsed.iterrows():
        dt: pd.Timestamp = row["dt"]
        timestamps.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
        t_val = round(float(row["temperature_2m"]), 1)
        temps.append(t_val)
        r_val = round(float(row["shortwave_radiation"]), 1)
        rads.append(r_val)
        w_val = round(float(row["wind_speed_10m"]), 1)
        winds.append(w_val)
        p_val = calculate_pzu_price(hour=dt.hour, irradiance_wm2=r_val)
        prices.append(p_val)

    macro_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temp_c": temps,
            "radiatie_solara_wm2": rads,
            "viteza_vant_kmh": winds,
            "pret_pzu_eur_mwh": prices,
        }
    )

    # Asigură ordinea exactă a coloanelor
    return macro_df[MACRO_COLUMNS]


def get_energy_datacenter_dataframe(
    api_data: Optional[Dict[str, Any]] = None,
    forecast_days: int = 7,
    past_days: int = 0,
    grid_limit_mw: float = 10.0,
    installed_capacity_mwp: float = 17.8,
    latitude: float = BUCHAREST_LAT,
    longitude: float = BUCHAREST_LON,
) -> pd.DataFrame:
    """
    Mulează răspunsul API meteo pe structura fișierului de simulare datacenter:
    data/energy_datacenter_data.csv
    
    Coloane returnate:
      - timestamp: string format ISO 8601 'YYYY-MM-DDTHH:MM:SS'
      - hour: int (0-23)
      - day_of_year: int (1-366)
      - solar_irradiance_wm2: float
      - ambient_temp_celsius: float
      - day_ahead_price_eur: float
      - grid_capacity_limit_mw: float
      - gross_solar_mw: float
      - surplus_mw: float
    """
    if api_data is None:
        api_data = fetch_weather_forecast(
            latitude=latitude,
            longitude=longitude,
            forecast_days=forecast_days,
            past_days=past_days,
        )

    parsed = parse_hourly_payload(api_data)

    records: List[Dict[str, Any]] = []
    for _, row in parsed.iterrows():
        dt: pd.Timestamp = row["dt"]
        hour = int(dt.hour)
        day_of_year = int(dt.day_of_year)
        ambient_temp = round(float(row["temperature_2m"]), 2)
        irradiance = round(float(row["shortwave_radiation"]), 2)
        da_price = calculate_pzu_price(hour=hour, irradiance_wm2=irradiance)

        # Calcul fizic al producției solare cu derating termic
        temp_derating = 1.0 - max(0.0, (ambient_temp - 25.0) * 0.004)
        gross_solar_mw = round((irradiance / 1000.0) * installed_capacity_mwp * temp_derating, 3)

        # Sarcină locală pe linie (1.0 - 2.4 MW în funcție de oră)
        local_load_mw = 1.4 + 0.5 * np.sin(np.pi * hour / 12.0)
        net_injection_mw = max(0.0, gross_solar_mw - local_load_mw)
        surplus_mw = round(float(max(0.0, net_injection_mw - grid_limit_mw)), 3)

        records.append(
            {
                "timestamp": dt.strftime("%Y-%m-%dT%H:%M:%S"),
                "hour": hour,
                "day_of_year": day_of_year,
                "solar_irradiance_wm2": irradiance,
                "ambient_temp_celsius": ambient_temp,
                "day_ahead_price_eur": da_price,
                "grid_capacity_limit_mw": float(grid_limit_mw),
                "gross_solar_mw": gross_solar_mw,
                "surplus_mw": surplus_mw,
            }
        )

    energy_df = pd.DataFrame(records)
    return energy_df[ENERGY_COLUMNS]


def fetch_bucharest_weather(
    forecast_days: int = 7,
    past_days: int = 1,
    format_type: str = "macro",
    latitude: float = BUCHAREST_LAT,
    longitude: float = BUCHAREST_LON,
) -> pd.DataFrame:
    """
    Interoghează API-ul Open-Meteo pentru zona parcului de panouri solare PPC din București.
    Permite interogarea simultană a datelor din trecut (past_days) și din viitor (forecast_days).
    Redenumește automat cheile API în coloanele existente în CSV-urile aplicației:
      - format_type="macro": conform 1_macro_vreme_piata.csv [timestamp, temp_c, radiatie_solara_wm2, viteza_vant_kmh, pret_pzu_eur_mwh]
      - format_type="energy": conform energy_datacenter_data.csv [timestamp, hour, day_of_year, solar_irradiance_wm2, ambient_temp_celsius, ...]
    """
    if format_type == "energy":
        return get_energy_datacenter_dataframe(
            forecast_days=forecast_days,
            past_days=past_days,
            latitude=latitude,
            longitude=longitude,
        )
    return get_macro_weather_dataframe(
        forecast_days=forecast_days,
        past_days=past_days,
        latitude=latitude,
        longitude=longitude,
    )


def get_weather_lookahead(
    hours_ahead: int = 2,
    current_time: Optional[Union[str, datetime]] = None,
    latitude: float = BUCHAREST_LAT,
    longitude: float = BUCHAREST_LON,
    timezone: str = BUCHAREST_TIMEZONE,
    grid_limit_mw: float = 10.0,
    installed_capacity_mwp: float = 17.8,
) -> Dict[str, Any]:
    """
    Furnizează datele meteo peste București (locația parcului solar PPC) în avans cu `hours_ahead` (implicit 2 ore),
    permițând formularea și plasarea ofertelor de flexibilitate înainte ca dezechilibrul să apară în rețea.
    
    Returnează cheile redenumite conform headerului CSV de vreme (1_macro_vreme_piata.csv),
    alături de indicatorul de oportunitate de ofertă (surplus estimat, offer_recommended).
    """
    if current_time is None:
        ref_dt = datetime.now()
    elif isinstance(current_time, str):
        ref_dt = pd.to_datetime(current_time).to_pydatetime()
    else:
        ref_dt = current_time

    forecast_days = max(2, int(np.ceil((hours_ahead + 24) / 24.0)))
    macro_df = get_macro_weather_dataframe(
        forecast_days=forecast_days,
        past_days=1,
        latitude=latitude,
        longitude=longitude,
    )
    macro_df["dt"] = pd.to_datetime(macro_df["timestamp"])

    target_dt = (ref_dt + pd.Timedelta(hours=hours_ahead)).replace(minute=0, second=0, microsecond=0)
    matching = macro_df[macro_df["dt"] == target_dt]
    if len(matching) > 0:
        row = matching.iloc[0]
    else:
        diffs = (macro_df["dt"] - target_dt).abs()
        row = macro_df.iloc[int(diffs.argmin())]

    temp_c = float(row["temp_c"])
    rad_wm2 = float(row["radiatie_solara_wm2"])
    wind_kmh = float(row["viteza_vant_kmh"])
    pzu_eur = float(row["pret_pzu_eur_mwh"])
    target_hour = int(row["dt"].hour)

    # Calcul al producției solare PPC la orizontul T + 2h
    temp_derating = 1.0 - max(0.0, (temp_c - 25.0) * 0.004)
    gross_solar_mw = round((rad_wm2 / 1000.0) * installed_capacity_mwp * temp_derating, 3)
    local_load_mw = 1.4 + 0.5 * np.sin(np.pi * target_hour / 12.0)
    net_injection_mw = max(0.0, gross_solar_mw - local_load_mw)
    surplus_mw = round(float(max(0.0, net_injection_mw - grid_limit_mw)), 3)
    offer_recommended = bool(surplus_mw > 0.1)

    return {
        "location": "București (PPC Solar)",
        "latitude": latitude,
        "longitude": longitude,
        "query_time": ref_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "lookahead_hours": hours_ahead,
        "target_timestamp": str(row["timestamp"]),
        "target_hour": target_hour,
        # Numele exacte din headerul CSV 1_macro_vreme_piata.csv:
        "timestamp": str(row["timestamp"]),
        "temp_c": temp_c,
        "radiatie_solara_wm2": rad_wm2,
        "viteza_vant_kmh": wind_kmh,
        "pret_pzu_eur_mwh": pzu_eur,
        # Numele compatibile cu energy_datacenter_data.csv și motorul de decizie:
        "ambient_temp_celsius": temp_c,
        "solar_irradiance_wm2": rad_wm2,
        "day_ahead_price_eur": pzu_eur,
        "gross_solar_mw": gross_solar_mw,
        "net_injection_mw": round(float(net_injection_mw), 3),
        "surplus_mw": surplus_mw,
        "grid_limit_mw": float(grid_limit_mw),
        "offer_recommended": offer_recommended,
    }


def sync_weather_to_csv(
    macro_csv_path: Union[str, Path] = DEFAULT_MACRO_CSV_PATH,
    energy_csv_path: Union[str, Path] = DEFAULT_ENERGY_CSV_PATH,
    forecast_days: int = 7,
    past_days: int = 0,
    backup: bool = True,
    update_macro: bool = True,
    update_energy: bool = True,
) -> Dict[str, str]:
    """
    Preia prognoza pe zona București de pe API și actualizează fișierele CSV,
    păstrând 100% schema existentă și făcând opțional copie de siguranță (.bak).
    """
    macro_csv_path = Path(macro_csv_path)
    energy_csv_path = Path(energy_csv_path)

    api_data = fetch_weather_forecast(
        latitude=BUCHAREST_LAT,
        longitude=BUCHAREST_LON,
        forecast_days=forecast_days,
        past_days=past_days,
    )

    results: Dict[str, str] = {}

    if update_macro:
        macro_df = get_macro_weather_dataframe(api_data=api_data)
        if backup and macro_csv_path.exists():
            bak_path = macro_csv_path.with_suffix(".csv.bak")
            shutil.copy2(macro_csv_path, bak_path)
        macro_csv_path.parent.mkdir(parents=True, exist_ok=True)
        macro_df.to_csv(macro_csv_path, index=False)
        results["macro_csv"] = f"{macro_csv_path} ({len(macro_df)} rânduri scrise din API)"
        logger.info("[✓] Macro CSV actualizat din API: %s", macro_csv_path)

    if update_energy:
        energy_df = get_energy_datacenter_dataframe(api_data=api_data)
        if backup and energy_csv_path.exists():
            bak_path = energy_csv_path.with_suffix(".csv.bak")
            shutil.copy2(energy_csv_path, bak_path)
        energy_csv_path.parent.mkdir(parents=True, exist_ok=True)
        energy_df.to_csv(energy_csv_path, index=False)
        results["energy_csv"] = f"{energy_csv_path} ({len(energy_df)} rânduri scrise din API)"
        logger.info("[✓] Energy CSV actualizat din API: %s", energy_csv_path)

    return results


def load_macro_weather(
    macro_csv_path: Union[str, Path] = DEFAULT_MACRO_CSV_PATH,
    use_api: bool = True,
    forecast_days: int = 7,
    past_days: int = 0,
) -> Tuple[pd.DataFrame, str]:
    """
    Încarcă datele macro de vreme: încearcă întâi prin API de vreme pentru București;
    dacă rețeaua este indisponibilă sau use_api=False, face fallback automat la CSV-ul local.
    """
    macro_path = Path(macro_csv_path)
    if use_api:
        try:
            df = get_macro_weather_dataframe(
                forecast_days=forecast_days,
                past_days=past_days,
            )
            return df, f"Open-Meteo Weather API (București, {len(df)}h prognoză)"
        except Exception as exc:
            logger.warning("Nu s-a putut apela API-ul meteo (%s). Fallback la CSV local...", exc)

    if macro_path.exists():
        df = pd.read_csv(macro_path)
        return df, f"Local CSV ({macro_path})"

    raise FileNotFoundError(f"Fișierul CSV {macro_path} nu a fost găsit și API-ul nu a putut fi interogat.")


def load_energy_weather(
    energy_csv_path: Union[str, Path] = DEFAULT_ENERGY_CSV_PATH,
    use_api: bool = True,
    forecast_days: int = 7,
    past_days: int = 0,
) -> Tuple[pd.DataFrame, str]:
    """
    Încarcă datele de telemetrie pentru simulare: încearcă întâi prin API de vreme pentru București;
    dacă rețeaua este indisponibilă sau use_api=False, face fallback pe CSV-ul existent.
    """
    energy_path = Path(energy_csv_path)
    if use_api:
        try:
            df = get_energy_datacenter_dataframe(
                forecast_days=forecast_days,
                past_days=past_days,
            )
            return df, f"Open-Meteo Weather API (București, {len(df)}h prognoză)"
        except Exception as exc:
            logger.warning("Nu s-a putut apela API-ul meteo (%s). Fallback la CSV local...", exc)

    if energy_path.exists():
        df = pd.read_csv(energy_path)
        return df, f"Local CSV ({energy_path})"

    raise FileNotFoundError(f"Fișierul CSV {energy_path} nu a fost găsit și API-ul nu a putut fi interogat.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preluare date meteo București prin API și mapare pe structura CSV existentă."
    )
    parser.add_argument(
        "--forecast-days",
        type=int,
        default=7,
        help="Număr de zile de prognoză (1 - 16, implicit 7).",
    )
    parser.add_argument(
        "--past-days",
        type=int,
        default=0,
        help="Număr de zile din trecut (0 - 92, implicit 0).",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Salvează datele primite din API direct în fișierele CSV.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Afișează primele 10 rânduri ale prognozei fără a scrie pe disc.",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=None,
        help="Afișează prognoza în avans cu N ore (ex: 2 ore) pentru ofertare.",
    )
    parser.add_argument(
        "--output-macro",
        type=str,
        default=str(DEFAULT_MACRO_CSV_PATH),
        help="Calea către fișierul macro CSV de actualizat.",
    )
    parser.add_argument(
        "--output-energy",
        type=str,
        default=str(DEFAULT_ENERGY_CSV_PATH),
        help="Calea către fișierul energy datacenter CSV de actualizat.",
    )

    args = parser.parse_args()

    print(f"[*] Interogare prognoză meteo pentru București ({BUCHAREST_LAT}°N, {BUCHAREST_LON}°E)...")
    macro_data = get_macro_weather_dataframe(
        forecast_days=args.forecast_days,
        past_days=args.past_days,
    )
    print(f"[+] Date preluate cu succes: {len(macro_data)} intervale orare.")
    print("\n--- Primele 5 înregistrări (format 1_macro_vreme_piata.csv) ---")
    print(macro_data.head())

    if args.preview:
        print("\n--- Primele 10 înregistrări ---")
        print(macro_data.head(10))

    if args.lookahead is not None:
        look = get_weather_lookahead(hours_ahead=args.lookahead)
        print(f"\n[⚡] Prognoză meteo în avans cu {args.lookahead} ore (T+{args.lookahead}h) pentru ofertare:")
        for k, v in look.items():
            print(f"  {k}: {v}")

    if args.sync:
        res = sync_weather_to_csv(
            macro_csv_path=args.output_macro,
            energy_csv_path=args.output_energy,
            forecast_days=args.forecast_days,
            past_days=args.past_days,
            backup=True,
        )
        print("\n[✓] Fișiere CSV actualizate cu succes din API:")
        for k, v in res.items():
            print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
