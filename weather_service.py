"""
Weather Service root shortcut module.
Permite importul direct `import weather_service` sau rularea din terminal `python3 weather_service.py`.
"""

import sys
import subprocess
from pathlib import Path

# Dacă scriptul este apelat direct din terminal fără activarea mediului virtual (.venv),
# dar .venv există local, repornim automat procesul folosind interpretorul Python din .venv.
_venv_dir = Path(__file__).resolve().parent / ".venv"
_venv_python = _venv_dir / "bin" / "python3"
if __name__ == "__main__" and _venv_python.exists() and sys.prefix != str(_venv_dir.resolve()):
    sys.exit(subprocess.call([str(_venv_python), *sys.argv]))

from services.weather_service import (
    API_TO_ENERGY_RENAME,
    API_TO_MACRO_RENAME,
    BUCHAREST_LAT,
    BUCHAREST_LON,
    BUCHAREST_TIMEZONE,
    DEFAULT_ENERGY_CSV_PATH,
    DEFAULT_MACRO_CSV_PATH,
    ENERGY_COLUMNS,
    MACRO_COLUMNS,
    calculate_pzu_price,
    fetch_bucharest_weather,
    fetch_weather_forecast,
    fetch_weather_historical,
    get_energy_datacenter_dataframe,
    get_macro_weather_dataframe,
    get_weather_lookahead,
    load_energy_weather,
    load_macro_weather,
    main,
    sync_weather_to_csv,
)

if __name__ == "__main__":
    main()
