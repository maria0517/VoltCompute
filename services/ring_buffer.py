import calendar
import ctypes
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class _SYSTEMTIME(ctypes.Structure):
    """Structura nativă Win32 SYSTEMTIME pentru interogarea ceasului local din Windows API."""

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
    """
    Interoghează API-ul nativ Windows (kernel32.GetLocalTime) pentru a obține anul și luna curentă
    și returnează prima oră a lunii (ziua 1, 00:00). Include fallback cross-platform pentru Linux/Docker.
    """
    try:
        if hasattr(ctypes, "windll") and hasattr(ctypes.windll, "kernel32"):
            st_time = _SYSTEMTIME()
            ctypes.windll.kernel32.GetLocalTime(ctypes.byref(st_time))
            if 1 <= int(st_time.wMonth) <= 12 and int(st_time.wYear) >= 2020:
                return datetime(int(st_time.wYear), int(st_time.wMonth), 1, 0, 0, 0)
    except Exception:
        pass
    now = datetime.now()
    return datetime(now.year, now.month, 1, 0, 0, 0)


BASE_LIVE_DT: datetime = get_windows_month_start()


def month_window(cycle: int, base_dt: datetime = BASE_LIVE_DT) -> Tuple[datetime, int, int]:
    """
    Returnează (data_start_lună, număr_ore_în_lună, offset_ore_față_de_BASE_LIVE_DT)
    pentru luna corespunzătoare ciclului `cycle` (0 = luna de start, 1 = luna următoare etc.).
    """
    total_month_idx = (base_dt.month - 1) + int(cycle)
    year = base_dt.year + (total_month_idx // 12)
    month = (total_month_idx % 12) + 1
    days_in_month = calendar.monthrange(year, month)[1]
    month_start = datetime(year, month, 1, 0, 0, 0)
    offset_hours = int(round((month_start - base_dt).total_seconds() / 3600.0))
    return month_start, days_in_month * 24, offset_hours


def ring_dt(v_idx: int, base_dt: datetime = BASE_LIVE_DT) -> datetime:
    """Returnează timestamp-ul continuu monoton pentru indexul virtual `v_idx`."""
    return base_dt + timedelta(hours=int(v_idx))


def cycle_variation(v_idx: int, n_feed: int) -> Tuple[float, float, float]:
    """
    La ciclurile ulterioare ale Ring Buffer-ului (`v_cycle != 0`), aplică o micro-variație
    stocastică deterministă (nor tranzitoriu + fluctuație de sarcină) astfel încât fluxul live
    să genereze mereu date noi, fără să repete identic aceiași megawați.
    """
    if n_feed <= 0:
        return 1.0, 0.0, 0.0
    v_cycle = int(v_idx) // int(n_feed)
    if v_cycle == 0:
        return 1.0, 0.0, 0.0
    rng = np.random.default_rng(seed=100_003 + int(v_idx) * 31)
    irr_factor = float(np.clip(1.0 + rng.normal(0.0, 0.08), 0.78, 1.22))
    temp_delta = float(rng.normal(0.0, 1.1))
    price_delta = float(rng.normal(0.0, 4.5))
    return irr_factor, temp_delta, price_delta


class RingBufferEngine:
    """
    Motor concurent de tip Ring Buffer cu thread daemon de fundal (`SCADA-RingBuffer-Daemon`).

    - Rulează autonom pe un thread separat față de UI-ul Streamlit/FastAPI.
    - Asigură sincronizarea prin `threading.RLock()` între toate sesiunile deschise
      (Dispecerat PPC, Harta SCADA 400 kV și terminalele Data Center).
    - Când ajunge la ultima oră a lunii (23:00), face wrap-around automat în ziua 1 ora 00:00
      a lunii următoare (`ring_cycle += 1`), aplicând micro-variații stocastice la epuizarea
      setului de bază.
    - Suportă oprirea automată Human-in-the-Loop (auto-pause) atunci când apare o ofertă nouă
      T+2 care necesită decizia operatorului.
    """

    THREAD_NAME = "SCADA-RingBuffer-Daemon"

    def __init__(
        self,
        initial_step: int = 11,
        datacenters: Optional[List[Dict[str, Any]]] = None,
        tick_interval_sec: float = 1.0,
        base_dt: datetime = BASE_LIVE_DT,
    ) -> None:
        self._lock = threading.RLock()
        self._base_dt = base_dt
        self._tick_interval_sec = max(0.05, float(tick_interval_sec))
        self._stop_event = threading.Event()
        self._autopause_checker: Optional[Callable[[int, int], Optional[str]]] = None

        _, init_month_hours, _ = month_window(0, self._base_dt)
        safe_step = int(initial_step) % max(1, init_month_hours)

        dc_list = datacenters or []
        self._state: Dict[str, Any] = {
            "lock": self._lock,
            "current_step": safe_step,
            "ring_cycle": 0,
            "is_running": False,
            "tick_count": 0,
            "last_tick_ts": None,
            "autopause_toast": None,
            "autopaused_targets": set(),
            "deals": {},
            "refused": {},
            "dc_history": {
                n["id"]: dict(n.get("history", {"offers_sent": 0, "offers_accepted": 0}))
                for n in dc_list
            },
        }

        self._thread = threading.Thread(
            target=self._daemon_loop,
            name=self.THREAD_NAME,
            daemon=True,
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Interfață compatibilă cu dicționarul partajat SH[...]
    # ------------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return self._state[key]

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            self._state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._state.get(key, default)

    def setdefault(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._state.setdefault(key, default)

    def pop(self, key: str, *args: Any) -> Any:
        with self._lock:
            return self._state.pop(key, *args)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._state

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    @property
    def is_running(self) -> bool:
        with self._lock:
            return bool(self._state["is_running"])

    @property
    def tick_interval_sec(self) -> float:
        return self._tick_interval_sec

    @tick_interval_sec.setter
    def tick_interval_sec(self, value: float) -> None:
        with self._lock:
            self._tick_interval_sec = max(0.05, float(value))

    # ------------------------------------------------------------------
    # Controlul ceasului și al Ring Buffer-ului
    # ------------------------------------------------------------------
    def set_autopause_checker(
        self, checker: Optional[Callable[[int, int], Optional[str]]]
    ) -> None:
        """
        Înregistrează un callback `checker(ring_cycle, current_step) -> Optional[str]`.
        Dacă returnează un mesaj nenul în timpul rulării Live, thread-ul pune automat pauză
        (Human-in-the-Loop) înainte de a trece la ora următoare.
        """
        with self._lock:
            self._autopause_checker = checker

    def start(self) -> None:
        """Pornește avansarea automată în thread-ul de fundal."""
        with self._lock:
            self._state["is_running"] = True

    def pause(self, toast_msg: Optional[str] = None) -> None:
        """Oprește avansarea automată în thread-ul de fundal."""
        with self._lock:
            self._state["is_running"] = False
            if toast_msg:
                self._state["autopause_toast"] = toast_msg

    def toggle(self) -> bool:
        """Comută între Play și Pause; returnează noua stare `is_running`."""
        with self._lock:
            new_state = not bool(self._state["is_running"])
            self._state["is_running"] = new_state
            return new_state

    def pop_autopause_toast(self) -> Optional[str]:
        """Returnează și consumă ultimul mesaj de auto-pauză generat de thread."""
        with self._lock:
            msg = self._state.get("autopause_toast")
            self._state["autopause_toast"] = None
            return msg

    def _advance_one_hour_unlocked(self) -> Tuple[int, int]:
        """Avansează cu o oră în Ring Buffer (apelat sub lock)."""
        cur_cycle = int(self._state["ring_cycle"])
        _, cur_month_hours, _ = month_window(cur_cycle, self._base_dt)
        nxt_step = int(self._state["current_step"]) + 1
        if nxt_step >= cur_month_hours:
            self._state["current_step"] = 0
            self._state["ring_cycle"] = cur_cycle + 1
        else:
            self._state["current_step"] = nxt_step
        return int(self._state["ring_cycle"]), int(self._state["current_step"])

    def step_forward(self, stop_autoplay: bool = True) -> Tuple[int, int]:
        """Avansează manual cu o oră în Ring Buffer."""
        with self._lock:
            if stop_autoplay:
                self._state["is_running"] = False
            return self._advance_one_hour_unlocked()

    def step_back(self) -> Tuple[int, int]:
        """Revine cu o oră în Ring Buffer (00:00 -> 23:00 a lunii anterioare)."""
        with self._lock:
            self._state["is_running"] = False
            cur_cycle = int(self._state["ring_cycle"])
            prv_step = int(self._state["current_step"]) - 1
            if prv_step < 0:
                prev_cycle = cur_cycle - 1
                _, prev_month_hours, _ = month_window(prev_cycle, self._base_dt)
                self._state["ring_cycle"] = prev_cycle
                self._state["current_step"] = prev_month_hours - 1
            else:
                self._state["current_step"] = prv_step
            return int(self._state["ring_cycle"]), int(self._state["current_step"])

    def set_step(self, step: int, stop_autoplay: bool = True) -> Tuple[int, int]:
        """Setează direct ora curentă în luna activă (ex. din slider-ul de timeline)."""
        with self._lock:
            if stop_autoplay:
                self._state["is_running"] = False
            cur_cycle = int(self._state["ring_cycle"])
            _, cur_month_hours, _ = month_window(cur_cycle, self._base_dt)
            self._state["current_step"] = int(min(max(0, int(step)), cur_month_hours - 1))
            return cur_cycle, int(self._state["current_step"])

    def virtual_index(self) -> int:
        """Returnează indexul virtual continuu (offset_lună + current_step)."""
        with self._lock:
            cur_cycle = int(self._state["ring_cycle"])
            cur_step = int(self._state["current_step"])
            _, _, offset_hours = month_window(cur_cycle, self._base_dt)
            return offset_hours + cur_step

    def status_snapshot(self) -> Dict[str, Any]:
        """Returnează telemetria completă a Ring Buffer-ului și a thread-ului daemon."""
        with self._lock:
            cur_cycle = int(self._state["ring_cycle"])
            cur_step = int(self._state["current_step"])
            m_start, m_hours, offset_hours = month_window(cur_cycle, self._base_dt)
            v_idx = offset_hours + cur_step
            cur_dt = ring_dt(v_idx, self._base_dt)
            return {
                "thread_name": self._thread.name,
                "thread_ident": self._thread.ident,
                "thread_alive": self._thread.is_alive(),
                "is_running": bool(self._state["is_running"]),
                "tick_count": int(self._state["tick_count"]),
                "tick_interval_sec": self._tick_interval_sec,
                "last_tick_ts": self._state["last_tick_ts"],
                "ring_cycle": cur_cycle,
                "current_step": cur_step,
                "month_hours": m_hours,
                "month_label": m_start.strftime("%Y-%m"),
                "virtual_index": v_idx,
                "current_timestamp": cur_dt.strftime("%Y-%m-%d %H:%M"),
            }

    def stop_daemon(self) -> None:
        """Oprește definitiv thread-ul daemon (util în teste unitare)."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.5)

    # ------------------------------------------------------------------
    # Bucla thread-ului de fundal
    # ------------------------------------------------------------------
    def _daemon_loop(self) -> None:
        """
        Bucla principală a thread-ului daemon `SCADA-RingBuffer-Daemon`.
        Așteaptă în pași fini (50ms) pentru reacție instantanee la Start/Pause,
        și avansează ceasul rețelei la fiecare `tick_interval_sec` secunde.
        """
        elapsed = 0.0
        slice_sec = 0.05
        while not self._stop_event.is_set():
            time.sleep(slice_sec)
            with self._lock:
                running = bool(self._state["is_running"])
                interval = self._tick_interval_sec
            if not running:
                elapsed = 0.0
                continue

            elapsed += slice_sec
            if elapsed + 1e-6 < interval:
                continue
            elapsed = 0.0

            with self._lock:
                if not self._state["is_running"]:
                    continue
                checker = self._autopause_checker
                cur_cycle = int(self._state["ring_cycle"])
                cur_step = int(self._state["current_step"])

            # Verificăm dacă există o ofertă nouă care cere intervenția dispecerului (Human-in-the-Loop)
            pause_reason: Optional[str] = None
            if checker is not None:
                try:
                    pause_reason = checker(cur_cycle, cur_step)
                except Exception:
                    pause_reason = None

            with self._lock:
                if not self._state["is_running"]:
                    continue
                if pause_reason:
                    self._state["is_running"] = False
                    self._state["autopause_toast"] = pause_reason
                    continue

                self._advance_one_hour_unlocked()
                self._state["tick_count"] = int(self._state["tick_count"]) + 1
                self._state["last_tick_ts"] = datetime.now().isoformat()


_GLOBAL_ENGINE_LOCK = threading.Lock()
_GLOBAL_ENGINE: Optional[RingBufferEngine] = None


def get_shared_ring_buffer(
    initial_step: int = 11,
    datacenters: Optional[List[Dict[str, Any]]] = None,
    tick_interval_sec: float = 1.0,
) -> RingBufferEngine:
    """
    Returnează instanța singleton a motorului `RingBufferEngine` partajată la nivel de proces
    între toate sesiunile Streamlit (`app.py`, `pages/1_Harta_Retea_SCADA.py`) și serverul FastAPI.
    """
    global _GLOBAL_ENGINE
    with _GLOBAL_ENGINE_LOCK:
        if _GLOBAL_ENGINE is None or not _GLOBAL_ENGINE.status_snapshot()["thread_alive"]:
            _GLOBAL_ENGINE = RingBufferEngine(
                initial_step=initial_step,
                datacenters=datacenters,
                tick_interval_sec=tick_interval_sec,
            )
        elif datacenters:
            with _GLOBAL_ENGINE.lock:
                dc_hist = _GLOBAL_ENGINE.setdefault("dc_history", {})
                for n in datacenters:
                    dc_hist.setdefault(
                        n["id"],
                        dict(n.get("history", {"offers_sent": 0, "offers_accepted": 0})),
                    )
        return _GLOBAL_ENGINE
