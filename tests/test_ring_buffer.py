import time
from datetime import datetime

from fastapi.testclient import TestClient

from server import app
from services.ring_buffer import RingBufferEngine, cycle_variation, month_window


def test_ring_buffer_month_rollover_and_step_back() -> None:
    """
    Verifică trecerea circulară (wrap-around) la finalul lunii (23:00 -> 00:00 luna următoare)
    și revenirea cu o oră înapoi în luna anterioară.
    """
    base_dt = datetime(2026, 11, 1, 0, 0, 0)  # Noiembrie are 30 zile = 720 ore
    _, m_hours, _ = month_window(0, base_dt)
    assert m_hours == 720

    rb = RingBufferEngine(initial_step=719, tick_interval_sec=0.5, base_dt=base_dt)
    try:
        assert rb["ring_cycle"] == 0
        assert rb["current_step"] == 719

        # Avansăm 1 oră: trebuie să treacă în ciclul 1 (Decembrie), pasul 0 (00:00)
        cyc, stp = rb.step_forward()
        assert cyc == 1
        assert stp == 0
        assert rb.virtual_index() == 720

        # Pas înapoi: trebuie să revină în ciclul 0 (Noiembrie), pasul 719 (23:00)
        cyc_b, stp_b = rb.step_back()
        assert cyc_b == 0
        assert stp_b == 719
    finally:
        rb.stop_daemon()


def test_background_daemon_thread_ticks_autonomously() -> None:
    """
    Verifică faptul că thread-ul de fundal `SCADA-RingBuffer-Daemon` avansează
    ceasul autonom în paralel și se oprește instant la `pause()`.
    """
    rb = RingBufferEngine(initial_step=10, tick_interval_sec=0.06)
    try:
        snap0 = rb.status_snapshot()
        assert snap0["thread_alive"] is True
        assert snap0["thread_name"] == "SCADA-RingBuffer-Daemon"
        assert snap0["is_running"] is False

        rb.start()
        time.sleep(0.22)
        rb.pause()

        snap1 = rb.status_snapshot()
        assert snap1["is_running"] is False
        assert snap1["tick_count"] >= 2
        assert snap1["current_step"] >= 12
    finally:
        rb.stop_daemon()


def test_background_daemon_autopause_human_in_the_loop() -> None:
    """
    Verifică oprirea automată (Human-in-the-Loop) direct din thread-ul daemon
    când apare o ofertă nouă la pasul țintă.
    """
    rb = RingBufferEngine(initial_step=5, tick_interval_sec=0.05)
    try:
        # Când ajunge la pasul 7, declanșăm auto-pauză
        rb.set_autopause_checker(
            lambda cyc, stp: "⏸ Ofertă nouă detectată la T+2" if stp >= 7 else None
        )
        rb.start()
        deadline = time.time() + 1.0
        while rb.is_running and time.time() < deadline:
            time.sleep(0.03)

        assert rb.is_running is False
        assert rb["current_step"] == 7
        toast = rb.pop_autopause_toast()
        assert toast is not None and "Ofertă nouă" in toast
    finally:
        rb.stop_daemon()


def test_cycle_variation_deterministic() -> None:
    """Ciclul 0 nu alterează datele; ciclurile >= 1 aplică micro-variație deterministă."""
    assert cycle_variation(10, 288) == (1.0, 0.0, 0.0)
    v1 = cycle_variation(300, 288)
    v2 = cycle_variation(300, 288)
    assert v1 == v2
    assert v1[0] != 1.0


def test_fastapi_ring_buffer_endpoints() -> None:
    """Verifică endpoint-urile REST `/api/ring-buffer/status` și `/api/ring-buffer/control`."""
    client = TestClient(app)
    r_status = client.get("/api/ring-buffer/status")
    assert r_status.status_code == 200
    body = r_status.json()
    assert body["thread_name"] == "SCADA-RingBuffer-Daemon"
    assert body["thread_alive"] is True

    r_step = client.post("/api/ring-buffer/control", json={"action": "set_step", "step": 15})
    assert r_step.status_code == 200
    assert r_step.json()["current_step"] == 15

    r_fwd = client.post("/api/ring-buffer/control", json={"action": "step_forward"})
    assert r_fwd.status_code == 200
    assert r_fwd.json()["current_step"] == 16
