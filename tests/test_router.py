import pytest

from services.router import (
    FIXED_TARIFF_EUR_MWH,
    POOL_ALL_DCS_EXHAUSTED,
    POOL_FULLY_ALLOCATED,
    POOL_OFFER_PENDING,
    OfferPool,
    load_profiles,
    match_score,
    rank_datacenters,
)


def _candidates():
    return [
        {"id": "ALPHA", "free_capacity_mw": 8.0, "temp_c": 21.0, "offers_sent": 40, "offers_accepted": 36},
        {"id": "BETA", "free_capacity_mw": 4.0, "temp_c": 22.0, "offers_sent": 40, "offers_accepted": 30},
        {"id": "GAMMA", "free_capacity_mw": 2.0, "temp_c": 23.0, "offers_sent": 40, "offers_accepted": 20},
    ]


def test_match_score_formula() -> None:
    assert match_score(1.0, 1.0, 0.0) == pytest.approx(0.8)
    assert match_score(0.5, 0.5, 1.0) == pytest.approx(0.2)


def test_ranking_orders_by_match_score() -> None:
    ranking = rank_datacenters(8.5, _candidates())
    assert [r["id"] for r in ranking] == ["ALPHA", "BETA", "GAMMA"]
    scores = [r["match_score"] for r in ranking]
    assert scores == sorted(scores, reverse=True)


def test_hvac_filter_makes_hot_node_ineligible() -> None:
    cands = _candidates()
    cands[0]["temp_c"] = 26.5  # peste limita HVAC de 26°C
    ranking = rank_datacenters(8.5, cands)
    assert ranking[-1]["id"] == "ALPHA"
    assert ranking[-1]["eligible"] is False
    # la exact 26°C nodul este încă eligibil (regula este temp <= 26°C)
    cands[0]["temp_c"] = 26.0
    assert rank_datacenters(8.5, cands)[0]["eligible"] is True


def test_rejection_reroutes_at_same_fixed_price() -> None:
    pool = OfferPool(8.5, rank_datacenters(8.5, _candidates()))
    first = pool.current_offer()
    assert first["dc_id"] == "ALPHA"
    assert first["tariff_eur_mwh"] == FIXED_TARIFF_EUR_MWH

    nxt = pool.respond("ALPHA", approved=False)
    assert nxt["dc_id"] == "BETA"
    assert nxt["tariff_eur_mwh"] == FIXED_TARIFF_EUR_MWH  # prețul NU scade
    assert pool.node_status("ALPHA") == "REJECTED"
    assert pool.status == POOL_OFFER_PENDING


def test_all_rejections_exhaust_pool() -> None:
    pool = OfferPool(8.5, rank_datacenters(8.5, _candidates()))
    for dc in ("ALPHA", "BETA", "GAMMA"):
        pool.respond(dc, approved=False)
    assert pool.current_offer() is None
    assert pool.status == POOL_ALL_DCS_EXHAUSTED


def test_approval_moves_node_to_computing_and_reduces_surplus() -> None:
    pool = OfferPool(6.0, rank_datacenters(6.0, _candidates()))
    pool.respond("ALPHA", approved=True)
    assert pool.node_status("ALPHA") == "COMPUTING"
    assert pool.remaining_mw == pytest.approx(0.0)
    assert pool.status == POOL_FULLY_ALLOCATED


def test_profiles_file_is_valid() -> None:
    profiles = load_profiles()
    assert profiles["fixed_tariff_eur_mwh"] == 30.0
    assert profiles["hvac_max_temp_c"] == 26.0
    assert len(profiles["datacenters"]) >= 3
    for dc in profiles["datacenters"]:
        assert dc["history"]["offers_accepted"] <= dc["history"]["offers_sent"]
