"""Interactive Streamlit checks using the actual session-owned agent."""
from pathlib import Path
from streamlit.testing.v1 import AppTest


def transaction(app):
    state = app.session_state
    return next(value for key, value in state["transactions"].items()
                if key[:2] == (state["scenario"], state["current_step"]))


def test_dashboard_negotiation_acceptance_persistence_and_safety():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
    assert not app.exception
    app.button(key="scenario_CRITICAL_SOLAR_SUNDAY").click().run()
    tx = transaction(app)
    assert tx["injection"] == 8
    assert tx["forecast"].prediction_source == "xgboost"
    offer = next(o for o in tx["decision"].dc_offers if o.offered_mw)
    prefix = f'{tx["decision"].decision_id}_{offer.dc_id}'
    app.button(key=f"{prefix}_DUTCH_AUCTION_STEP").click().run()
    assert next(o for o in transaction(app)["decision"].dc_offers if o.dc_id == offer.dc_id).offered_price_eur == 20
    app.button(key=f"{prefix}_APPROVED").click().run()
    total = transaction(app)["decision"].executed_mw
    assert total > 0
    assert transaction(app)["result"]["webhook_sent"] is False
    assert app.button(key=f"{prefix}_APPROVED").disabled
    app.button(key="scenario_UNCERTAIN_WEATHER_SHOCK").click().run()
    assert transaction(app)["decision"].status == "ESCALATE_TO_HUMAN"
    assert all(button.disabled for button in app.button if button.key.endswith("_APPROVED"))
    app.button(key="scenario_CRITICAL_SOLAR_SUNDAY").click().run()
    assert transaction(app)["decision"].executed_mw == total
    other = next(o for o in transaction(app)["decision"].dc_offers if o.status == "OFFER_GENERATED")
    key = f'{transaction(app)["decision"].decision_id}_{other.dc_id}_REJECTED'
    app.button(key=key).click().run()
    assert transaction(app)["decision"].executed_mw == total
    assert transaction(app)["decision"].fallback.ppc_blue_ev_mw > 0
    dec_after_reject = transaction(app)["decision"]
    fb_key = f"{dec_after_reject.decision_id}_PPC_FALLBACK_r{dec_after_reject.fallback_plan_revision}_APPROVED"
    app.button(key=fb_key).click().run()
    assert transaction(app)["decision"].executed_ppc_blue_ev_mw > 0
    app.button(key="scenario_DATACENTER_THERMAL_RISK").click().run()
    assert any(o.status == "REJECTED_THERMAL_SAFETY" for o in transaction(app)["decision"].dc_offers)
    assert not app.exception
