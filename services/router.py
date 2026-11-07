"""
Pasul 2 (DECIDE) — Router determinist pentru Data Center-e.

De ce NU un al doilea model ML: predicția de surplus are deja o marjă de eroare; un al doilea
model probabilistic peste ea ar compune incertitudinea. Data Center-ele sunt active IT cu
parametri clari (capacitate, temperatură, istoric de contracte), deci le clasăm euristic:

    Score_i = 0.4 * VolumFit_i + 0.4 * RataAcceptareIstorica_i - 0.2 * PenalizareTermica_i

Filtru dur de siguranță: temperatura HVAC <= 26°C (altfel nodul este INELIGIBIL).

Regula de ofertare în pool (Pasul 3, ACT):
  * oferta pleacă la primul clasat, la tariful FIX de 30 EUR/MWh;
  * dacă refuză, tariful NU scade (fără dumping / licitație olandeză), nodul devine inactiv
    și oferta sare automat la următorul nod eligibil, la același preț;
  * dacă refuză toți -> ALL_DCS_EXHAUSTED (intervenția dispeceratului).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROFILES_PATH = Path(__file__).resolve().parent.parent / "data" / "dc_profiles_and_history.json"

FIXED_TARIFF_EUR_MWH = 30.0
HVAC_MAX_TEMP_C = 26.0
COMFORT_TEMP_C = 20.0  # sub această temperatură penalizarea termică este 0

W_VOLUME_FIT = 0.4
W_ACCEPTANCE = 0.4
W_THERMAL = 0.2

MIN_OFFER_MW = 0.05

# Stările pool-ului de ofertare
POOL_NO_SURPLUS = "NO_SURPLUS"
POOL_OFFER_PENDING = "OFFER_PENDING"
POOL_FULLY_ALLOCATED = "FULLY_ALLOCATED"
POOL_ALL_DCS_EXHAUSTED = "ALL_DCS_EXHAUSTED"


def load_profiles(path: Path = PROFILES_PATH) -> Dict[str, Any]:
    """Încarcă `dc_profiles_and_history.json`."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def acceptance_rate(offers_sent: int, offers_accepted: int) -> float:
    """Rata istorică de acceptare cu netezire Laplace (un nod nou pornește de la 0.5)."""
    return (float(offers_accepted) + 1.0) / (float(offers_sent) + 2.0)


def volume_fit(surplus_mw: float, free_capacity_mw: float) -> float:
    """Ce fracție din surplus poate absorbi nodul (0..1)."""
    if surplus_mw <= 0:
        return 0.0
    return max(0.0, min(free_capacity_mw, surplus_mw)) / surplus_mw


def thermal_penalty(temp_c: float, limit_c: float = HVAC_MAX_TEMP_C) -> float:
    """0 la temperatură confortabilă (<= 20°C), 1 la limita HVAC."""
    span = max(limit_c - COMFORT_TEMP_C, 1e-6)
    return min(1.0, max(0.0, (temp_c - COMFORT_TEMP_C) / span))


def match_score(vol_fit: float, acc_rate: float, therm_pen: float) -> float:
    return W_VOLUME_FIT * vol_fit + W_ACCEPTANCE * acc_rate - W_THERMAL * therm_pen


def rank_datacenters(
    surplus_mw: float,
    candidates: Iterable[Dict[str, Any]],
    hvac_max_temp_c: float = HVAC_MAX_TEMP_C,
) -> List[Dict[str, Any]]:
    """
    Clasează nodurile pentru un surplus dat. Fiecare candidat are cheile:
      id, free_capacity_mw, temp_c, offers_sent, offers_accepted (+ orice alte câmpuri, păstrate).
    Returnează lista sortată: nodurile eligibile după Match Score descrescător, apoi cele ineligibile.
    """
    ranked: List[Dict[str, Any]] = []
    for c in candidates:
        temp = float(c["temp_c"])
        cap = max(0.0, float(c["free_capacity_mw"]))
        limit = float(c.get("thermal_limit_c", hvac_max_temp_c))
        limit = min(limit, hvac_max_temp_c)
        vf = volume_fit(surplus_mw, cap)
        ar = acceptance_rate(int(c.get("offers_sent", 0)), int(c.get("offers_accepted", 0)))
        tp = thermal_penalty(temp, limit)

        if temp > limit:
            eligible, reason = False, f"HVAC {temp:.1f}°C > {limit:.0f}°C"
        elif cap < MIN_OFFER_MW:
            eligible, reason = False, "fără capacitate liberă"
        else:
            eligible, reason = True, ""

        ranked.append(
            {
                **c,
                "volume_fit": round(vf, 4),
                "acceptance_rate": round(ar, 4),
                "thermal_penalty": round(tp, 4),
                "match_score": round(match_score(vf, ar, tp), 4),
                "eligible": eligible,
                "ineligible_reason": reason,
            }
        )
    ranked.sort(key=lambda r: (not r["eligible"], -r["match_score"], str(r["id"])))
    return ranked


def next_offer(
    ranking: List[Dict[str, Any]],
    remaining_mw: float,
    responded_ids: Iterable[str],
    tariff_eur_mwh: float = FIXED_TARIFF_EUR_MWH,
) -> Optional[Dict[str, Any]]:
    """
    Primul nod eligibil care nu a răspuns încă primește oferta, mereu la același tarif fix.
    Returnează None dacă nu mai există niciun nod disponibil.
    """
    if remaining_mw < MIN_OFFER_MW:
        return None
    responded = set(responded_ids)
    for node in ranking:
        if not node["eligible"] or node["id"] in responded:
            continue
        mw = round(min(remaining_mw, float(node["free_capacity_mw"])), 3)
        if mw >= MIN_OFFER_MW:
            return {"dc_id": node["id"], "mw": mw, "tariff_eur_mwh": tariff_eur_mwh}
    return None


class OfferPool:
    """
    Mașina de stări a pool-ului de ofertare pentru o singură oră de surplus.
    Tariful este FIX pe toată durata pool-ului; refuzul doar mută oferta la următorul nod.
    """

    def __init__(
        self,
        surplus_mw: float,
        ranking: List[Dict[str, Any]],
        tariff_eur_mwh: float = FIXED_TARIFF_EUR_MWH,
    ):
        self.surplus_mw = float(surplus_mw)
        self.ranking = ranking
        self.tariff_eur_mwh = float(tariff_eur_mwh)
        self.allocations: Dict[str, float] = {}  # dc_id -> MW (COMPUTING)
        self.rejected: List[str] = []            # dc_id (inactive)

    @property
    def remaining_mw(self) -> float:
        return max(0.0, self.surplus_mw - sum(self.allocations.values()))

    def current_offer(self) -> Optional[Dict[str, Any]]:
        return next_offer(
            self.ranking,
            self.remaining_mw,
            list(self.allocations) + self.rejected,
            self.tariff_eur_mwh,
        )

    def respond(self, dc_id: str, approved: bool) -> Optional[Dict[str, Any]]:
        """Înregistrează răspunsul nodului curent și returnează următoarea ofertă (sau None)."""
        offer = self.current_offer()
        if offer is None or offer["dc_id"] != dc_id:
            raise ValueError(f"Nu există o ofertă activă pentru {dc_id}.")
        if approved:
            self.allocations[dc_id] = offer["mw"]
        else:
            self.rejected.append(dc_id)
        return self.current_offer()

    def node_status(self, dc_id: str) -> str:
        if dc_id in self.allocations:
            return "COMPUTING"
        if dc_id in self.rejected:
            return "REJECTED"
        node = next((n for n in self.ranking if n["id"] == dc_id), None)
        if node is not None and not node["eligible"]:
            return "INELIGIBLE"
        offer = self.current_offer()
        if offer is not None and offer["dc_id"] == dc_id:
            return "OFFER_PENDING"
        return "IN_QUEUE"

    @property
    def status(self) -> str:
        if self.surplus_mw < MIN_OFFER_MW:
            return POOL_NO_SURPLUS
        if self.remaining_mw < MIN_OFFER_MW:
            return POOL_FULLY_ALLOCATED
        if self.current_offer() is not None:
            return POOL_OFFER_PENDING
        return POOL_ALL_DCS_EXHAUSTED
