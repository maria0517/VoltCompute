"""
Autentificare simplă pentru demo: utilizatorii stau în `data/users.json`, cu parola salvată ca
hash PBKDF2-SHA256 cu salt (modulul standard `hashlib`, fără dependențe externe).

Roluri:
  * "ppc"        -> dispeceratul PPC (grafice, tabel, control temporal)
  * "datacenter" -> un Data Center (vede doar ofertele și contractele lui); are și `dc_id`

NOTĂ: potrivit pentru demo / hackathon, nu pentru producție (fără rate limiting, fără HTTPS propriu).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from pathlib import Path
from typing import Any, Dict, Optional

USERS_PATH = Path(__file__).resolve().parent.parent / "data" / "users.json"
_ITERATIONS = 120_000


def hash_password(password: str, salt: Optional[str] = None) -> Dict[str, str]:
    """Întoarce {"salt", "hash"} pentru o parolă (salt nou dacă nu e dat)."""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), _ITERATIONS)
    return {"salt": salt, "hash": digest.hex()}


def load_users(path: Path = USERS_PATH) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return {u["username"]: u for u in json.load(f)["users"]}


def authenticate(username: str, password: str, path: Path = USERS_PATH) -> Optional[Dict[str, Any]]:
    """Verifică datele de logare. Întoarce profilul public al utilizatorului sau None."""
    user = load_users(path).get((username or "").strip().lower())
    if user is None:
        return None
    expected = hash_password(password or "", user["salt"])["hash"]
    if not hmac.compare_digest(expected, user["hash"]):
        return None
    return {k: v for k, v in user.items() if k not in ("salt", "hash")}
