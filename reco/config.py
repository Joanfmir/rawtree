"""Configuración: lee .env (TEAM_ID, RAWTREE_API_KEY) una sola vez."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"   # cachés locales (no se suben a git)
LOGS_DIR = ROOT / "logs"

_env = ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ.get("TEAM_ID", "")
KEY = os.environ.get("RAWTREE_API_KEY", "")


def check() -> None:
    if not TEAM or TEAM == "team_NN" or not KEY or KEY == "rt_...":
        raise SystemExit("Rellena TEAM_ID y RAWTREE_API_KEY en .env (te los dan en el stand de RawTree).")
