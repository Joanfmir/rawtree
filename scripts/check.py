"""Comprueba credenciales y cuántos registros hay en cada tabla.   python scripts/check.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reco import config  # noqa: E402
from reco.client import query  # noqa: E402

config.check()
for table in ["shop.catalog", "shop.history_sessions", "shop.history_events", "shop.requests",
              "shop.events", "shop.catalog_updates", f"{config.TEAM}.submissions", f"{config.TEAM}.feedback",
              f"{config.TEAM}.scratch"]:
    try:
        print(f"{table:32} {query(f'SELECT count() AS n FROM {table}')[0]['n']:>12}")
    except Exception as e:
        print(f"{table:32} ERROR {str(e)[:80]}")
