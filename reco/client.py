"""Acceso a RawTree: query() para leer con SQL, insert() para escribir en team_NN."""
from __future__ import annotations

from typing import Any

import requests

from . import config

Record = dict[str, Any]

_http = requests.Session()
_http.headers["Authorization"] = f"Bearer {config.KEY}"


def query(sql: str, timeout: float | tuple[float, float] = 10) -> list[Record]:
    r = _http.post(f"{config.API}/v1/query", json={"sql": sql}, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"RawTree {r.status_code}: {r.text[:300]}")
    return r.json()["data"]


def query_df(sql: str, timeout: float = 600):
    """Para análisis / notebooks: devuelve un DataFrame."""
    import pandas as pd
    return pd.DataFrame(query(sql, timeout=timeout))


def insert(table: str, rows: list[Record], timeout: float | tuple[float, float] = 5) -> None:
    """Inserta en team_NN.<table> (submissions, scratch...)."""
    r = _http.post(f"{config.API}/v1/tables/{table}", params={"database": config.TEAM}, json=rows, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"RawTree insert {r.status_code}: {r.text[:300]}")
