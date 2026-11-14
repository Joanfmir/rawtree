"""Garantiza que TODA respuesta es válida: 10 IDs distintos, activos y en stock.

Una respuesta inválida = fallback del organizador = perdemos la petición. Por eso cualquier
lista que salga de una política pasa por aquí y se rellena con best sellers si hace falta.
"""
from __future__ import annotations

from .state import State


def finalize(items: list[str], state: State) -> list[str]:
    out: list[str] = []
    for pid in list(items) + state.fallback() + state.best:
        if pid not in out and state.is_valid(pid):
            out.append(pid)
            if len(out) == 10:
                break
    return out
