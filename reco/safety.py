"""Garantiza que TODA respuesta es válida: 10 IDs distintos, activos y en stock.

Una respuesta inválida = fallback del organizador = perdemos la petición. Por eso cualquier
lista que salga de una política pasa por aquí y se rellena con best sellers si hace falta.
"""
from __future__ import annotations

from .state import State


def finalize(items: list[str], state: State) -> list[str]:
    state = state.snapshot()
    out: list[str] = []
    for pid in list(items or []) + state.fallback():
        if state.is_valid(pid) and pid not in out:
            out.append(pid)
            if len(out) == 10:
                break
    if len(out) != 10:
        raise ValueError("No hay diez productos activos y en stock en el estado disponible")
    return out
