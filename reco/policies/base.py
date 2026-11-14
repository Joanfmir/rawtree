"""Interfaz común de una política de recomendación."""
from __future__ import annotations

from ..client import Record
from ..state import State


class Policy:
    name = "base"
    refresh_every = 60.0   # segundos entre llamadas a refresh()

    def fit(self, state: State) -> None:
        """Trabajo pesado al arrancar (cargar modelos, cachés). Puede tardar."""

    def refresh(self, state: State) -> None:
        """Actualización periódica con datos en vivo (tendencias, feedback). Debe ser rápida."""

    def recommend(self, req: Record, state: State) -> list[str]:
        """Lista ordenada de IDs para una petición. Debe tardar milisegundos.
        No hace falta que sea válida: safety.finalize() la limpia y la completa."""
        raise NotImplementedError
