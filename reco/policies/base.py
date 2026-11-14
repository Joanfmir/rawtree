"""Interfaz común de una política de recomendación."""
from __future__ import annotations

from ..client import Record
from ..state import State


class Policy:
    name = "base"
    refresh_every = 60.0   # segundos entre llamadas a refresh()

    def __init__(self, until: str | None = None) -> None:
        """until=None → producción (features_live). until='YYYY-MM-DD' → evaluación offline."""
        self.until = until

    def fit(self, state: State) -> None:
        """Trabajo pesado al arrancar (cargar features, modelos). Puede tardar."""

    def refresh(self, state: State) -> None:
        """Actualización periódica con datos en vivo (tendencias, feedback). Debe ser rápida."""

    def recommend(self, req: Record, state: State) -> list[str]:
        """Lista ordenada de IDs para una petición. Debe tardar milisegundos.
        No hace falta que sea válida: safety.finalize() la limpia y la completa."""
        raise NotImplementedError

    def load_features(self):
        """Features de self.until; en producción las construye si no existen."""
        from .. import features
        if not features.folder(self.until).exists():
            features.build(until=self.until)
        return features.load(until=self.until)
