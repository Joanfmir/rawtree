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

    def signals(self, req: Record, state: State):
        """Opcional. (nombres, {producto: [valor por nombre]}): señales de cada candidato, la última
        es la nota de la política. Es lo que necesita una política para servir de base al re-ranker hgb."""
        raise NotImplementedError(f"{self.name} no expone signals(): no puede ser la base de hgb")

    def share(self, policies: list["Policy"]) -> None:
        """serve la llama antes de fit() con todas las políticas del proceso (test A/B), por si
        alguna puede reutilizar a otra en vez de cargar lo mismo dos veces."""

    def load_features(self):
        """Features de self.until; en producción las construye si no existen."""
        from .. import features
        if not features.folder(self.until).exists():
            features.build(until=self.until)
        return features.load(until=self.until)
