from __future__ import annotations

from .base import Policy


class Baseline(Policy):
    """Best sellers en stock: lo mismo que el fallback del organizador (score 0)."""
    name = "baseline"

    def recommend(self, req, state):
        return state.fallback()
