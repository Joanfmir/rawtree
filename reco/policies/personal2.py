"""personal2: `personal` corregida con lo que dice el feedback en vivo (scripts/analyze_feedback.py).

El histórico premia la recompra, pero el simulador en vivo no: en Leattner.v1 los productos ya
comprados ocupaban el 36 % de los slots 1–2 y dieron 2 compras de 21. Cambios respecto a `personal`:
  rebuy        peso menor: sigue por encima de la popularidad, pero por debajo de las señales de sesión
  rebuy_ramp   la recencia va al revés: lo comprado hace poco casi no cuenta y sube hasta 1 a los
               `rebuy_ramp` días (en vivo se recompra lo de hace > 30 días, no lo de las últimas 2 semanas)
  sub          señal nueva: más vendidos de la subcategoría de la última vista (se compran 2,4× más)

No toca SIGNALS: el re-ranker hgb entrenado sobre `personal` sigue viendo las mismas columnas.
"""
from __future__ import annotations

from .personal import _BRAND, _SCORE, Personal

WEIGHTS = {"rebuy": 0.3, "rebuy_ramp": 30.0, "sub": 0.3}


class Personal2(Personal):
    name = "personal2"

    def __init__(self, until: str | None = None, weights: dict | None = None) -> None:
        super().__init__(until, {**WEIGHTS, **(weights or {})})
        self._shared = None   # instancia de `personal` del mismo proceso (test A/B): reutiliza su índice

    # Las features son las mismas que las de personal: en un A/B contra ella, una sola copia y un solo refresco
    def share(self, policies):
        for p in policies:
            if type(p) is Personal and p.until == self.until:
                self._shared = p

    @property
    def ix(self):
        return self._shared.ix if self._shared is not None else self._ix

    @ix.setter
    def ix(self, value):
        self._ix = value

    def fit(self, state):
        if self._shared is None:
            super().fit(state)

    def refresh(self, state):
        if self._shared is None:
            super().refresh(state)

    def recommend(self, req, state):
        _, c = self.signals(req, state)
        score = {q: v[_SCORE] for q, v in c.items()}
        views = req.get("recent_views") or []
        w_sub, ix = self.w["sub"], self.ix
        if views and w_sub:
            sub = ix.subcat_of.get(views[-1]["product_id"])
            for q, n in ix.subcat_top.get(sub, ()):
                brand = c[q][_BRAND] if q in c else 0.0
                score[q] = score.get(q, 0.0) + w_sub * n * (1 + self.w["brand"] * brand)
        return sorted(score, key=score.get, reverse=True)[:10]
