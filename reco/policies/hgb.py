"""Política hgb: una política base propone candidatos y un gradient boosting los reordena.

La base (por defecto `personal`) hace lo de siempre y expone signals(). Aquí se puntúan sus mejores
candidatos con el modelo de reco/ranker.py y salen los 10 con más probabilidad de clic/compra.

Seguridad: si no hay modelo entrenado, o si algo falla, se responde con el orden de la base. En el
peor caso hgb se comporta exactamente como su base.

En vivo recarga el modelo solo cuando cambia el fichero: se puede reentrenar (make train) sin parar serve.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import features, ranker
from .base import Policy

TOP_CANDIDATES = 150   # candidatos de la base (por su nota) que se pasan al modelo


class Hgb(Policy):
    name = "hgb"
    refresh_every = 30.0

    def __init__(self, until: str | None = None, base: str | None = None) -> None:
        super().__init__(until)
        from . import POLICIES
        self.ranker = ranker.load(until)
        self.base = POLICIES[base or (self.ranker.base if self.ranker else "personal")](until=until)
        self._own_base = True      # False si la base es una instancia compartida que ya gestiona serve
        self._base_refreshed = 0.0
        self._model_mtime = self._mtime(ranker.model_path(until))
        self._items_mtime = 0.0
        self.fz = None
        self._errors = 0

    def share(self, policies):
        """En un test A/B contra la propia base, reutiliza esa instancia: una sola copia de las
        features en memoria y un solo refresco."""
        for p in policies:
            if p is not self and type(p) is type(self.base) and p.until == self.until \
                    and getattr(p, "w", None) == getattr(self.base, "w", None):
                self.base, self._own_base = p, False

    def fit(self, state):
        if self._own_base:
            self.base.fit(state)
        self._load_items()
        if self.ranker is None:
            print(f"[hgb] no hay modelo en {ranker.model_path(self.until)}: respondo como {self.base.name}. "
                  f"Entrena con scripts/train_hgb.py")
        else:
            print(f"[hgb] modelo {self.ranker.meta} sobre {self.base.name}")

    def refresh(self, state):
        if self._own_base and time.time() - self._base_refreshed >= self.base.refresh_every:
            self._base_refreshed = time.time()
            self.base.refresh(state)
        m = self._mtime(ranker.model_path(self.until))
        if m != self._model_mtime:   # modelo reentrenado: se carga sin reiniciar
            self.ranker, self._model_mtime = ranker.load(self.until), m
            self._items_mtime = 0.0
            print(f"[hgb] modelo recargado: {self.ranker.meta if self.ranker else None}")
        if self._mtime(self._items_path()) != self._items_mtime:   # la base ha refrescado las features
            self._load_items()

    def recommend(self, req, state):
        names, cands = self.base.signals(req, state)
        score = len(names) - 1   # la última señal es la nota de la base
        pids = sorted(cands, key=lambda q: cands[q][score], reverse=True)[:TOP_CANDIDATES]
        rk, fz = self.ranker, self.fz
        if rk is None or fz is None or len(pids) <= 1:
            return pids[:10]
        try:
            prob = rk.score(fz.matrix(req, names, cands, pids), names)
            return [pids[i] for i in np.argsort(-prob, kind="stable")[:10]]
        except Exception as e:
            self._errors += 1
            if self._errors <= 5 or self._errors % 500 == 0:
                print(f"[hgb] fallo al puntuar ({self._errors}), respondo como {self.base.name}: {e}")
            return pids[:10]

    def _items_path(self):
        return features.folder(self.until) / "item_features.parquet"

    def _load_items(self):
        path = self._items_path()
        if not path.exists():
            return
        self._items_mtime = self._mtime(path)
        self.fz = ranker.Featurizer(pd.read_parquet(path), self.ranker.categories if self.ranker else None)

    @staticmethod
    def _mtime(path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0
