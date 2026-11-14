"""Política "gulias" (ranker LightGBM): candidatos (vistas, vecinos, subcategorías, usuario, contexto, populares) puntuados
por un LightGBM entrenado con las impresiones del histórico (scripts/train_ranker.py). Ver reco/ranker.py.

Encima del modelo, dos reglas que el log no puede enseñar (el sistema viejo solo enseñaba productos de la
categoría de la última vista) y que el feedback en vivo confirma:
  cat_boost   + a los productos de la misma categoría que la última vista (en vivo se compran ~4× más)
Sin recompra: los productos que el usuario ya compró (histórico + compras en vivo de nuestro feedback)
nunca se recomiendan.

Aprendizaje del stream (solo en producción, en un hilo para no bloquear el bucle):
  cada FEATURES_EVERY s se recalculan las features con shop.events (tendencias de la última hora/día,
  productos nuevos, cambios de precio, nuevos usuarios) y se sustituye el Featurizer.
"""
from __future__ import annotations

import threading
import time

import lightgbm as lgb
import numpy as np

from .. import features
from ..client import query
from ..config import DATA_DIR, TEAM
from ..ranker import FEATURES, Featurizer
from .base import Policy

SAME_CAT = FEATURES.index("same_cat_last")
FEATURES_EVERY = 1200.0


class Ranker(Policy):
    name = "gulias"
    refresh_every = 120.0
    cat_boost = 1.0

    def fit(self, state):
        self.fz = Featurizer(self.load_features())
        path = DATA_DIR / (f"ranker_until_{self.until}.txt" if self.until else "ranker_live.txt")
        self.model = lgb.Booster(model_file=str(path))
        self._live_bought: set[tuple[str, str]] = set()
        self._rebuilt_at = time.time()
        self._thread: threading.Thread | None = None

    def refresh(self, state):
        if self.until:
            return
        self._add_live_purchases()
        busy = self._thread is not None and self._thread.is_alive()
        if not busy and time.time() - self._rebuilt_at >= FEATURES_EVERY:
            self._rebuilt_at = time.time()
            self._thread = threading.Thread(target=self._rebuild, daemon=True)
            self._thread.start()

    def _add_live_purchases(self) -> None:
        """Lo que nuestros carruseles han vendido no se vuelve a ofrecer a ese usuario."""
        rows = query(f"""SELECT r.user_id::String AS u, f.purchased::String AS p
                         FROM {TEAM}.feedback AS f INNER JOIN shop.requests AS r
                           ON r.request_id::String = f.request_id::String
                         WHERE f.purchased::String != ''""", timeout=30)
        self._live_bought |= {(r["u"], r["p"]) for r in rows}
        for u, p in self._live_bought:
            if p in self.fz.idx:
                self.fz.bought[u].add(self.fz.idx[p])

    def _rebuild(self) -> None:
        try:
            t = time.time()
            fz = Featurizer(features.build(until=None, verbose=False))
            for u, p in self._live_bought:
                if p in fz.idx:
                    fz.bought[u].add(fz.idx[p])
            self.fz = fz   # cambio atómico: recommend() usa el nuevo a partir de aquí
            print(f"[ranker] features recalculadas en {time.time() - t:.0f}s")
        except Exception as e:
            print(f"[ranker] rebuild falló: {e}")

    def recommend(self, req, state):
        fz = self.fz
        user = None if req.get("is_new_user") else req.get("user_id")
        cand = fz.candidates(req, user)
        cand = cand[np.array([state.is_valid(p) for p in fz.ids[cand]], dtype=bool)]
        if len(cand) == 0:
            return []
        X = fz.matrix(req, user, cand)
        score = self.model.predict(X, num_threads=1) + self.cat_boost * (X[:, SAME_CAT] == 1)
        return list(fz.ids[cand[np.argsort(-score)[:10]]])
