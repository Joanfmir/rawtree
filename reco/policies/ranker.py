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

import subprocess
import sys
import threading
import time

import lightgbm as lgb
import numpy as np

from .. import features
from ..client import query
from ..config import DATA_DIR, ROOT, TEAM
from ..ranker import FEATURES, Featurizer
from .base import Policy

SAME_CAT = FEATURES.index("same_cat_last")
FEATURES_EVERY = 1200.0


class Ranker(Policy):
    name = "gulias"
    refresh_every = 120.0
    cat_boost = 1.0
    model_tag = ""     # sufijo del fichero del modelo (experimentos)

    def fit(self, state):
        self.fz = Featurizer(self.load_features())
        path = DATA_DIR / ((f"ranker_until_{self.until}" if self.until else "ranker_live") + f"{self.model_tag}.txt")
        self.model = lgb.Booster(model_file=str(path))
        self._live_bought: frozenset[tuple[str, str]] = frozenset()
        self._rebuilt_at = time.monotonic()
        self._thread: threading.Thread | None = None

    def refresh(self, state):
        if self.until:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._refresh_background, daemon=True)
        self._thread.start()

    def _refresh_background(self) -> None:
        """Ni la red ni la reconstrucción bloquean el bucle de respuestas."""
        try:
            self._add_live_purchases()
        except Exception as e:
            print(f"[ranker] compras en vivo no actualizadas: {e}")
        if time.monotonic() - self._rebuilt_at >= FEATURES_EVERY:
            self._rebuilt_at = time.monotonic()
            self._rebuild()

    def _add_live_purchases(self) -> None:
        """Lo que nuestros carruseles han vendido no se vuelve a ofrecer a ese usuario."""
        rows = query(f"""SELECT r.user_id::String AS u, f.purchased::String AS p
                         FROM {TEAM}.feedback AS f INNER JOIN shop.requests AS r
                           ON r.request_id::String = f.request_id::String
                         WHERE f.purchased::String != '' AND f.request_id::String != '__init__'
                           AND f.ts::DateTime64(3) >= now64(3) - INTERVAL 2 DAY
                         ORDER BY f.ts::DateTime64(3) DESC LIMIT 500000""", timeout=30)
        # Publicación inmutable: recommend nunca observa un set en plena modificación.
        self._live_bought = frozenset((r["u"], r["p"]) for r in rows)

    def _rebuild(self) -> None:
        try:
            t = time.time()
            # En un proceso aparte: descargar y decodificar ~1 M de filas en un hilo bloquea el GIL
            # varios segundos y retrasa las respuestas (medido: p95 de 0,9 s a 6,7 s).
            subprocess.run([sys.executable, str(ROOT / "scripts" / "build_features.py")], cwd=ROOT,
                           check=True, capture_output=True, timeout=900)
            fz = Featurizer(features.load(until=None))
            self.fz = fz   # cambio atómico: recommend() usa el nuevo a partir de aquí
            print(f"[ranker] features recalculadas en {time.time() - t:.0f}s")
        except Exception as e:
            print(f"[ranker] rebuild falló: {e}")

    def recommend(self, req, state):
        fz = self.fz
        user = None if req.get("is_new_user") else req.get("user_id")
        cand = fz.candidates(req, user)
        bought = self._live_bought
        cand = cand[np.array([state.is_valid(p) and (user, p) not in bought
                              for p in fz.ids[cand]], dtype=bool)]
        if len(cand) == 0:
            return []
        X = fz.matrix(req, user, cand)
        score = self.model.predict(X, num_threads=1) + self.cat_boost * (X[:, SAME_CAT] == 1)
        return list(fz.ids[cand[np.argsort(-score)[:10]]])


class RankerFeedback(Ranker):
    """Ranker de feedback real; sin refuerzo manual, ya aprendido por el modelo.

    Conserva la exclusión de recompra y el refresco de candidatos de gulias.
    El entrenamiento se ejecuta fuera del servidor; el modelo se carga al arrancar.
    """
    name = "gulias_fb"
    cat_boost = 0.0

    def fit(self, state):
        super().fit(state)
        path = DATA_DIR / "ranker_live_fb.txt"
        if not self.until and path.exists():
            self.cat_boost = 0.0
            self.model = lgb.Booster(model_file=str(path))
        else:
            self.cat_boost = Ranker.cat_boost
