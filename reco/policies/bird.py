"""Política "bird_v1" (v1-Bird): CatBoost predice P(compra) de cada producto activo y se devuelven los 10 mejores.

Entrenamiento aparte (scripts/bird.py train); aquí solo se carga el artefacto de data/bird/<versión>/.
Todos los candidatos se puntúan con la misma posición de referencia (schema.reference_position).

Dos caminos, elegidos por el esquema del modelo:
  estático      ninguna feature depende de la petición (caso V1): el ranking del catálogo se calcula en
                fit()/refresh() y recommend() solo descarta lo que no esté activo o en stock.
  por petición  hay features de usuario/contexto: se construyen y se puntúan los candidatos en cada petición.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import features
from ..bird.builder import FeatureState, build_features
from ..bird.model import BirdModel, model_dir
from .base import Policy


class Bird(Policy):
    name = "bird_v1"
    version = "v1"
    refresh_every = 60.0
    folder: Path | None = None        # carpeta del modelo; por defecto data/bird/<versión>[_until_<fecha>]
    force_per_request = False         # para medir la latencia del camino general con un esquema estático

    def fit(self, state):
        folder = self.folder or model_dir(self.version, self.until)
        if self.folder is None and self.until and not folder.exists():
            folder = model_dir(self.version)
            print(f"[bird] sin modelo para el corte {self.until}; se usa {folder} "
                  "(si se entrenó con datos posteriores al corte, la evaluación offline es optimista)")
        self.model = BirdModel.load(folder)
        self.static = self.model.schema.is_static and not self.force_per_request
        self._publish(state)

    def refresh(self, state):
        if not self.until:   # en vivo cambian el catálogo y los precios
            self._publish(state)

    def _feature_state(self, state) -> FeatureState:
        catalog = state.snapshot().catalog
        if catalog and "category" in next(iter(catalog.values())):
            return FeatureState.from_catalog(catalog)
        # evaluación offline: el estado solo sabe qué productos existen
        return FeatureState.from_items(pd.read_parquet(features.folder(self.until) / "item_features.parquet"))

    def _publish(self, state) -> None:
        fs = self._feature_state(state)
        ids = fs.items.index.to_numpy()
        ranked: list[str] = []
        if self.static:
            score = self.model.predict(build_features({}, ids, fs, self.model.schema))
            ranked = ids[np.argsort(-score, kind="stable")].tolist()
        self._view = (fs, ids, ranked)   # una sola asignación: recommend() nunca ve un estado a medias

    def recommend(self, req, state):
        fs, ids, ranked = self._view
        if self.static:
            out = []
            for pid in ranked:
                if state.is_valid(pid):
                    out.append(pid)
                    if len(out) == 10:
                        break
            return out
        cand = [p for p in ids if state.is_valid(p)]
        if not cand:
            return []
        score = self.model.predict(build_features(req, cand, fs, self.model.schema))
        return [cand[i] for i in np.argsort(-score, kind="stable")[:10]]


class BirdV2(Policy):
    """Política "bird_v2": CatBoost con las 56 features de v2_norebuy (producto, sesión, usuario, afinidad).

    Puntúa todo el catálogo en cada petición con reco/bird/live.py (el mismo significado de las features que en
    el entrenamiento). Nunca ofrece lo que el usuario ya compró: en vivo convierte ~9× menos.
    El estado de usuarios se vuelve a descargar cada STATE_EVERY s en un proceso aparte.
    """
    name = "bird_v2"
    model_name = "v2_fresh"
    final_name = "v2_live"   # modelo final (todos los datos); solo se usa en vivo
    folder = None            # carpeta del modelo (tests)
    refresh_every = 120.0
    STATE_EVERY = 1800.0
    top = 30   # finalize() se queda con los 10 primeros válidos; el margen evita rellenar con best sellers

    def fit(self, state):
        import threading
        import time

        from ..bird.live import LiveState
        from ..config import DATA_DIR
        folder = self.folder or DATA_DIR / "bird" / self.model_name
        final = DATA_DIR / "bird" / self.final_name
        if self.folder is None and not self.until and (final / "model.cbm").exists():
            folder = final   # en vivo: el modelo entrenado con todos los datos, si existe
        self.model = BirdModel.load(folder)
        print(f"[bird_v2] modelo {folder.name} ({self.model.info.get('trees', '?')} árboles)")
        self.live = LiveState.load(self.until)
        self._sold: dict[str, frozenset[str]] = {}
        self._valid: tuple[tuple[int, int], np.ndarray] | None = None
        self._state_at = time.monotonic()
        self._thread: threading.Thread | None = None

    def refresh(self, state):
        import threading
        if self.until:
            return
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._refresh_background, daemon=True)
            self._thread.start()

    def _refresh_background(self) -> None:
        import subprocess
        import sys
        import time

        from ..bird.live import LiveState
        from ..client import query
        from ..config import ROOT, TEAM
        try:   # lo que nuestro carrusel ya ha vendido en vivo
            rows = query(f"""SELECT DISTINCT r.user_id::String AS u, f.purchased::String AS p
                             FROM {TEAM}.feedback AS f INNER JOIN shop.requests AS r
                               ON r.request_id::String = f.request_id::String
                             WHERE f.purchased::String != '' AND f.request_id::String != '__init__'""", timeout=30)
            sold: dict[str, set[str]] = {}
            for r in rows:
                sold.setdefault(r["u"], set()).add(r["p"])
            self._sold = {u: frozenset(p) for u, p in sold.items()}
        except Exception as e:
            print(f"[bird_v2] ventas en vivo no actualizadas: {e}")
        if time.monotonic() - self._state_at >= self.STATE_EVERY:
            self._state_at = time.monotonic()
            try:
                t = time.time()
                subprocess.run([sys.executable, str(ROOT / "scripts" / "bird.py"), "state"], cwd=ROOT, check=True,
                               capture_output=True, timeout=1500)
                self.live = LiveState.load(None)   # cambio atómico
                print(f"[bird_v2] estado de usuarios recargado en {time.time() - t:.0f}s")
            except Exception as e:
                print(f"[bird_v2] recarga del estado falló, sigo con el anterior: {e}")

    def _valid_mask(self, state, live) -> np.ndarray:
        snap = state.snapshot()
        key = (id(snap.catalog), id(snap.unavailable), id(live))
        if self._valid is None or self._valid[0] != key:
            self._valid = (key, np.array([snap.is_valid(p) for p in live.ids], dtype=bool))
        return self._valid[1]

    def recommend(self, req, state):
        live = self.live
        mask = self._valid_mask(state, live).copy()
        mask[live.bought(live.user(req))] = False
        for p in self._sold.get(req.get("user_id"), ()):
            i = live.idx.get(p)
            if i is not None:
                mask[i] = False
        cand = np.flatnonzero(mask)
        if len(cand) == 0:
            return []
        score = self.model.predict_matrices(*live.matrices(req, self.model.schema, cand))
        return live.ids[cand[np.argsort(-score, kind="stable")[:self.top]]].tolist()

    def score(self, req, state, product_ids: list[str]) -> np.ndarray:
        """Puntuación de Bird para una lista corta de productos (-inf: no se puede recomendar a este usuario)."""
        live = self.live
        mask = self._valid_mask(state, live).copy()
        mask[live.bought(live.user(req))] = False
        sold = self._sold.get(req.get("user_id"), ())
        rows = np.array([live.idx.get(p, -1) for p in product_ids], dtype=int)
        ok = np.array([i >= 0 and mask[i] and p not in sold for i, p in zip(rows, product_ids)], dtype=bool)
        out = np.full(len(product_ids), -np.inf)
        if ok.any():
            out[ok] = self.model.predict_matrices(*live.matrices(req, self.model.schema, rows[ok]))
        return out


class BirdRank(BirdV2):
    """bird_v2 con el modelo entrenado para ordenar (YetiRank). Solo, sobre todo el catálogo, es peor que
    bird_v2 (aprende el orden dentro de un carrusel, no el nivel absoluto); su sitio es dentro de la mezcla."""
    name = "bird_v2_rank"
    model_name = "v2_rank"
    final_name = "v2_rank_live"


class BirdBlend(Policy):
    """Mezcla: personal_v6 (personal_v5 sin recompra) propone sus N mejores candidatos y Bird decide el orden.

    Cada uno cubre el punto débil del otro: personal acierta QUÉ productos merecen estar (también con
    usuarios nuevos) y Bird los ORDENA mejor. Offline (corte 07/11): ips_ndcg 0,25 frente a 0,21 de Bird
    solo y 0,13 de personal_v5; con usuario oculto 0,15 frente a 0,12 y 0,11. Ver scripts/bird_blend.py.
    """
    name = "bird_blend"
    bird_class = BirdV2
    personal_class = None   # por defecto PersonalV6: no gasta candidatos en lo que el usuario ya compró
    n = 50
    refresh_every = 120.0

    def fit(self, state):
        from .personal import PersonalV6
        self.personal = (self.personal_class or PersonalV6)(until=self.until, weights={"top": self.n})
        self.personal.fit(state)
        self.bird = self.bird_class(until=self.until)
        self.bird.fit(state)

    def refresh(self, state):
        for policy in (self.personal, self.bird):   # cada una lleva su propio ritmo y sus hilos
            try:
                policy.refresh(state)
            except Exception as e:
                print(f"[{self.name}] refresh de {policy.name}: {e}")

    def recommend(self, req, state):
        cand = self.personal.recommend(req, state)
        if len(cand) < 10:   # sin candidatos suficientes: lista propia de Bird
            return cand + [p for p in self.bird.recommend(req, state) if p not in cand]
        score = self.bird.score(req, state, cand)
        order = np.argsort(-score, kind="stable")   # a igualdad (o sin puntuación), el orden de personal
        return [cand[i] for i in order if np.isfinite(score[i])]


class BirdBlendRank(BirdBlend):
    """La misma mezcla con el modelo de ranking: empata en ips_ndcg y va algo mejor con usuarios nuevos."""
    name = "bird_blend_rank"
    bird_class = BirdRank
