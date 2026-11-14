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
