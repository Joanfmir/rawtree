"""Política "joan": bird_v2 + reordenador de compra orgánica para conocidos, mezcla para usuarios nuevos.

Diagnóstico (test orgánico, peticiones reales del 14/11, ver scripts/offline_vs_live.py):
  - bird_v2 es lo mejor con usuarios conocidos (ndcg 0,066) pero de lo peor con nuevos (0,032 ≈ baseline);
    gulias (0,051) y v3-bird (0,047) son mucho mejores con nuevos. El leaderboard dice lo mismo en la
    conversión de nuevos (2,04 y 2,02 frente a 1,91).
  - Con conocidos, lo que compran está en el top 10 de bird_v2 el 12 % de las veces, pero en el top 100
    el 38 %: hay margen para reordenar sus candidatos.

Por eso:
  joan_rerank  bird_v2 propone sus K mejores (catálogo entero, sin lo ya comprado) y un CatBoost de ranking
               entrenado con el objetivo que mejor predice el leaderboard (la próxima compra orgánica) los
               reordena con las 56 features de Bird + la puntuación y el puesto de bird_v2.
               Entrenamiento: scripts/joan_ranker.py (datos anteriores al 07/11: los dos tests quedan limpios).
  joan         enrutador: conocidos → joan_rerank; nuevos → RRF(gulias, v3-bird). En el leaderboard: v3-joan.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import Policy
from .bird import BirdBlendRank, BirdV2

EXTRA = ("bird_score", "bird_rank", "bird_gap")   # features añadidas a las 56 de Bird


def candidates(bird: BirdV2, req: dict, state, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Los k mejores de bird_v2 para `req` (mismas exclusiones que BirdV2.recommend) con sus features:
    (índices en live.ids, numéricas [k, 56+3] float32, categóricas [k, 8]). Igual al entrenar y al servir."""
    live = bird.live
    mask = bird._valid_mask(state, live).copy()
    mask[live.bought(live.user(req))] = False
    for p in bird._sold.get(req.get("user_id"), ()):
        i = live.idx.get(p)
        if i is not None:
            mask[i] = False
    cand = np.flatnonzero(mask)
    if len(cand) == 0:
        return cand, np.zeros((0, len(bird.model.schema.numeric) + len(EXTRA)), np.float32), np.zeros((0, 0), object)
    num, cat = live.matrices(req, bird.model.schema, cand)
    score = bird.model.predict_matrices(num, cat)
    order = np.argsort(-score, kind="stable")[:k]
    s = score[order]
    extra = np.column_stack([s, np.arange(len(order), dtype=float), s[0] - s]).astype(np.float32)
    return cand[order], np.hstack([num[order], extra]), cat[order]


class RankerModel:
    """CatBoost de ranking guardado en data/joan/<nombre>/ (model.cbm + meta.json)."""

    def __init__(self, folder: Path) -> None:
        from catboost import CatBoost
        self.meta = json.loads((folder / "meta.json").read_text())
        self.model = CatBoost()
        self.model.load_model(str(folder / "model.cbm"))
        self.k = int(self.meta["k"])

    def predict(self, num: np.ndarray, cat: np.ndarray) -> np.ndarray:
        from catboost import FeaturesData, Pool
        data = FeaturesData(num_feature_data=np.ascontiguousarray(num, dtype=np.float32), cat_feature_data=cat,
                            num_feature_names=self.meta["num"], cat_feature_names=self.meta["cat"])
        return np.asarray(self.model.predict(Pool(data), prediction_type="RawFormulaVal", thread_count=1))


class JoanRerank(BirdV2):
    """bird_v2 propone K candidatos y el reordenador orgánico decide el orden (ver docstring del módulo)."""
    name = "joan_rerank"
    ranker_name = "organic_ranker"

    def fit(self, state):
        from ..config import DATA_DIR
        super().fit(state)
        self.ranker = RankerModel(DATA_DIR / "joan" / self.ranker_name)
        print(f"[joan] reordenador {self.ranker_name} (K={self.ranker.k}, {self.ranker.model.tree_count_} árboles)")

    # Peso del reordenador al mezclar puestos: clave = w·puesto_reordenador + (1−w)·puesto_bird.
    # Test orgánico desde las 13:00, conocidos, base v2_live (la de producción): w=0,5 → +0,0027 [+0,0003, +0,0051]
    # sobre bird solo (0,0696); el reordenador puro (w=1) no es significativo. Mezclar estabiliza.
    w = 0.5

    def recommend(self, req, state):
        idx, num, cat = candidates(self, req, state, self.ranker.k)
        if len(idx) == 0:
            return []
        rr = np.empty(len(idx))
        rr[np.argsort(-self.ranker.predict(num, cat), kind="stable")] = np.arange(len(idx))
        key = self.w * rr + (1 - self.w) * np.arange(len(idx))   # a igualdad, manda el orden de bird
        return self.live.ids[idx[np.argsort(key, kind="stable")[:self.top]]].tolist()


class Joan(Policy):
    """Enrutador: usuarios conocidos → joan_rerank; nuevos (o sin historia) → RRF(gulias, v3-bird).
    Un solo estado de Bird compartido: el bird de v3-bird usa el de joan_rerank y no lo recarga por su cuenta
    (dos recargas a la vez corrompían data/bird/state_live)."""
    name = "joan"
    refresh_every = 60.0

    def fit(self, state):
        from .ranker import Ranker
        self.known = JoanRerank(until=self.until)
        self.known.fit(state)
        self.gulias = Ranker(until=self.until)
        self.gulias.fit(state)
        self.blend = BirdBlendRank(until=self.until)
        self.blend.fit(state)
        self.blend.bird.STATE_EVERY = float("inf")   # nunca recarga el estado: usa el de self.known
        self.blend.bird.live = self.known.live

    def refresh(self, state):
        for p in (self.known, self.gulias, self.blend):
            try:
                p.refresh(state)
            except Exception as e:
                print(f"[joan] refresh de {p.name}: {e}")

    def is_new(self, req) -> bool:
        return self.known.live.user(req) is None

    def recommend(self, req, state):
        if not self.is_new(req):
            return self.known.recommend(req, state)
        from ..bird.blend import rrf
        self.blend.bird.live = self.known.live   # por si joan_rerank ha recargado el estado
        return rrf(self.gulias.recommend(req, state), self.blend.recommend(req, state))[:30]
