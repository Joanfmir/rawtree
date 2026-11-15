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
    state_script = "bird.py"   # script cuyo comando `state` vuelve a descargar el estado de usuarios

    def _load_state(self, until):
        from ..bird.live import LiveState
        return LiveState.load(until)

    def fit(self, state):
        import threading
        import time

        from ..config import DATA_DIR
        folder = self.folder or DATA_DIR / "bird" / self.model_name
        final = DATA_DIR / "bird" / self.final_name
        if self.folder is None and not self.until and (final / "model.cbm").exists():
            folder = final   # en vivo: el modelo entrenado con todos los datos, si existe
        self.model = BirdModel.load(folder)
        print(f"[{self.name}] modelo {folder.name} ({self.model.info.get('trees', '?')} árboles)")
        self.live = self._load_state(self.until)
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
        self._refresh_sold()
        self._reload_state()

    def _refresh_sold(self) -> None:
        from ..client import query
        from ..config import TEAM
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
            print(f"[{self.name}] ventas en vivo no actualizadas: {e}")

    def _reload_state(self) -> None:
        import subprocess
        import sys
        import time

        from ..config import ROOT
        if time.monotonic() - self._state_at >= self.STATE_EVERY:
            self._state_at = time.monotonic()
            try:
                t = time.time()
                subprocess.run([sys.executable, str(ROOT / "scripts" / self.state_script), "state"], cwd=ROOT,
                               check=True, capture_output=True, timeout=1500)
                self.live = self._load_state(None)   # cambio atómico
                print(f"[{self.name}] estado de usuarios recargado en {time.time() - t:.0f}s")
            except Exception as e:
                print(f"[{self.name}] recarga del estado falló, sigo con el anterior: {e}")

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


class BirdV5(BirdV2):
    """Política "bird_v5" (v5-bird): el modelo de v2-bird (CatBoost, P(compra), sin recompra) con el dataset v5.

    Qué cambia respecto a v2-bird (docs/DATASET_V5.md): features invariantes al volumen de tráfico, promoción
    (precio actual frente al precio base), tendencia de la última hora, respuesta de cada producto en nuestro
    carrusel, qué compra cada dispositivo / región, más detalle de la sesión y de la afinidad del usuario; y
    se entrena también con nuestro feedback en vivo. Las features se calculan con las MISMAS funciones al
    entrenar y al servir (reco/bird/feat5.py).
    Cada refresco (2 min) actualiza catálogo, promociones, tendencias y feedback con 3 consultas de segundos;
    la historia de los usuarios se vuelve a descargar cada STATE_EVERY s en un proceso aparte.
    """
    name = "bird_v5"
    model_name = "v5_fresh"   # solo histórico anterior al 07/11: el que se evalúa offline
    final_name = "v5_live"    # histórico + feedback en vivo: el que se sirve
    state_script = "bird_v5.py"
    use_delta = False         # v6-bird (BirdV5Delta) lo activa
    _hub = None               # estado en memoria compartido por todos los brazos v5 del proceso

    def _load_state(self, until):
        from ..bird.live5 import LiveState5
        if until:                       # evaluación offline: estado propio, a fecha del corte
            return LiveState5.load(until)
        if BirdV5._hub is None:
            BirdV5._hub = _V5Hub(LiveState5.load(None), self.STATE_EVERY, self.state_script)
        return BirdV5._hub.pick(self.use_delta)

    def _refresh_background(self) -> None:
        self._refresh_sold()
        hub = BirdV5._hub
        hub.refresh(self.name)
        self.live = hub.pick(self.use_delta)   # cambio atómico: recommend() usa el estado que cogió al empezar


class BirdV5Delta(BirdV5):
    """Política "bird_v5_delta" (v6-bird): v5-bird con la historia de los usuarios al minuto.

    Mismo modelo y mismas features. Lo único que cambia: además de la descarga completa de cada 30 min,
    en cada refresco (2 min) se suma lo que cada usuario ha hecho desde entonces (reco/bird/live5.Delta), y
    lo que haya comprado en ese rato deja de ofrecérsele. Sirve sobre todo cuando entran compradores nuevos
    de golpe: hasta la siguiente descarga completa solo existen en el delta.
    En un A/B con v5-bird los dos brazos comparten la misma descarga: la diferencia es solo el delta.
    """
    name = "bird_v5_delta"
    use_delta = True


class BirdV5DeltaRetrained(BirdV5Delta):
    """Política "bird_v5_delta_r2" (v7-bird): v6-bird con el modelo reentrenado (data/bird/v5_live2).

    Mismas 81 features y mismo estado (base + delta); solo cambia el fichero del modelo, que se entrenó
    con el histórico y TODO el feedback en vivo hasta la madrugada del 15/11, incluido el de v5-bird y
    v6-bird. Si falta v5_live2 arranca con v5_fresh: comprobar la línea "[...] modelo ..." al arrancar.
    """
    name = "bird_v5_delta_r2"
    final_name = "v5_live2"


class BirdV5Organic(BirdV5Delta):
    """Política "bird_v5_org" (v8-bird): v7-bird con un modelo que además aprende de las compras orgánicas.

    Mismas 81 features y mismo estado. El modelo (data/bird/v5_org) se entrena con histórico + nuestro
    feedback + filas "org": para cada petición del stream, lo que el usuario compró por su cuenta en los
    30 min siguientes (positivo) frente a ~20 productos al azar (negativos). Esas filas no dependen de lo
    que enseñó nuestro carrusel: dicen qué quería el usuario también cuando no pulsó nada.
    Ver `python scripts/bird_v5.py compare-org`. Si falta v5_org arranca con v5_fresh: mirar el log.
    """
    name = "bird_v5_org"
    final_name = "v5_org"


class _V5Hub:
    """Estado de servicio de v5 compartido entre brazos: una sola descarga, una sola copia en memoria.

    plain   la descarga completa con catálogo, promociones y popularidad al día (lo que sirve v5-bird)
    delta   lo mismo + lo que han hecho los usuarios desde la descarga (lo que sirve v6-bird)
    Cada brazo llama a refresh() a su ritmo; si otro lo acaba de hacer (o lo está haciendo), no se repite.
    """

    def __init__(self, base, state_every: float, script: str) -> None:
        import threading
        import time
        self.lock = threading.Lock()
        self.base = self.plain = self.delta = base
        self.state_every, self.script = state_every, script
        self.state_at, self.items_at = time.monotonic(), 0.0

    def pick(self, use_delta: bool):
        return self.delta if use_delta else self.plain

    def refresh(self, who: str) -> None:
        import subprocess
        import sys
        import time

        from ..bird.live5 import LiveState5
        from ..config import ROOT
        if not self.lock.acquire(blocking=False):
            return
        try:
            if time.monotonic() - self.items_at < 60:
                return
            self.items_at = time.monotonic()
            base = self.base
            if time.monotonic() - self.state_at >= self.state_every:
                self.state_at = time.monotonic()
                try:
                    t = time.time()
                    subprocess.run([sys.executable, str(ROOT / "scripts" / self.script), "state"], cwd=ROOT,
                                   check=True, capture_output=True, timeout=1500)
                    base = LiveState5.load(None)
                    print(f"[{who}] estado de usuarios recargado en {time.time() - t:.0f}s (corte {base.t0})")
                except Exception as e:
                    print(f"[{who}] recarga del estado falló, sigo con el anterior: {e}")
            try:
                t = time.time()
                base = base.with_items()
                if time.time() - t > 20:
                    print(f"[{who}] refresco de producto lento: {time.time() - t:.0f}s")
            except Exception as e:
                print(f"[{who}] refresco de producto falló, sigo con el anterior: {e}")
            try:   # segunda capa: lo ocurrido desde el corte de la descarga
                delta = base.with_delta()
                d = delta.delta
                if d is not None and d.seconds > 20:
                    print(f"[{who}] delta de usuarios lento: {d.seconds:.0f}s ({d.events:,} eventos desde {d.t0})")
            except Exception as e:
                print(f"[{who}] delta de usuarios falló; este refresco v6-bird va sin delta: {e}")
                delta = base
            self.base, self.plain, self.delta = base, base, delta
        finally:
            self.lock.release()


class BirdRank(BirdV2):
    """bird_v2 con el modelo entrenado para ordenar (YetiRank). Solo, sobre todo el catálogo, es peor que
    bird_v2 (aprende el orden dentro de un carrusel, no el nivel absoluto); su sitio es dentro de la mezcla."""
    name = "bird_v2_rank"
    model_name = "v2_rank"
    final_name = "v2_rank_live"


class BirdBlend(Policy):
    """Mezcla: personal_v5 propone sus N mejores candidatos y Bird decide el orden.

    Cada uno cubre el punto débil del otro: personal acierta QUÉ productos merecen estar (también con
    usuarios nuevos) y Bird los ORDENA mejor. Offline (corte 07/11): ips_ndcg 0,25 frente a 0,21 de Bird
    solo y 0,13 de personal_v5; con usuario oculto 0,15 frente a 0,12 y 0,11. Ver scripts/bird_blend.py.
    """
    name = "bird_blend"
    bird_class = BirdV2
    n = 50
    refresh_every = 120.0

    def fit(self, state):
        from .personal import PersonalV5
        self.personal = PersonalV5(until=self.until, weights={"top": self.n})
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


class BirdTwoStage(BirdV2):
    """Política "bird_two_stage" (v4-bird): dos modelos de CatBoost seguidos, sin heurística.

      1. selector   el clasificador de bird_v2 puntúa todo el catálogo y se queda con los `n` mejores
      2. ordenador  un modelo de ranking (dataset v3) decide el orden usando las features de lista de
                    reco/bird/stage2.py: qué opina el selector, cómo queda el producto frente a los demás
                    candidatos, redundancia, y unas pocas señales directas

    Mismas exclusiones que bird_v2 (activo, en stock, nunca lo ya comprado).
    """
    name = "bird_two_stage"
    ranker_name = "v3_rank"            # ordenador validado (para evaluar offline)
    ranker_final = "v3_rank_live"      # ordenador entrenado con todos los datos (solo en vivo, si existe)
    n = 50

    def fit(self, state):
        from ..config import DATA_DIR
        super().fit(state)
        folder = DATA_DIR / "bird" / self.ranker_name
        final = DATA_DIR / "bird" / self.ranker_final
        if not self.until and (final / "model.cbm").exists():
            folder = final
        self.ranker = BirdModel.load(folder)
        print(f"[{self.name}] ordenador {folder.name} ({self.ranker.info.get('trees', '?')} árboles)")

    def recommend(self, req, state):
        from ..bird import stage2
        live = self.live
        mask = self._valid_mask(state, live).copy()
        mask[live.bought(live.user(req))] = False
        for p in self._sold.get(req.get("user_id"), ()):
            i = live.idx.get(p)
            if i is not None:
                mask[i] = False
        if not mask.any():
            return []
        cols, prob, chosen = stage2.select(live, self.model, req, mask, size=self.n)
        feats = {**stage2.list_features(live, cols, prob, chosen, chosen), **stage2.direct_features(cols, chosen),
                 "position": np.full(len(chosen), float(self.ranker.schema.reference_position))}
        score = self.ranker.predict(pd.DataFrame(feats))
        return live.ids[chosen[np.argsort(-score, kind="stable")[:self.top]]].tolist()
