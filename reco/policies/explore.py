"""Exploración uniforme de un único slot, con probabilidades auditables."""
from __future__ import annotations

import atexit
import json
import os
import queue
import random
import sys
import threading
from datetime import datetime, timezone

from .base import Policy
from .personal import Personal
from .ranker import Ranker
from ..config import LOGS_DIR


class _Registro:
    """La ruta crítica solo encola; disco y red pertenecen a este hilo."""

    def __init__(self):
        self.cola = queue.Queue(maxsize=10000)
        self.parar = threading.Event()
        self.fallo = False
        self.avisado = False
        self.remoto = os.environ.get("RECO_EXPLORE_REMOTE") == "1"
        self.hilo = threading.Thread(target=self._trabajar, daemon=True)
        self.hilo.start()
        atexit.register(self.close)

    def put(self, row):
        try:
            self.cola.put_nowait(row)
        except queue.Full:
            self.fallo = True

    def close(self):
        self.parar.set()
        self.hilo.join(timeout=6)

    def _aviso(self):
        if not self.avisado:
            print("[explore] fallo de registro: puede haber filas perdidas", file=sys.stderr)
            self.avisado = True

    def _trabajar(self):
        while not self.parar.is_set() or not self.cola.empty():
            batch = []
            try:
                batch.append(self.cola.get(timeout=0.5))
            except queue.Empty:
                pass
            while len(batch) < 100:
                try:
                    batch.append(self.cola.get_nowait())
                except queue.Empty:
                    break
            if self.fallo:
                self._aviso()
            if not batch:
                continue
            try:
                LOGS_DIR.mkdir(parents=True, exist_ok=True)
                with (LOGS_DIR / "explore.jsonl").open("a") as f:
                    f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in batch))
            except Exception:
                self._aviso()
            if self.remoto:
                try:
                    from ..client import insert
                    insert("scratch", batch)
                except Exception:
                    self._aviso()


class Explore(Policy):
    """Uniforme sobre la unión de populares elegibles; slots públicos en base 1.

    La propensity es conjunta (activar exploración, elegir slot y producto).
    No describe la probabilidad de toda la lista ni permite evaluar fuera del soporte.
    """
    name = "explore"
    inner_class = Personal

    def __init__(self, inner=None, eps=0.10, until=None, seed=None):
        super().__init__(until)
        if not 0 <= eps <= 1:
            raise ValueError("eps debe estar entre 0 y 1")
        self.inner = inner if inner is not None else self.inner_class(until=until)
        self.eps = eps
        self.refresh_every = self.inner.refresh_every
        self.rng = random.Random(seed)
        self.registro = None

    def fit(self, state):
        self.inner.fit(state)
        self._pools(state)
        if self.registro is None:
            self.registro = _Registro()

    def _pools(self, state):
        # Ranking de ventas del estado: no se consulta RawTree desde recommend().
        cats = {}
        for p in state.best:
            c = state.catalog.get(p, {}).get("category")
            if c is not None:
                pool = cats.setdefault(c, [])
                if len(pool) < 40:
                    pool.append(p)
        self.cat_top = cats
        self.global_top = state.best[:60]

    def refresh(self, state):
        self.inner.refresh(state)
        self._pools(state)

    def recommend(self, req, state):
        original = self.inner.recommend(req, state)
        # Una lista incompleta/inválida sería reordenada por finalize(): no explorarla.
        if (len(original) != 10 or len(set(original)) != 10
                or any(not state.is_valid(p) for p in original)
                or self.rng.random() >= self.eps):
            return original
        cats = {state.catalog.get(v.get("product_id"), {}).get("category")
                for v in (req.get("recent_views") or []) if v}
        cats.discard(None)
        pool = (dict.fromkeys(p for c in sorted(cats) for p in self.cat_top.get(c, ()))
                if cats else self.global_top)
        candidates = [p for p in pool if p not in original and state.is_valid(p)]
        if not candidates:
            return original
        slot = self.rng.choice((8, 9, 10))
        product = self.rng.choice(candidates)
        out = list(original)
        out[slot - 1] = product
        row = dict(kind="explore", request_id=req.get("request_id"), slot=slot,
                   product_id=product, replaced_product_id=original[slot - 1],
                   propensity=self.eps / (3 * len(candidates)), eps=self.eps,
                   inner_policy=self.inner.name, ts=datetime.now(timezone.utc).isoformat(),
                   category=state.catalog.get(product, {}).get("category", "desconocida"),
                   candidate_count=len(candidates), slot_count=3, shown=out,
                   dry_run="--dry-run" in sys.argv)
        if self.registro is not None:
            self.registro.put(row)
        return out


class PersonalExplore(Explore):
    name = "personal_explore"
    inner_class = Personal


class GuliasExplore(Explore):
    name = "gulias_explore"
    inner_class = Ranker
