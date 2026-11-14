"""Team-draft reproducible; disco y red fuera de recommend()."""
from __future__ import annotations

import atexit
import json
import os
import queue
import random
import sys
import threading
import time
import zlib
from datetime import datetime, timezone

from .base import Policy
from .personal import Personal, PersonalV3
from .ranker import Ranker, RankerFeedback
from ..config import LOGS_DIR


def mezclar(a, b, request_id):
    """Prefijo común neutral; cada ronda aporta uno de cada equipo sin repetir."""
    out, autores = [], []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
        autores.append("ambas")
    rng = random.Random(zlib.crc32(str(request_id).encode()))
    usados = set(out)
    while len(out) < 10:
        orden = (0, 1) if rng.getrandbits(1) else (1, 0)
        for equipo in orden:
            producto = next((p for p in (a, b)[equipo] if p not in usados), None)
            if producto is not None:
                out.append(producto)
                usados.add(producto)
                autores.append(("A", "B")[equipo])
            if len(out) == 10:
                break
        else:
            if all(p in usados for p in a + b):
                break
    return out, autores


class _Registro:
    def __init__(self):
        self.cola = queue.Queue(maxsize=10000)
        self.parar = threading.Event()
        self.fallo = False
        self.avisado = False
        self.remoto = os.environ.get("RECO_INTERLEAVE_REMOTE") == "1"
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
            print("[interleave] fallo de registro: puede haber filas perdidas", file=sys.stderr)
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
                with (LOGS_DIR / "interleave.jsonl").open("a") as f:
                    f.write("".join(json.dumps(r) + "\n" for r in batch))
            except Exception:
                self._aviso()
            if self.remoto:
                try:
                    from ..client import insert
                    insert("scratch", batch)
                except Exception:
                    self._aviso()


class Interleave(Policy):
    name = "interleave"
    pair = (PersonalV3, Ranker)

    def __init__(self, a=None, b=None, until=None):
        super().__init__(until)
        self.a = a if a is not None else self.pair[0](until=until)
        self.b = b if b is not None else self.pair[1](until=until)
        self.refresh_every = min(self.a.refresh_every, self.b.refresh_every)
        self._refreshed = [float('-inf'), float('-inf')]
        self.registro = None

    def fit(self, state):
        for p in (self.a, self.b):
            p.fit(state)
        if self.registro is None:
            self.registro = _Registro()

    def refresh(self, state):
        errores = []
        for i, p in enumerate((self.a, self.b)):
            if time.monotonic() - self._refreshed[i] >= p.refresh_every:
                self._refreshed[i] = time.monotonic()
                try:
                    p.refresh(state)
                except Exception as e:
                    errores.append(e)
        if errores:
            raise errores[0]

    def recommend(self, req, state):
        listas = []
        for p in (self.a, self.b):
            try:
                # Se limpia conservando el orden antes de decidir si hay diez elegibles.
                lista = list(dict.fromkeys(x for x in p.recommend(req, state) if state.is_valid(x)))
                listas.append(lista[:10] if len(lista) >= 10 else None)
            except Exception:
                listas.append(None)
        a, b = listas
        if a is None or b is None:
            return a if a is not None else b if b is not None else state.fallback()
        out, autores = mezclar(a, b, req.get("request_id", ""))
        if self.registro is not None:
            self.registro.put(dict(kind="interleave", request_id=req.get("request_id"),
                                   shown=out, owners=autores, policy_a=self.a.name,
                                   policy_b=self.b.name, ts=datetime.now(timezone.utc).isoformat(),
                                   is_new_user=bool(req.get("is_new_user")),
                                   home=(req.get("context") or {}).get("page") == "home",
                                   dry_run="--dry-run" in sys.argv))
        return out


def crear_par(nombre, a, b):
    """Añadir pares solo requiere otra entrada en PARES."""
    return type(nombre, (Interleave,), {"name": nombre, "pair": (a, b)})


PARES = [crear_par(nombre, a, b) for nombre, a, b in (
    ("il_v3_gulias", PersonalV3, Ranker),
    ("il_v3_guliasfb", PersonalV3, RankerFeedback),
    ("il_gulias_guliasfb", Ranker, RankerFeedback),
    ("il_v2_v3", Personal, PersonalV3),
)]
