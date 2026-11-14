"""Política personalizada: suma de señales por producto candidato.

Candidatos y puntuación (los pesos están en DEFAULT_WEIGHTS y se ajustan con reco.evaluate):
  self      el propio producto que acaba de mirar (el 16 % de las compras del log son de algo recién visto)
  buy       vecinos "miró p → compró / añadió al carrito q" de cada producto visto
  view      vecinos "vistos en la misma sesión"
  user      top productos de las subcategorías favoritas del usuario
  ctx       más comprados en su (device, region) en las últimas 2 semanas: clave para usuarios nuevos
  pop       popularidad de los últimos 7 días (prior pequeño para todos)
Multiplicadores:
  brand     × (1 + brand · afinidad del usuario por la marca)
  rebuy     productos que el usuario ya compró: el 62 % de las compras del test son repeticiones y un
            producto ya comprado se compra 6,5× más. Peso (0,5 + 0,5·min(n,4)/4) · exp(-días / rebuy_days)
Las vistas recientes pesan más cuanto más recientes: × decay^k (k = 0 la última).
"""
from __future__ import annotations

import math
import subprocess
import sys
import threading
from collections import defaultdict
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .base import Policy

# Ajustados con scripts/tune.py search (búsqueda coordenada sobre ips_ndcg, corte 2026-11-07)
DEFAULT_WEIGHTS = {
    "self": 0.15, "buy": 1.0, "view": 0.1, "user": 1.0, "ctx": 0.15, "pop": 0.4,
    "brand": 0.5, "rebuy": 1.0, "rebuy_days": 30.0, "decay": 0.6,
}

# Señales que devuelve signals(), en este orden. Las anteriores a 'brand' se suman con su peso;
# 'brand' multiplica; 'score' es el resultado.
SIGNALS = ("self", "buy", "view", "user", "ctx", "pop", "rebuy", "brand", "score")
_SELF, _BUY, _VIEW, _USER, _CTX, _POP, _REBUY, _BRAND, _SCORE = range(len(SIGNALS))


def _norm_lists(df, key, item, val, top):
    """{key: [(item, val / max_val_of_key), ...]} ordenado por val desc, máximo `top` por clave."""
    df = df.sort_values([key, val], ascending=[True, False])
    out: dict = {}
    cur, lst, vmax = object(), None, 1.0
    for k, i, v in zip(df[key].tolist(), df[item].tolist(), df[val].tolist()):
        if k != cur:
            cur, lst, vmax = k, [], max(float(v), 1e-9)
            out[k] = lst
        if len(lst) < top:
            lst.append((i, v / vmax))
    return out


def _group(keys, *cols):
    """{key: [(col1, col2, ...), ...]} conservando el orden de las filas."""
    out: dict = {}
    for k, *vals in zip(keys, *cols):
        out.setdefault(k, []).append(tuple(vals))
    return out


class Personal(Policy):
    name = "personal"

    def __init__(self, until: str | None = None, weights: dict | None = None) -> None:
        super().__init__(until)
        self.w = {**DEFAULT_WEIGHTS, **(weights or {})}

    refresh_every = 600.0   # en vivo: recalcula las features cada 10 min en segundo plano

    def fit(self, state):
        """Carga las features del disco (si no hay, las construye). En producción, el primer refresh()
        del bucle las recalcula en segundo plano, así que arrancar con unas antiguas es seguro y rápido."""
        self.ix = self._index(self.load_features())
        self._building = False

    def refresh(self, state):
        """En vivo: reconstruye las features en un hilo y cambia el índice de golpe al terminar."""
        if self.until is not None or self._building:
            return
        self._building = True

        def work():
            # La descarga y el JSON (600k filas) bloquean el GIL varios segundos: en un proceso aparte
            # no frenan el bucle de respuestas. Aquí solo leemos el Parquet y montamos el índice (~1,5 s).
            from .. import features
            from ..config import ROOT
            try:
                subprocess.run([sys.executable, str(ROOT / "scripts" / "build_features.py")], cwd=ROOT,
                               check=True, capture_output=True, timeout=600)
                self.ix = self._index(features.load(until=None))
                print(f"[personal] features actualizadas: {len(self.ix.user_bought)} usuarios con compras")
            except Exception as e:
                print(f"[personal] refresh falló, sigo con las anteriores: {e}")
            finally:
                self._building = False

        threading.Thread(target=work, daemon=True).start()

    @staticmethod
    def _index(f) -> SimpleNamespace:
        """De las tablas de features a diccionarios rápidos de consultar."""
        ix = SimpleNamespace()
        items = f["item_features"].fillna(0)
        ix.brand_of = dict(zip(items.product_id, items.brand))
        ix.subcat_of = dict(zip(items.product_id, items.subcategory))
        pop = np.log1p(items.purchases_7d.to_numpy(dtype=float))
        items = items.assign(pop=pop / max(pop.max(), 1e-9))
        ix.top_pop = list(items.nlargest(50, "pop")[["product_id", "pop"]].itertuples(index=False, name=None))
        ix.subcat_top = _norm_lists(items, "subcategory", "product_id", "pop", 10)

        nb = f["item_neighbors"]
        ix.nb_buy = _norm_lists(nb[nb.kind == "buy"], "p", "q", "n", 20)
        ix.nb_view = _norm_lists(nb[nb.kind == "view"], "p", "q", "n", 20)

        us = f["user_subcat"].sort_values(["user_id", "w"], ascending=[True, False])
        us = us.assign(a=us.w / us.groupby("user_id").w.transform("sum"))
        ix.user_subcat = {u: v[:5] for u, v in _group(us.user_id.tolist(), us.subcategory.tolist(), us.a.tolist()).items()}
        ub = f["user_brand"]
        ub = ub.assign(a=ub.w / ub.groupby("user_id").w.transform("max"))
        ix.user_brand = {u: dict(v) for u, v in _group(ub.user_id.tolist(), ub.brand.tolist(), ub.a.tolist()).items()}
        up = f["user_purchases"]
        t = (pd.to_datetime(up.last_ts) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
        ix.user_bought = _group(up.user_id.tolist(), up.product_id.tolist(), up.n.clip(upper=4).tolist(), t.tolist())

        cp = f["context_popular"].assign(ctx=lambda d: list(zip(d.device, d.region)))
        ix.ctx_pop = _norm_lists(cp, "ctx", "product_id", "n", 30)
        return ix

    def signals(self, req, state=None):
        """Señales SIN pesos de cada candidato: (SIGNALS, {producto: [valor por señal]}).
        La última, 'score', es la nota de esta política (suma con pesos × marca). recommend() ordena
        por ella y el re-ranker (policies/hgb.py) usa todas como columnas: una señal nueva añadida
        aquí y en SIGNALS le llega sola al reentrenar."""
        w, ix = self.w, self.ix   # una sola lectura: si el hilo de refresh cambia el índice, no nos afecta
        c: dict[str, list[float]] = defaultdict(lambda: [0.0] * len(SIGNALS))

        views = req.get("recent_views") or []
        for k, v in enumerate(reversed(views)):
            d = w["decay"] ** k
            p = v["product_id"]
            c[p][_SELF] += d
            for q, n in ix.nb_buy.get(p, ()):
                c[q][_BUY] += d * n
            for q, n in ix.nb_view.get(p, ()):
                c[q][_VIEW] += d * n

        user = None if req.get("is_new_user") else req.get("user_id")
        for sub, a in ix.user_subcat.get(user, ()):
            for q, n in ix.subcat_top.get(sub, ()):
                c[q][_USER] += a * n

        ctx = req.get("context") or {}
        for q, n in ix.ctx_pop.get((ctx.get("device"), ctx.get("region")), ()):
            c[q][_CTX] += n
        for q, n in ix.top_pop:
            c[q][_POP] += n

        bought = ix.user_bought.get(user, ())
        if bought:
            now = _epoch(req.get("ts"))
            for q, n, t in bought:
                days = max(now - t, 0) / 86400
                # rebuy_ramp > 0 (personal2): al revés, lo recién comprado casi no cuenta y sube hasta 1
                ramp = w.get("rebuy_ramp", 0)
                rec = min(days / ramp, 1.0) if ramp else math.exp(-days / w["rebuy_days"])
                c[q][_REBUY] += (0.5 + 0.125 * n) * rec

        brands = ix.user_brand.get(user)
        ws, wb = [w[k] for k in SIGNALS[:_BRAND]], w["brand"]
        for q, v in c.items():
            if brands:
                v[_BRAND] = brands.get(ix.brand_of.get(q), 0)
            v[_SCORE] = sum(x * y for x, y in zip(v, ws)) * (1 + wb * v[_BRAND])
        return SIGNALS, c

    def recommend(self, req, state):
        _, c = self.signals(req, state)
        return sorted(c, key=lambda q: c[q][_SCORE], reverse=True)[:10]


def _epoch(ts) -> float:
    """ts de la petición ('YYYY-MM-DD HH:MM:SS.fff…', UTC) en segundos; ahora si falta."""
    try:
        return datetime.fromisoformat(str(ts)[:19]).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return datetime.now(timezone.utc).timestamp()
