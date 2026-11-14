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

    def recommend(self, req, state):
        w, ix = self.w, self.ix   # una sola lectura: si el hilo de refresh cambia el índice, no nos afecta
        s: dict[str, float] = defaultdict(float)

        views = req.get("recent_views") or []
        for k, v in enumerate(reversed(views)):
            d = w["decay"] ** k
            p = v["product_id"]
            s[p] += w["self"] * d
            for q, n in ix.nb_buy.get(p, ()):
                s[q] += w["buy"] * d * n
            for q, n in ix.nb_view.get(p, ()):
                s[q] += w["view"] * d * n

        user = None if req.get("is_new_user") else req.get("user_id")
        for sub, a in ix.user_subcat.get(user, ()):
            for q, n in ix.subcat_top.get(sub, ()):
                s[q] += w["user"] * a * n

        ctx = req.get("context") or {}
        for q, n in ix.ctx_pop.get((ctx.get("device"), ctx.get("region")), ()):
            s[q] += w["ctx"] * n
        for q, n in ix.top_pop:
            s[q] += w["pop"] * n

        bought = ix.user_bought.get(user, ())
        if bought:
            now = _epoch(req.get("ts"))
            for q, n, t in bought:
                s[q] += w["rebuy"] * (0.5 + 0.125 * n) * math.exp(-max(now - t, 0) / 86400 / w["rebuy_days"])

        brands = ix.user_brand.get(user)
        if brands:
            for q in s:
                s[q] *= 1 + w["brand"] * brands.get(ix.brand_of.get(q), 0)
        return sorted(s, key=s.get, reverse=True)[:10]


def _epoch(ts) -> float:
    """ts de la petición ('YYYY-MM-DD HH:MM:SS.fff…', UTC) en segundos; ahora si falta."""
    try:
        return datetime.fromisoformat(str(ts)[:19]).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return datetime.now(timezone.utc).timestamp()
