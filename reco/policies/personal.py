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
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .base import Policy

DEFAULT_WEIGHTS = {
    "self": 1.0, "buy": 1.0, "view": 0.3, "user": 0.5, "ctx": 0.3, "pop": 0.2,
    "brand": 0.5, "rebuy": 2.0, "rebuy_days": 30.0, "decay": 0.6,
}


def _norm_lists(df, key, item, val, top):
    """{key: [(item, val / max_val_of_key), ...]} ordenado por val, máximo `top` por clave."""
    out = {}
    df = df.sort_values([key, val], ascending=[True, False])
    for k, g in df.groupby(key, sort=False):
        v = g[val].to_numpy(dtype=float)[:top]
        out[k] = list(zip(g[item].to_numpy()[:top], v / max(v[0], 1e-9)))
    return out


class Personal(Policy):
    name = "personal"

    def __init__(self, until: str | None = None, weights: dict | None = None) -> None:
        super().__init__(until)
        self.w = {**DEFAULT_WEIGHTS, **(weights or {})}

    def fit(self, state):
        f = self.load_features()
        items = f["item_features"].fillna(0)
        self.brand_of = dict(zip(items.product_id, items.brand))
        pop = np.log1p(items.purchases_7d.to_numpy(dtype=float))
        items = items.assign(pop=pop / max(pop.max(), 1e-9))
        self.top_pop = list(items.nlargest(50, "pop")[["product_id", "pop"]].itertuples(index=False, name=None))
        self.subcat_top = _norm_lists(items, "subcategory", "product_id", "pop", 10)

        nb = f["item_neighbors"]
        self.nb_buy = _norm_lists(nb[nb.kind == "buy"], "p", "q", "n", 20)
        self.nb_view = _norm_lists(nb[nb.kind == "view"], "p", "q", "n", 20)

        us = f["user_subcat"]
        us = us.assign(a=us.w / us.groupby("user_id").w.transform("sum"))
        self.user_subcat = {u: list(zip(g.subcategory, g.a))[:5] for u, g in us.groupby("user_id", sort=False)}
        ub = f["user_brand"]
        ub = ub.assign(a=ub.w / ub.groupby("user_id").w.transform("max"))
        self.user_brand = {u: dict(zip(g.brand, g.a)) for u, g in ub.groupby("user_id", sort=False)}
        up = f["user_purchases"]
        t = (pd.to_datetime(up.last_ts) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
        self.user_bought = {u: list(zip(g.product_id, g.n.clip(upper=4), t[g.index]))
                            for u, g in up.groupby("user_id", sort=False)}

        cp = f["context_popular"].assign(ctx=lambda d: list(zip(d.device, d.region)))
        self.ctx_pop = _norm_lists(cp, "ctx", "product_id", "n", 30)

    def recommend(self, req, state):
        w = self.w
        s: dict[str, float] = defaultdict(float)

        views = req.get("recent_views") or []
        for k, v in enumerate(reversed(views)):
            d = w["decay"] ** k
            p = v["product_id"]
            s[p] += w["self"] * d
            for q, n in self.nb_buy.get(p, ()):
                s[q] += w["buy"] * d * n
            for q, n in self.nb_view.get(p, ()):
                s[q] += w["view"] * d * n

        user = None if req.get("is_new_user") else req.get("user_id")
        for sub, a in self.user_subcat.get(user, ()):
            for q, n in self.subcat_top.get(sub, ()):
                s[q] += w["user"] * a * n

        ctx = req.get("context") or {}
        for q, n in self.ctx_pop.get((ctx.get("device"), ctx.get("region")), ()):
            s[q] += w["ctx"] * n
        for q, n in self.top_pop:
            s[q] += w["pop"] * n

        bought = self.user_bought.get(user, ())
        if bought:
            now = _epoch(req.get("ts"))
            for q, n, t in bought:
                s[q] += w["rebuy"] * (0.5 + 0.125 * n) * math.exp(-max(now - t, 0) / 86400 / w["rebuy_days"])

        brands = self.user_brand.get(user)
        if brands:
            for q in s:
                s[q] *= 1 + w["brand"] * brands.get(self.brand_of.get(q), 0)
        return sorted(s, key=s.get, reverse=True)[:10]


def _epoch(ts) -> float:
    """ts de la petición ('YYYY-MM-DD HH:MM:SS.fff…', UTC) en segundos; ahora si falta."""
    try:
        return datetime.fromisoformat(str(ts)[:19]).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return datetime.now(timezone.utc).timestamp()


class PersonalNoRebuy(Personal):
    """La política personal de Joan sin el bonus de recompra (para compararla sin esa señal)."""
    name = "personal_norebuy"

    def __init__(self, until: str | None = None, weights: dict | None = None) -> None:
        super().__init__(until, {"rebuy": 0.0, **(weights or {})})
