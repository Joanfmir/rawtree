"""Ranker aprendido: genera candidatos y features para cada (petición, producto) y los puntúa con LightGBM.

El mismo código construye las features al entrenar (sobre los 10 productos que enseñó el sistema viejo en
shop.history_sessions) y al servir (sobre ~200 candidatos), así que lo que se evalúa es lo que corre en vivo.

Recompra: NO se usa. Los productos que el usuario ya compró se quitan de los candidatos (el simulador dice
que quien ya compró algo tiende a no quererlo otra vez) y, al entrenar, se descartan esas impresiones.

Features (FEATURES, en este orden):
  sesión     in_views, view_recency, view_count, cobuy, coview, same_sub_last, same_cat_last,
             sub_share_views, cat_share_views, same_brand_views, logp_vs_views
  producto   logp, pop_all, pop_7d, pop_1d, views_7d, buy_rate, cart_rate, sub_rank, trend, is_new
  usuario    known, n_events, user_sub, user_cat, user_brand, logp_vs_user
  contexto   ctx_pop, page, device
  posición   pos (solo al entrenar; al servir se pone 0 = "como si estuviera arriba")
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

FEATURES = [
    "in_views", "view_recency", "view_count", "cobuy", "coview", "same_sub_last", "same_cat_last",
    "sub_share_views", "cat_share_views", "same_brand_views", "logp_vs_views",
    "logp", "pop_all", "pop_7d", "pop_1d", "views_7d", "buy_rate", "cart_rate", "sub_rank", "trend", "is_new",
    "known", "n_events", "user_sub", "user_cat", "user_brand", "logp_vs_user",
    "ctx_pop", "page", "device", "pos",
]
PAGES = {"home": 0, "category": 1, "product": 2}
DEVICES = {"web": 0, "ios": 1, "android": 2}
DECAY = 0.6


def _views(req) -> list[str]:
    return [v["product_id"] for v in (req.get("recent_views") or []) if v and v.get("product_id")]


class Featurizer:
    def __init__(self, feats: dict[str, pd.DataFrame]) -> None:
        it = feats["item_features"].copy()
        for c in ("views", "carts", "purchases", "views_7d", "purchases_7d", "purchases_1d"):
            it[c] = it[c].fillna(0).astype(float)
        it["price_eur"] = it.price_eur.fillna(it.price_eur.median()).astype(float)
        self.ids = it.product_id.to_numpy()
        self.idx = {p: i for i, p in enumerate(self.ids)}
        n = len(it)
        self.sub = it.subcategory.astype("category").cat.codes.to_numpy()
        self.sub_name = dict(zip(it.product_id, it.subcategory))
        self.cat = it.category.astype("category").cat.codes.to_numpy()
        self.cat_name = dict(zip(it.product_id, it.category))
        self.brand = it.brand.astype("category").cat.codes.to_numpy()
        self.brand_name = dict(zip(it.product_id, it.brand))
        self.logp = np.log1p(it.price_eur.to_numpy())

        # producto nuevo (sin historia): hereda la popularidad mediana de su subcategoría
        new = (it.views.to_numpy() == 0) | it.is_new.fillna(0).astype(bool).to_numpy()
        self.is_new = new.astype(float)
        it["pop_all"] = np.log1p(it.purchases)
        it["pop_7d"] = np.log1p(it.purchases_7d)
        it["pop_1d"] = np.log1p(it.purchases_1d)
        it["lv7"] = np.log1p(it.views_7d)
        it["buy_rate"] = (it.purchases + 1) / (it.views + 50)
        it["cart_rate"] = (it.carts + 1) / (it.views + 50)
        it["trend"] = np.log1p(it.purchases_1d * 7) - np.log1p(it.purchases_7d)
        cols = ["pop_all", "pop_7d", "pop_1d", "lv7", "buy_rate", "cart_rate", "trend"]
        med = it[~new].groupby("subcategory")[cols].median()
        for c in cols:
            fill = it.subcategory.map(med[c]).fillna(it.loc[~new, c].median()).to_numpy()
            it[c] = np.where(new, fill, it[c].to_numpy())
        self.item_num = it[cols].to_numpy(dtype=float)
        it["sub_rank"] = it.groupby("subcategory").pop_7d.rank(ascending=False, method="first")
        self.sub_rank = np.log1p(it.sub_rank.to_numpy())

        # top productos por subcategoría / categoría / global (para generar candidatos)
        score = it.pop_7d.to_numpy() + 0.5 * it.pop_all.to_numpy() / max(it.pop_all.max(), 1) * 3
        order = np.argsort(-score)
        self.top_global = order[:60]
        self.top_sub: dict[int, np.ndarray] = defaultdict(lambda: np.array([], int))
        self.top_cat: dict[int, np.ndarray] = defaultdict(lambda: np.array([], int))
        for s in np.unique(self.sub):
            self.top_sub[s] = order[self.sub[order] == s][:25]
        for c in np.unique(self.cat):
            self.top_cat[c] = order[self.cat[order] == c][:60]

        nb = feats["item_neighbors"]
        self.nb = {}
        for kind in ("buy", "view"):
            d = nb[nb.kind == kind]
            d = d.assign(nn=d.n / d.groupby("p").n.transform("max"))
            d = d.sort_values(["p", "nn"], ascending=[True, False])
            self.nb[kind] = {p: [(self.idx[q], v) for q, v in zip(g.q, g.nn) if q in self.idx][:20]
                             for p, g in d.groupby("p", sort=False)}

        us = feats["user_subcat"]
        us = us.assign(a=us.w / us.groupby("user_id").w.transform("sum"))
        self.user_sub = {u: dict(zip(g.subcategory, g.a)) for u, g in us.groupby("user_id", sort=False)}
        uc = feats["user_cat"]
        uc = uc.assign(a=uc.w / uc.groupby("user_id").w.transform("sum"))
        self.user_cat = {u: dict(zip(g.category, g.a)) for u, g in uc.groupby("user_id", sort=False)}
        ub = feats["user_brand"]
        ub = ub.assign(a=ub.w / ub.groupby("user_id").w.transform("max"))
        self.user_brand = {u: dict(zip(g.brand, g.a)) for u, g in ub.groupby("user_id", sort=False)}
        st = feats["user_stats"]
        self.user_stats = {u: (np.log1p(n), lp if lp == lp else lpa)
                           for u, n, lp, lpa in zip(st.user_id, st.n_events, st.buy_logp, st.avg_logp)}
        up = feats["user_purchases"]
        self.bought: dict[str, set[int]] = defaultdict(set)
        for u, p in zip(up.user_id, up.product_id):
            if p in self.idx:
                self.bought[u].add(self.idx[p])

        cp = feats["context_popular"]
        cp = cp.assign(a=cp.n / cp.groupby(["device", "region"]).n.transform("max"))
        self.ctx_pop = {k: {self.idx[p]: a for p, a in zip(g.product_id, g.a) if p in self.idx}
                        for k, g in cp.groupby(["device", "region"], sort=False)}

    # ── candidatos ───────────────────────────────────────────────────────────────────────────────
    def candidates(self, req, user: str | None) -> np.ndarray:
        views = [self.idx[p] for p in _views(req) if p in self.idx]
        c: list[int] = list(views)
        for p in _views(req)[-5:]:
            for kind in ("buy", "view"):
                c += [q for q, _ in self.nb[kind].get(p, ())]
        for s in {self.sub[v] for v in views[-5:]}:
            c += list(self.top_sub[s])
        if views:
            c += list(self.top_cat[self.cat[views[-1]]])
        if user in self.user_sub:
            subs = sorted(self.user_sub[user].items(), key=lambda x: -x[1])[:3]
            for name, _ in subs:
                c += list(self._top_sub_by_name(name)[:10])
        ctx = req.get("context") or {}
        c += list(self.ctx_pop.get((ctx.get("device"), ctx.get("region")), {}).keys())[:30]
        c += list(self.top_global)
        out = np.unique(np.array(c, dtype=int))
        bought = self.bought.get(user)
        if bought:
            out = out[~np.isin(out, list(bought))]
        return out

    def _top_sub_by_name(self, name: str) -> np.ndarray:
        if not hasattr(self, "_sub_codes"):
            self._sub_codes = {self.sub_name[p]: self.sub[i] for i, p in enumerate(self.ids)}
        code = self._sub_codes.get(name)
        return self.top_sub[code] if code is not None else np.array([], int)

    # ── features ─────────────────────────────────────────────────────────────────────────────────
    def matrix(self, req, user: str | None, cand: np.ndarray, pos: np.ndarray | None = None) -> np.ndarray:
        n = len(cand)
        X = np.zeros((n, len(FEATURES)), dtype=np.float32)
        vids = _views(req)
        views = [self.idx[p] for p in vids if p in self.idx]
        nv = len(views)

        # sesión
        recency = defaultdict(float)
        count = defaultdict(float)
        cobuy = defaultdict(float)
        coview = defaultdict(float)
        for k, p in enumerate(reversed(vids)):
            d = DECAY ** k
            if p in self.idx:
                i = self.idx[p]
                recency[i] = max(recency[i], d)
                count[i] += 1
            if k < 5:
                for q, v in self.nb["buy"].get(p, ()):
                    cobuy[q] += d * v
                for q, v in self.nb["view"].get(p, ()):
                    coview[q] += d * v
        X[:, 0] = [1.0 if c in recency else 0.0 for c in cand]
        X[:, 1] = [recency.get(c, 0.0) for c in cand]
        X[:, 2] = [count.get(c, 0.0) for c in cand]
        X[:, 3] = [cobuy.get(c, 0.0) for c in cand]
        X[:, 4] = [coview.get(c, 0.0) for c in cand]
        csub, ccat, cbr = self.sub[cand], self.cat[cand], self.brand[cand]
        if nv:
            last = views[-1]
            vsub, vcat, vbr = self.sub[views], self.cat[views], self.brand[views]
            X[:, 5] = csub == self.sub[last]
            X[:, 6] = ccat == self.cat[last]
            X[:, 7] = (csub[:, None] == vsub[None, :]).mean(1)
            X[:, 8] = (ccat[:, None] == vcat[None, :]).mean(1)
            X[:, 9] = (cbr[:, None] == vbr[None, :]).any(1)
            X[:, 10] = self.logp[cand] - self.logp[views].mean()
        else:
            X[:, 5:10] = -1
        # producto
        X[:, 11] = self.logp[cand]
        X[:, 12:18] = self.item_num[cand, :6]   # pop_all pop_7d pop_1d views_7d buy_rate cart_rate
        X[:, 18] = self.sub_rank[cand]
        X[:, 19] = self.item_num[cand, 6]
        X[:, 20] = self.is_new[cand]
        # usuario
        if user is not None and user in self.user_stats:
            X[:, 21] = 1
            ne, ulp = self.user_stats[user]
            X[:, 22] = ne
            us, uc, ub = self.user_sub.get(user, {}), self.user_cat.get(user, {}), self.user_brand.get(user, {})
            ids = self.ids[cand]
            X[:, 23] = [us.get(self.sub_name[p], 0.0) for p in ids]
            X[:, 24] = [uc.get(self.cat_name[p], 0.0) for p in ids]
            X[:, 25] = [ub.get(self.brand_name[p], 0.0) for p in ids]
            X[:, 26] = self.logp[cand] - ulp
        else:
            X[:, 22:27] = 0
        # contexto
        ctx = req.get("context") or {}
        cp = self.ctx_pop.get((ctx.get("device"), ctx.get("region")), {})
        X[:, 27] = [cp.get(c, 0.0) for c in cand]
        X[:, 28] = PAGES.get(ctx.get("page"), -1)
        X[:, 29] = DEVICES.get(ctx.get("device"), -1)
        X[:, 30] = 0 if pos is None else pos
        return X
