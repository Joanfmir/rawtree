"""Fórmulas del dataset v5. UNA sola implementación para entrenar (build_v5.py) y para servir (live5.py).

En v2 cada feature estaba escrita dos veces (SQL + pandas al entrenar, numpy al servir) y había que vigilar
que dijeran lo mismo. Aquí los dos lados solo consiguen los RECUENTOS (cada uno a su manera: "hasta el
instante de la petición" al entrenar, "hasta ahora" al servir) y estas funciones los convierten en features.

Criterio de diseño (medido el 14/11, ver docs/DATASET_V5.md):
  - El stream en vivo tiene ~29 veces más eventos por hora que el histórico. Un recuento absoluto
    (compras en 7 días, eventos del usuario) no significa lo mismo en los dos sitios; una CUOTA o un
    COCIENTE sí. Todas las features de popularidad y de usuario son invariantes al volumen.
  - Lo que solo pasa en vivo (bajadas de precio, productos nuevos, tendencias de una hora) vale un valor
    neutro en el histórico (0 o lift 1) y se aprende de las filas de hoy.
"""
from __future__ import annotations

import math

import numpy as np

from ..features import ATTRS

DECAY = 0.6                # peso de cada vista de la sesión: la última 1, la anterior 0,6, ...
HALF_LIFE_DAYS = 2.0       # semivida de la afinidad "reciente" del usuario
TAU_MS = HALF_LIFE_DAYS * 86400e3 / math.log(2)
RATE_PRIOR = 50.0          # vistas ficticias con la tasa de la subcategoría (compras/vista de un producto)
PRIOR_IMPRESSIONS = 200.0  # impresiones ficticias con el CTR global (carrusel del histórico), como en v2
TREND_1H_PRIOR, TREND_1D_PRIOR = 5.0, 20.0
FB_PRIOR_CLICKS, FB_PRIOR_BUYS = 2.0, 1.0
CTX_PRIOR_GROUP, CTX_PRIOR_ITEM = 20.0, 10.0
HIST_CONF = 50.0           # u_hist_conf = eventos / (eventos + 50)
BASE_CATALOG_UNTIL = "2026-11-14 09:00:00"   # lo creado después son productos nuevos del stream


def stored_pair(views, carts, c_clicks, buys):
    """¿Guarda el estado de servicio este par usuario × producto? Solo los que tienen carrito, clic, compra
    o ≥ 2 vistas (los de una sola vista son millones). El dataset aplica la MISMA regla: un par que al
    servir no existe tampoco cuenta al entrenar."""
    return (np.asarray(carts) > 0) | (np.asarray(c_clicks) > 0) | (np.asarray(buys) > 0) | (np.asarray(views) >= 2)

ATTR_CODE = {a: i for i, a in enumerate(ATTRS)}
N_ATTR = len(ATTRS)

# ── nombres de las columnas que produce cada bloque ──────────────────────────────────────────────────
ITEM_POP = ("it_buy_share", "it_view_share", "it_buy_rate", "it_cart_rate", "it_buy_share_7d",
            "it_view_share_7d", "it_buy_share_1d", "it_trend_1d", "it_trend_1h", "it_sub_rank")
ITEM_FB = ("it_fb_click_lift", "it_fb_buy_lift")
CTX_LIFT = ("ctx_dev_item", "ctx_dev_sub", "ctx_dev_cat", "ctx_dev_brand", "ctx_dev_attr", "ctx_dev_price",
            "ctx_reg_item", "ctx_reg_sub", "ctx_reg_cat", "ctx_reg_attr")
SESSION_V2 = ("n_recent_views", "recently_viewed", "sv_recency", "sv_count", "sv_same_sub_last",
              "sv_same_cat_last", "sv_same_brand_last", "sv_sub_share", "sv_cat_share", "sv_brand_share",
              "sv_attr_share", "sv_logp_diff", "view_to_purchase_score", "coview_score")
SESSION_NEW = ("sv_secs_since_last", "sv_attr_last", "sv_attr_w", "sv_sub_w", "sv_cat_w", "sv_logp_diff_last",
               "sv_n_subs")
SESSION = SESSION_V2 + SESSION_NEW
USER = ("u_hist_conf", "u_buy_ratio", "u_logp_diff", "u_buy_logp_diff", "u_logp_z")
AFFINITY = ("u_sub_share", "u_brand_share", "u_cat_share", "u_attr_share", "u_attr_max_share",
            "u_attr_min_share", "u_sub_share_rec", "u_cat_share_rec", "u_attr_share_rec", "u_sub_buy_share",
            "u_brand_buy_share", "u_cat_buy_share", "u_attr_buy_share")
# recuentos por (usuario, candidato) de los que salen USER y AFFINITY: los da el SQL al entrenar y el
# estado en memoria al servir
USER_AGG = ("u_events", "u_buys", "u_w", "u_r", "u_logp", "u_buy_logp", "u_logp_sd", "u_sub_w", "u_brand_w",
            "u_cat_w", "u_attr_w1", "u_attr_w2", "u_sub_r", "u_cat_r", "u_attr_r", "u_sub_buys", "u_brand_buys",
            "u_cat_buys", "u_attr_buys")


# ── catálogo ─────────────────────────────────────────────────────────────────────────────────────────

class Items:
    """Arrays del catálogo (una posición por producto) que usan todos los bloques."""

    def __init__(self, cat) -> None:
        """`cat`: DataFrame ordenado por product_id con category, subcategory, brand, attrs (lista),
        base_price (precio del catálogo, sin promociones) y created_at."""
        self.ids = cat.product_id.to_numpy()
        self.idx = {p: i for i, p in enumerate(self.ids)}
        self.n = len(cat)
        self.codes: dict[str, dict[str, int]] = {}
        for col, name in (("subcategory", "sub"), ("category", "cat"), ("brand", "brand")):
            c = cat[col].astype("category")
            self.codes[name] = {k: i for i, k in enumerate(c.cat.categories)}
            setattr(self, name, c.cat.codes.to_numpy().astype(np.int64))
        attrs = [sorted([a for a in (v if v is not None else []) if a in ATTR_CODE], key=ATTR_CODE.get)
                 for v in cat["attrs"]]
        self.attr1 = np.array([a[0] if len(a) > 0 else "UNKNOWN" for a in attrs], dtype=object)
        self.attr2 = np.array([a[1] if len(a) > 1 else "UNKNOWN" for a in attrs], dtype=object)
        # sin atributo: código negativo y distinto en cada hueco, para que nunca cuente como coincidencia
        self.a1 = np.array([ATTR_CODE.get(a, -1) for a in self.attr1], dtype=np.int64)
        self.a2 = np.array([ATTR_CODE.get(a, -2) for a in self.attr2], dtype=np.int64)
        self.base_price = cat.base_price.to_numpy(dtype=float)
        self.logp = np.log1p(self.base_price)            # nivel de precio del producto (sin promociones)
        self.created = np.asarray(cat.created_at.to_numpy(), dtype="datetime64[ns]")
        # mediana de precio de la subcategoría: solo con el catálogo inicial, para que no se mueva cada vez
        # que entra un producto nuevo (y valga lo mismo en el dataset que al servir)
        original = self.created < np.datetime64(BASE_CATALOG_UNTIL)
        med = np.zeros(self.sub.max() + 1)
        for s in np.unique(self.sub):
            same = self.sub == s
            med[s] = np.median(self.logp[same & original] if (same & original).any() else self.logp[same])
        self.price_vs_sub = self.logp - med[self.sub]
        self.price_bin = np.clip(np.floor(np.log2(np.maximum(self.base_price, 1.0))), 0, 9).astype(np.int64)


# ── producto: popularidad a partir de recuentos por ventana ──────────────────────────────────────────

def item_pop(c: dict[str, np.ndarray], sub: np.ndarray) -> dict[str, np.ndarray]:
    """`c[k]`: matriz [instantes, productos] con views, carts, buys (todo lo anterior al instante) y
    views_7d, buys_7d, views_1d, buys_1d, views_1h (ventanas que acaban en el instante).
    Devuelve features invariantes al volumen de tráfico: cuotas (×10⁴), tasas y log-lifts suavizados."""
    v, ca, b = (np.atleast_2d(c[k]).astype(float) for k in ("views", "carts", "buys"))
    v7, b7, v1, b1, vh = (np.atleast_2d(c[k]).astype(float) for k in ("views_7d", "buys_7d", "views_1d",
                                                                    "buys_1d", "views_1h"))
    tot = lambda m: np.maximum(m.sum(axis=1, keepdims=True), 1.0)   # noqa: E731
    share = lambda m: m / tot(m)                                    # noqa: E731
    # tasa de compra por vista, encogida hacia la de su subcategoría (un producto nuevo hereda la de su grupo)
    n_sub = int(sub.max()) + 1
    onehot = np.zeros((len(sub), n_sub))
    onehot[np.arange(len(sub)), sub] = 1.0
    sub_v = np.maximum(v @ onehot, 1.0)
    r_buy, r_cart = ((m @ onehot) / sub_v for m in (b, ca))
    rank = np.zeros_like(b7)
    for s in range(n_sub):
        cols = np.flatnonzero(sub == s)
        if len(cols):
            rank[:, cols] = (-b7[:, cols]).argsort(axis=1, kind="stable").argsort(axis=1, kind="stable") + 1
    sv_all, sv_7 = share(v), share(v7)
    return {
        "it_buy_share": share(b) * 1e4,
        "it_view_share": sv_all * 1e4,
        "it_buy_rate": (b + RATE_PRIOR * r_buy[:, sub]) / (v + RATE_PRIOR),
        "it_cart_rate": (ca + RATE_PRIOR * r_cart[:, sub]) / (v + RATE_PRIOR),
        "it_buy_share_7d": share(b7) * 1e4,
        "it_view_share_7d": sv_7 * 1e4,
        "it_buy_share_1d": share(b1) * 1e4,
        # vistas de las últimas 24 h frente a las que le tocarían por su cuota de siempre
        "it_trend_1d": np.log((v1 + TREND_1D_PRIOR) / (tot(v1) * sv_all + TREND_1D_PRIOR)),
        # vistas de la última hora frente a las que le tocarían por su cuota de la semana
        "it_trend_1h": np.log((vh + TREND_1H_PRIOR) / (tot(vh) * sv_7 + TREND_1H_PRIOR)),
        "it_sub_rank": rank,
    }


def carousel_rates(impr, clicks, buys):
    """CTR y CVR en el carrusel del sistema viejo, encogidos hacia la media global (igual que v2)."""
    impr, clicks, buys = (np.atleast_2d(x).astype(float) for x in (impr, clicks, buys))
    total = np.maximum(impr.sum(axis=1, keepdims=True), 1.0)
    g_ctr, g_cvr = clicks.sum(axis=1, keepdims=True) / total, buys.sum(axis=1, keepdims=True) / total
    return {"it_c_ctr": (clicks + PRIOR_IMPRESSIONS * g_ctr) / (impr + PRIOR_IMPRESSIONS),
            "it_c_cvr": (buys + PRIOR_IMPRESSIONS * g_cvr) / (impr + PRIOR_IMPRESSIONS)}


def feedback_lift(clicks, exp_clicks, buys, exp_buys):
    """COEC (clicks over expected clicks) en NUESTRO carrusel: clics reales entre los que habría tenido un
    producto medio en esas mismas posiciones. Sin impresiones vale 1 (sin evidencia)."""
    return {"it_fb_click_lift": (np.asarray(clicks, float) + FB_PRIOR_CLICKS) / (np.asarray(exp_clicks, float) + FB_PRIOR_CLICKS),
            "it_fb_buy_lift": (np.asarray(buys, float) + FB_PRIOR_BUYS) / (np.asarray(exp_buys, float) + FB_PRIOR_BUYS)}


# ── contexto: qué compra cada dispositivo / región ───────────────────────────────────────────────────

def _group_lift(counts: np.ndarray, code: np.ndarray, size: int) -> np.ndarray:
    """log-lift [K, grupos] de P(grupo | contexto) frente a P(grupo), encogido hacia 0."""
    onehot = np.zeros((counts.shape[1], size))
    onehot[np.arange(len(code)), code] = 1.0
    g = counts @ onehot                                           # [K, grupos]
    n_k = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    expected = n_k * g.sum(axis=0, keepdims=True) / max(counts.sum(), 1.0)
    return np.log((g + CTX_PRIOR_GROUP) / (expected + CTX_PRIOR_GROUP))


def context_lifts(counts: np.ndarray, items: Items, prefix: str, levels: tuple[str, ...]) -> dict[str, np.ndarray]:
    """`counts`: compras orgánicas [valores del contexto, productos]. Devuelve, por nivel, una matriz
    [valores del contexto, productos] con el log-lift del producto (o de su grupo) en ese contexto.
    El nivel `item` se encoge hacia el de su subcategoría: un producto nuevo hereda el de su grupo."""
    counts = np.asarray(counts, dtype=float)
    out = {}
    sub_lift = _group_lift(counts, items.sub, int(items.sub.max()) + 1)[:, items.sub]
    for level in levels:
        if level == "sub":
            m = sub_lift
        elif level in ("cat", "brand"):
            code = getattr(items, level)
            m = _group_lift(counts, code, int(code.max()) + 1)[:, code]
        elif level == "price":
            m = _group_lift(counts, items.price_bin, 10)[:, items.price_bin]
        elif level == "attr":
            has = np.zeros((items.n, N_ATTR + 1))                  # columna extra: "sin atributo"
            has[np.arange(items.n), np.where(items.a1 >= 0, items.a1, N_ATTR)] = 1.0
            has[np.arange(items.n), np.where(items.a2 >= 0, items.a2, N_ATTR)] = 1.0
            g = counts @ has
            n_k = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
            expected = n_k * g.sum(axis=0, keepdims=True) / max(counts.sum(), 1.0)
            la = np.log((g + CTX_PRIOR_GROUP) / (expected + CTX_PRIOR_GROUP))
            la[:, N_ATTR] = 0.0
            m = (la[:, np.where(items.a1 >= 0, items.a1, N_ATTR)] + la[:, np.where(items.a2 >= 0, items.a2, N_ATTR)]) / 2
        elif level == "item":
            n_k = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
            expected = n_k * counts.sum(axis=0, keepdims=True) / max(counts.sum(), 1.0)
            m = np.log((counts + CTX_PRIOR_ITEM * np.exp(sub_lift)) / (expected + CTX_PRIOR_ITEM))
        else:
            raise ValueError(level)
        out[f"ctx_{prefix}_{level}"] = m
    return out


DEV_LEVELS = ("item", "sub", "cat", "brand", "attr", "price")
REG_LEVELS = ("item", "sub", "cat", "attr")


# ── sesión: recent_views frente a cada producto del catálogo ─────────────────────────────────────────

def session_block(items: Items, views: np.ndarray, secs_since_last: float,
                  nb: dict[str, dict[int, list[tuple[int, float]]]]) -> dict[str, np.ndarray]:
    """`views`: índices de los productos vistos, del más antiguo al más reciente. Una posición por
    producto del catálogo. `secs_since_last`: segundos entre la última vista y la petición (NaN si no hay)."""
    n = items.n
    z = np.zeros(n)
    out = {name: z for name in SESSION}
    out["n_recent_views"] = np.full(n, float(len(views)))
    out["sv_logp_diff"] = out["sv_logp_diff_last"] = out["sv_secs_since_last"] = np.full(n, np.nan)
    if len(views) == 0:
        return out
    va = np.asarray(views, dtype=np.int64)
    rev = va[::-1]                                   # la última vista primero
    k = len(rev)
    w = DECAY ** np.arange(k)
    count, recency = np.zeros(n), np.zeros(n)
    for i, x in enumerate(rev):
        count[x] += 1
        if recency[x] == 0:
            recency[x] = 1 / (i + 1)
    last = rev[0]

    def share(code: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        size = int(code.max()) + 1
        if weights is None:
            return np.bincount(code[rev], minlength=size)[code] / k
        return np.bincount(code[rev], weights=weights, minlength=size)[code] / weights.sum()

    # atributos: cuántas vistas (y con qué peso) tienen cada atributo en cualquiera de sus dos huecos
    a1v, a2v = items.a1[rev], items.a2[rev]
    per_attr, per_attr_w = np.zeros(N_ATTR + 1), np.zeros(N_ATTR + 1)
    for a_v in (a1v, a2v):
        ok = a_v >= 0
        np.add.at(per_attr, a_v[ok], 1.0)
        np.add.at(per_attr_w, a_v[ok], w[ok])
    i1, i2 = np.where(items.a1 >= 0, items.a1, N_ATTR), np.where(items.a2 >= 0, items.a2, N_ATTR)
    last_attrs = {int(a) for a in (items.a1[last], items.a2[last]) if a >= 0}
    in_last = np.zeros(N_ATTR + 1)
    in_last[list(last_attrs)] = 1.0
    score = {}
    for kind in ("buy", "view"):
        s = np.zeros(n)
        for i, x in enumerate(rev):
            for q, nn in nb[kind].get(int(x), ()):
                s[q] += w[i] * nn
        score[kind] = s
    out.update({
        "recently_viewed": (count > 0).astype(float), "sv_recency": recency, "sv_count": count,
        "sv_same_sub_last": (items.sub == items.sub[last]).astype(float),
        "sv_same_cat_last": (items.cat == items.cat[last]).astype(float),
        "sv_same_brand_last": (items.brand == items.brand[last]).astype(float),
        "sv_sub_share": share(items.sub), "sv_cat_share": share(items.cat), "sv_brand_share": share(items.brand),
        "sv_attr_share": (per_attr[i1] + per_attr[i2]) / k / 2,
        "sv_logp_diff": items.logp - items.logp[rev].mean(),
        "view_to_purchase_score": score["buy"], "coview_score": score["view"],
        "sv_secs_since_last": np.full(n, float(secs_since_last)),
        "sv_attr_last": in_last[i1] + in_last[i2],
        "sv_attr_w": (per_attr_w[i1] + per_attr_w[i2]) / w.sum() / 2,
        "sv_sub_w": share(items.sub, w), "sv_cat_w": share(items.cat, w),
        "sv_logp_diff_last": items.logp - items.logp[last],
        "sv_n_subs": np.full(n, float(len(np.unique(items.sub[rev])))),
    })
    return out


def parse_views(items: Items, recent_views, now) -> tuple[np.ndarray, float]:
    """recent_views de la petición → (índices de producto en orden, segundos desde la última vista)."""
    idx, last_ts = [], None
    for x in recent_views if recent_views is not None else ():
        if isinstance(x, dict) and x.get("product_id") in items.idx:
            idx.append(items.idx[x["product_id"]])
            last_ts = x.get("ts") or last_ts
    secs = np.nan
    if idx and last_ts is not None and now is not None:
        try:
            secs = (np.datetime64(now, "ms") - np.datetime64(str(last_ts)[:23].replace("T", " "), "ms")) / np.timedelta64(1, "s")
            secs = float(np.clip(secs, 0.0, 86400.0))
        except ValueError:
            secs = np.nan
    return np.asarray(idx, dtype=np.int64), secs


# ── usuario: base (descarga completa) + delta (lo ocurrido desde entonces) ──────────────────────────

def combine_user_scalars(base: dict[str, float] | None, delta: dict[str, float] | None) -> dict[str, float]:
    """Recuentos del usuario sumando las dos capas del estado de servicio.
    `base`: u_events, u_buys, u_w, u_r, u_logp, u_buy_logp, u_logp_sd (medias, como las da el SQL).
    `delta`: u_events, u_buys, u_w, u_r, s_logp, s_logp2, s_buy_logp (sumas, que sí se pueden sumar).
    Las medias y la desviación se rehacen a partir de las sumas, así que el resultado es exacto."""
    ev = buys = w = r = s = s2 = sb = 0.0
    if base is not None:
        n, m = float(base["u_events"]), float(np.nan_to_num(base["u_logp"]))
        sd = float(np.nan_to_num(base["u_logp_sd"]))
        ev, buys, w, r = n, float(base["u_buys"]), float(base["u_w"]), float(base["u_r"])
        s, s2 = m * n, (sd * sd + m * m) * n
        sb = float(np.nan_to_num(base["u_buy_logp"])) * buys
    if delta is not None:
        ev, buys = ev + float(delta["u_events"]), buys + float(delta["u_buys"])
        w, r = w + float(delta["u_w"]), r + float(delta["u_r"])
        s, s2, sb = s + float(delta["s_logp"]), s2 + float(delta["s_logp2"]), sb + float(delta["s_buy_logp"])
    mean = s / ev if ev > 0 else np.nan
    return {"u_events": ev, "u_buys": buys, "u_w": w, "u_r": r, "u_logp": mean,
            "u_buy_logp": sb / buys if buys > 0 else np.nan,
            "u_logp_sd": math.sqrt(max(s2 / ev - mean * mean, 0.0)) if ev > 0 else np.nan}


# ── usuario: de recuentos a features ─────────────────────────────────────────────────────────────────

def user_feats(a: dict[str, np.ndarray], logp: np.ndarray) -> dict[str, np.ndarray]:
    """`a`: los recuentos de USER_AGG para cada fila (candidato); `logp`: nivel de precio del candidato.
    Usuario sin historia: recuentos a 0 → cuotas 0 y diferencias de precio NaN."""
    g = lambda k: np.asarray(a[k], dtype=float)   # noqa: E731
    ev, buys, w, r = g("u_events"), g("u_buys"), g("u_w"), g("u_r")
    known = ev > 0
    div = lambda num, den: np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)   # noqa: E731
    w1, w2 = g("u_attr_w1"), g("u_attr_w2")
    u_logp = np.where(known, g("u_logp"), np.nan)
    return {
        "u_hist_conf": ev / (ev + HIST_CONF),
        "u_buy_ratio": div(buys, ev),
        "u_logp_diff": logp - u_logp,
        "u_buy_logp_diff": logp - np.where(buys > 0, g("u_buy_logp"), np.nan),
        "u_logp_z": (logp - u_logp) / np.maximum(np.nan_to_num(g("u_logp_sd")), 0.1),
        "u_sub_share": div(g("u_sub_w"), w), "u_brand_share": div(g("u_brand_w"), w),
        "u_cat_share": div(g("u_cat_w"), w),
        "u_attr_share": div(w1 + w2, 2 * w),
        "u_attr_max_share": div(np.maximum(w1, w2), w), "u_attr_min_share": div(np.minimum(w1, w2), w),
        "u_sub_share_rec": div(g("u_sub_r"), r), "u_cat_share_rec": div(g("u_cat_r"), r),
        "u_attr_share_rec": div(g("u_attr_r"), 2 * r),
        "u_sub_buy_share": div(g("u_sub_buys"), buys), "u_brand_buy_share": div(g("u_brand_buys"), buys),
        "u_cat_buy_share": div(g("u_cat_buys"), buys), "u_attr_buy_share": div(g("u_attr_buys"), 2 * buys),
    }
