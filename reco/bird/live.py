"""Features v2 al servir: el mismo significado que en build_dataset.py, pero para TODO el catálogo y en milisegundos.

Al entrenar, la historia de cada usuario se resume en SQL "hasta el instante de la petición". Al servir se
descarga ese mismo resumen "hasta ahora" para todos los usuarios (download), se guarda en Parquet y se carga
en memoria (LiveState). Cada petición solo hace búsquedas en arrays de numpy.

    download(until=None)          en vivo: historia + shop.events + nuestro feedback
    download(until="2026-11-07")  evaluación offline: solo lo anterior al corte
    LiveState.load(until).features(req, schema)  → DataFrame con una fila por producto del catálogo

Diferencias conocidas con el entrenamiento (documentadas, no hay más):
  - usuario × producto: solo se guardan los pares con carrito, clic, compra o ≥ 2 vistas (los de 1 vista son
    millones); para el resto up_views vale 0 en vez de 1.
  - la foto es de cuando se descargó: lo que el usuario haga después no se ve hasta el siguiente refresco
    (las vistas de la sesión sí, porque llegan en recent_views).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import features
from ..client import query
from ..config import DATA_DIR, TEAM
from ..features import _ATTR_SQL, ATTRS
from .build_dataset import BUY, DECAY, PRIOR_IMPRESSIONS

NEVER = np.datetime64("1971-01-01")
CHUNKED, CHUNKS = ("user_sub", "user_brand", "user_cat", "user_attr", "user_prod"), 6


def folder(until: str | None) -> Path:
    return DATA_DIR / "bird" / (f"state_until_{features.tag(until)}" if until else "state_live")


# ── SQL ──────────────────────────────────────────────────────────────────────────────────────────────

def _catalog_sql(until: str | None) -> str:
    """Catálogo con atributos. En vivo: última versión de cada producto (precio, productos nuevos)."""
    cols = ("product_id::String AS product_id, category::String AS r_cat, subcategory::String AS r_sub, "
            "brand::String AS r_brand, price_eur::Float64 AS r_price, description::String AS r_desc, "
            "toString(created_at::DateTime64(3)) AS r_created")
    if until:   # catalog_updates anteriores al corte (vacía antes del 14/11)
        inner = (f"SELECT {cols}, '' AS v FROM shop.catalog UNION ALL "
                 f"SELECT {cols}, toString(ts::DateTime64(3)) AS v FROM shop.catalog_updates WHERE ts < '{until}'")
    else:
        inner = (f"SELECT {cols}, '' AS v FROM shop.catalog UNION ALL "
                 f"SELECT {cols}, toString(ts::DateTime64(3)) AS v FROM shop.catalog_updates")
    return f"""(SELECT product_id, argMax(r_cat, v) AS category, argMax(r_sub, v) AS subcategory,
            argMax(r_brand, v) AS brand, argMax(r_price, v) AS price_eur, log(1 + argMax(r_price, v)) AS logp,
            min(r_created) AS created_at,
            arrayFilter(a -> position(argMax(r_desc, v), a) > 0, {_ATTR_SQL}) AS attrs
        FROM ({inner}) GROUP BY product_id)"""


def _events_sql(until: str | None) -> str:
    """Mismos eventos y pesos que build_dataset.user_item_sql (vista 1, carrito 3, compra 5, clic de carrusel 1)."""
    ev = ("user_id::String AS user_id, product_id::String AS product_id, ts::DateTime64(3) AS ts, "
          "event_type::String AS kind, multiIf(event_type::String = 'purchase', 5, "
          "event_type::String = 'add_to_cart', 3, 1) AS w")
    ca = ("user_id::String AS user_id, clicked::String AS product_id, ts::DateTime64(3) AS ts, "
          "if(purchased::String != '', 'c_purchase', 'c_click') AS kind, if(purchased::String != '', 5, 1) AS w")
    if until:   # todo lo anterior al corte; shop.events y feedback están vacíos antes del 14/11
        return (f"(SELECT {ev} FROM shop.history_events WHERE ts < '{until}' UNION ALL "
                f"SELECT {ev} FROM shop.events WHERE ts < '{until}' UNION ALL "
                f"SELECT {ca} FROM shop.history_sessions WHERE clicked::String != '' AND ts < '{until}' UNION ALL "
                f"SELECT r.user_id::String, f.clicked::String, f.ts::DateTime64(3), "
                f"if(f.purchased::String != '', 'c_purchase', 'c_click'), if(f.purchased::String != '', 5, 1) "
                f"FROM {TEAM}.feedback AS f INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String "
                f"WHERE f.clicked::String != '' AND f.request_id::String != '__init__' "
                f"AND f.ts::DateTime64(3) < '{until}')")
    return (f"(SELECT {ev} FROM shop.history_events UNION ALL SELECT {ev} FROM shop.events UNION ALL "
            f"SELECT {ca} FROM shop.history_sessions WHERE clicked::String != '' UNION ALL "
            f"SELECT r.user_id::String, f.clicked::String, f.ts::DateTime64(3), "
            f"if(f.purchased::String != '', 'c_purchase', 'c_click'), if(f.purchased::String != '', 5, 1) "
            f"FROM {TEAM}.feedback AS f INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String "
            f"WHERE f.clicked::String != '' AND f.request_id::String != '__init__')")


def queries(until: str | None) -> dict[str, str]:
    now = f"toDateTime64('{until}', 3)" if until else "now64(3)"
    E, C = _events_sql(until), _catalog_sql(until)
    joined = f"{E} AS e INNER JOIN {C} AS c ON c.product_id = e.product_id"
    organic = "e.kind IN ('view', 'add_to_cart', 'purchase')"
    cut = f" AND ts < '{until}'" if until else ""
    return {
        "catalog": f"SELECT product_id, category, subcategory, brand, price_eur, logp, created_at, attrs FROM {C}",
        "user": f"""SELECT e.user_id AS user_id, count() AS u_events, sum(e.w) AS u_w,
                countIf(e.kind IN {BUY}) AS u_buys, avg(c.logp) AS u_logp,
                avgIf(c.logp, e.kind IN {BUY}) AS u_buy_logp, toString(max(e.ts)) AS u_last
            FROM {joined} GROUP BY user_id""",
        "user_sub": f"""SELECT e.user_id AS user_id, c.subcategory AS k, sum(e.w) AS w,
                sumIf(e.w, e.ts >= {now} - INTERVAL 7 DAY) AS w7, countIf(e.kind IN {BUY}) AS buys
            FROM {joined} GROUP BY user_id, k""",
        "user_brand": f"""SELECT e.user_id AS user_id, c.brand AS k, sum(e.w) AS w, countIf(e.kind IN {BUY}) AS buys
            FROM {joined} GROUP BY user_id, k""",
        "user_cat": f"SELECT e.user_id AS user_id, c.category AS k, sum(e.w) AS w FROM {joined} GROUP BY user_id, k",
        "user_attr": f"""SELECT e.user_id AS user_id, arrayJoin(c.attrs) AS k, sum(e.w) AS w
            FROM {joined} GROUP BY user_id, k""",
        "user_prod": f"""SELECT e.user_id AS user_id, e.product_id AS k, countIf(e.kind = 'view') AS views,
                countIf(e.kind = 'add_to_cart') AS carts, countIf(e.kind = 'c_click') AS c_clicks,
                countIf(e.kind IN {BUY}) AS buys, toString(max(e.ts)) AS last
            FROM {joined} GROUP BY user_id, k HAVING carts > 0 OR c_clicks > 0 OR buys > 0 OR views >= 2""",
        "item": f"""SELECT e.product_id AS product_id, countIf(e.kind = 'view') AS views,
                countIf(e.kind = 'add_to_cart') AS carts, countIf(e.kind = 'purchase') AS buys,
                countIf(e.kind = 'view' AND e.ts >= {now} - INTERVAL 7 DAY) AS views_7d,
                countIf(e.kind = 'purchase' AND e.ts >= {now} - INTERVAL 7 DAY) AS buys_7d,
                countIf(e.kind = 'purchase' AND e.ts >= {now} - INTERVAL 1 DAY) AS buys_1d
            FROM {E} AS e WHERE {organic} GROUP BY product_id""",
        "item_shown": f"""SELECT arrayJoin(shown::Array(String)) AS product_id, count() AS c_impr
            FROM shop.history_sessions WHERE 1{cut} GROUP BY product_id""",
        "item_clicked": f"""SELECT clicked::String AS product_id, count() AS c_clicks,
                countIf(purchased::String != '') AS c_buys
            FROM shop.history_sessions WHERE clicked::String != ''{cut} GROUP BY product_id""",
    }


def download(until: str | None = None, verbose: bool = True) -> Path:
    """Descarga las tablas y las deja en Parquet. Escribe en una carpeta temporal y la cambia al final."""
    out = folder(until)
    tmp = out.with_name(out.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    for name, sql in queries(until).items():
        t = time.time()
        if name in CHUNKED:   # RawTree devuelve como mucho 100 MiB por consulta: se parte por hash de usuario
            df = pd.concat([pd.DataFrame(query(sql.replace(
                "GROUP BY user_id", f"WHERE cityHash64(e.user_id) % {CHUNKS} = {k} GROUP BY user_id"), timeout=900))
                for k in range(CHUNKS)], ignore_index=True)
        else:
            df = pd.DataFrame(query(sql, timeout=900))
        df.to_parquet(tmp / f"{name}.parquet", index=False)
        if verbose:
            print(f"  {name:<13} {len(df):>9,} filas  {time.time() - t:.1f}s")
    out.mkdir(parents=True, exist_ok=True)
    for f in tmp.glob("*.parquet"):
        f.replace(out / f.name)
    tmp.rmdir()
    return out


# ── estado en memoria ────────────────────────────────────────────────────────────────────────────────

class _ByUser:
    """Tabla (usuario, clave, valores) ordenada por usuario: las filas de un usuario son un trozo contiguo."""

    def __init__(self, df: pd.DataFrame, ucode: dict[str, int], kcode: dict[str, int], cols: list[str]) -> None:
        u = df.user_id.map(ucode).to_numpy()
        k = df.k.map(kcode).to_numpy()
        ok = ~(np.isnan(u) | np.isnan(k))
        u, k = u[ok].astype(np.int64), k[ok].astype(np.int64)
        order = np.argsort(u, kind="stable")
        self.k = k[order]
        self.v = {c: df[c].to_numpy()[ok][order] for c in cols}
        self.ptr = np.searchsorted(u[order], np.arange(len(ucode) + 1))

    def rows(self, u: int) -> slice:
        return slice(self.ptr[u], self.ptr[u + 1])


@dataclass
class LiveState:
    ids: np.ndarray
    until: str | None

    @classmethod
    def load(cls, until: str | None = None) -> LiveState:
        src = folder(until)
        if not (src / "catalog.parquet").exists():
            download(until)
        read = lambda name: pd.read_parquet(src / f"{name}.parquet")   # noqa: E731
        cat = read("catalog").sort_values("product_id").reset_index(drop=True)
        self = cls(ids=cat.product_id.to_numpy(), until=until)
        self.idx = {p: i for i, p in enumerate(self.ids)}
        n = len(cat)
        order = {a: i for i, a in enumerate(ATTRS)}
        attrs = cat["attrs"].apply(lambda v: sorted(list(v) if v is not None else [], key=order.get))
        cat["attr1"] = attrs.str[0].fillna("UNKNOWN")
        cat["attr2"] = attrs.str[1].fillna("UNKNOWN")
        self.attr_code = {a: i for i, a in enumerate(ATTRS)}
        self.a1 = cat.attr1.map(self.attr_code).fillna(-1).to_numpy(dtype=int)
        self.a2 = cat.attr2.map(self.attr_code).fillna(-2).to_numpy(dtype=int)
        codes = {}
        for col in ("subcategory", "brand", "category"):
            c = cat[col].astype("category")
            codes[col] = {k: i for i, k in enumerate(c.cat.categories)}
            setattr(self, {"subcategory": "sub", "brand": "brand", "category": "cat"}[col], c.cat.codes.to_numpy())
        self.logp = cat.logp.to_numpy(dtype=float)
        self.created = pd.to_datetime(cat.created_at).to_numpy()

        # producto: mismas fórmulas que build_dataset.item_tables
        it = cat[["product_id"]].merge(read("item"), how="left").merge(read("item_shown"), how="left") \
            .merge(read("item_clicked"), how="left").fillna(0)
        g = lambda c: it[c].to_numpy(dtype=float)   # noqa: E731
        views, carts, buys, impr = g("views"), g("carts"), g("buys"), g("c_impr")
        sub_rank = it.assign(s=cat.subcategory, b=-g("buys_7d")).groupby("s").b.rank(method="first").to_numpy()
        g_ctr, g_cvr = g("c_clicks").sum() / max(impr.sum(), 1), g("c_buys").sum() / max(impr.sum(), 1)
        self.static = pd.DataFrame({
            "category": cat.category, "subcategory": cat.subcategory, "brand": cat.brand,
            "attr1": cat.attr1, "attr2": cat.attr2,
            "price_eur": cat.price_eur.astype(float),
            "it_price_vs_sub": self.logp - cat.groupby("subcategory").logp.transform("median").to_numpy(),
            "it_buy_share": buys / max(buys.sum(), 1) * 1e4,
            "it_view_share": views / max(views.sum(), 1) * 1e4,
            "it_buy_rate": (buys + 1) / (views + 50),
            "it_cart_rate": (carts + 1) / (views + 50),
            "it_buys_7d": g("buys_7d"), "it_views_7d": g("views_7d"), "it_buys_1d": g("buys_1d"),
            "it_trend": np.log1p(g("buys_1d") * 7) - np.log1p(g("buys_7d")),
            "it_sub_rank": sub_rank,
            "it_c_ctr": (g("c_clicks") + PRIOR_IMPRESSIONS * g_ctr) / (impr + PRIOR_IMPRESSIONS),
            "it_c_cvr": (g("c_buys") + PRIOR_IMPRESSIONS * g_cvr) / (impr + PRIOR_IMPRESSIONS),
        })
        self.static_bytes = {}
        for col in ("category", "subcategory", "brand", "attr1", "attr2"):
            self.static[col] = self.static[col].astype(object)
            self.static_bytes[col] = np.array([str(x).encode() for x in self.static[col]], dtype=object)

        # usuario
        user = read("user")
        self.ucode = {u: i for i, u in enumerate(user.user_id)}
        self.u_events = user.u_events.to_numpy(dtype=float)
        self.u_w = user.u_w.to_numpy(dtype=float)
        self.u_buys = user.u_buys.to_numpy(dtype=float)
        self.u_logp = user.u_logp.to_numpy(dtype=float)
        self.u_buy_logp = user.u_buy_logp.to_numpy(dtype=float)
        self.u_last = pd.to_datetime(user.u_last).to_numpy()
        self.by_sub = _ByUser(read("user_sub"), self.ucode, codes["subcategory"], ["w", "w7", "buys"])
        self.by_brand = _ByUser(read("user_brand"), self.ucode, codes["brand"], ["w", "buys"])
        self.by_cat = _ByUser(read("user_cat"), self.ucode, codes["category"], ["w"])
        self.by_attr = _ByUser(read("user_attr"), self.ucode, self.attr_code, ["w"])
        up = read("user_prod")
        up["last"] = pd.to_datetime(up["last"])
        self.by_prod = _ByUser(up, self.ucode, self.idx, ["views", "carts", "c_clicks", "buys", "last"])
        self.sizes = (len(codes["subcategory"]), len(codes["brand"]), len(codes["category"]))

        # vecinos (co-vista / vista→compra): las mismas tablas que usa el resto del sistema
        self.nb: dict[str, dict[int, list[tuple[int, float]]]] = {"buy": {}, "view": {}}
        nb_path = features.folder(until) / "item_neighbors.parquet"
        if nb_path.exists():
            nb = pd.read_parquet(nb_path)
            for kind in ("buy", "view"):
                d = nb[nb.kind == kind]
                d = d.assign(nn=d.n / d.groupby("p").n.transform("max")).sort_values(["p", "nn"],
                                                                                     ascending=[True, False])
                self.nb[kind] = {self.idx[p]: [(self.idx[q], v) for q, v in zip(g_.q, g_.nn) if q in self.idx][:20]
                                 for p, g_ in d.groupby("p", sort=False) if p in self.idx}
        self.n = n
        return self

    # ── por petición ─────────────────────────────────────────────────────────────────────────────────

    def user(self, req) -> int | None:
        """Código del usuario, o None si es nuevo o no tiene historia (cold start)."""
        if req.get("is_new_user"):
            return None
        return self.ucode.get(req.get("user_id"))

    def bought(self, u: int | None) -> np.ndarray:
        """Índices de los productos que el usuario ya compró (para no volver a ofrecérselos)."""
        if u is None:
            return np.zeros(0, dtype=int)
        r = self.by_prod.rows(u)
        return self.by_prod.k[r][self.by_prod.v["buys"][r] > 0]

    def features(self, req, schema) -> pd.DataFrame:
        """Una fila por producto del catálogo (en el orden de self.ids), con las features del esquema."""
        out = self._columns(req, schema)
        return pd.DataFrame({f: out[f] for f in schema.features})

    def matrices(self, req, schema, rows: np.ndarray, out: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Lo mismo que features() para las filas `rows`, ya en el formato rápido de CatBoost:
        (numéricas float32 [n, k], categóricas en bytes [n, c]), en el orden de schema.numeric / schema.categorical.
        Evita pasar por pandas, que era el 75 % del tiempo de cada petición."""
        out = self._columns(req, schema) if out is None else out   # `out`: columnas ya calculadas
        num = np.empty((len(rows), len(schema.numeric)), dtype=np.float32)
        for j, f in enumerate(schema.numeric):
            num[:, j] = np.asarray(out[f], dtype=float)[rows]
        cat = np.empty((len(rows), len(schema.categorical)), dtype=object)
        for j, f in enumerate(schema.categorical):
            col = self.static_bytes.get(f)
            cat[:, j] = col[rows] if col is not None else str(out[f][0]).encode()
        return num, cat

    def _columns(self, req, schema) -> dict[str, object]:
        n = self.n
        now = np.datetime64(pd.Timestamp(str(req.get("ts"))[:23]) if req.get("ts") else pd.Timestamp.now("UTC")
                            .tz_localize(None))
        out: dict[str, object] = {c: self.static[c].to_numpy() for c in self.static.columns}
        out["it_age_days"] = (now - self.created) / np.timedelta64(1, "D")
        ctx = req.get("context") or {}
        for key in ("page", "device", "region"):
            out[key] = np.full(n, ctx.get(key) or "UNKNOWN", dtype=object)
        out["position"] = np.full(n, float(schema.reference_position))

        # sesión
        v = [self.idx[x["product_id"]] for x in (req.get("recent_views") or [])
             if isinstance(x, dict) and x.get("product_id") in self.idx]
        z = np.zeros(n)
        out["n_recent_views"] = np.full(n, float(len(v)))
        names = ("recently_viewed", "sv_recency", "sv_count", "sv_same_sub_last", "sv_same_cat_last",
                 "sv_same_brand_last", "sv_sub_share", "sv_cat_share", "sv_brand_share", "sv_attr_share",
                 "view_to_purchase_score", "coview_score")
        for name in names:
            out[name] = z
        out["sv_logp_diff"] = np.full(n, np.nan)
        if v:
            va = np.array(v)
            rev = va[::-1]
            count, recency = np.zeros(n), np.zeros(n)
            for i, x in enumerate(rev):
                count[x] += 1
                if recency[x] == 0:
                    recency[x] = 1 / (i + 1)
            last = rev[0]
            share = lambda code: np.bincount(code[va], minlength=code.max() + 1)[code] / len(va)   # noqa: E731
            attr_n = np.bincount(np.concatenate([self.a1[va], self.a2[va]]).clip(0), minlength=len(ATTRS) + 1)
            a1v, a2v = self.a1[va], self.a2[va]
            both = np.bincount(a1v[a1v >= 0], minlength=len(ATTRS)), np.bincount(a2v[a2v >= 0], minlength=len(ATTRS))
            per_attr = np.zeros(len(ATTRS) + 1)
            # nº de vistas que tienen el atributo a (en cualquiera de sus dos huecos)
            for a in range(len(ATTRS)):
                per_attr[a] = ((a1v == a) | (a2v == a)).sum()
            del attr_n, both
            same = np.where(self.a1 >= 0, per_attr[self.a1.clip(0)], 0) + np.where(self.a2 >= 0,
                                                                                  per_attr[self.a2.clip(0)], 0)
            score = {}
            for kind in ("buy", "view"):
                s = np.zeros(n)
                for i, x in enumerate(rev):
                    for q, nn in self.nb[kind].get(int(x), ()):
                        s[q] += DECAY ** i * nn
                score[kind] = s
            out.update({
                "recently_viewed": (count > 0).astype(float), "sv_recency": recency, "sv_count": count,
                "sv_same_sub_last": (self.sub == self.sub[last]).astype(float),
                "sv_same_cat_last": (self.cat == self.cat[last]).astype(float),
                "sv_same_brand_last": (self.brand == self.brand[last]).astype(float),
                "sv_sub_share": share(self.sub), "sv_cat_share": share(self.cat), "sv_brand_share": share(self.brand),
                "sv_attr_share": same / len(va) / 2,
                "sv_logp_diff": self.logp - self.logp[va].mean(),
                "view_to_purchase_score": score["buy"], "coview_score": score["view"],
            })

        # usuario
        u = self.user(req)
        nan = np.full(n, np.nan)
        user_cols = ("u_events", "u_buys", "u_sub_share", "u_brand_share", "u_cat_share", "u_attr_share", "u_sub_w",
                     "u_brand_w", "u_sub_w_7d", "u_sub_buys", "u_brand_buys", "up_views", "up_carts", "up_c_clicks",
                     "up_buys", "up_c_buys", "previous_purchase_count", "already_purchased")
        for name in user_cols:
            out[name] = z
        for name in ("u_days_since_last", "u_logp_diff", "u_buy_logp_diff", "up_days_since_seen",
                     "days_since_last_purchase"):
            out[name] = nan
        out["is_new_user"] = np.full(n, 1.0 if u is None else 0.0)
        if u is not None:
            n_sub, n_brand, n_cat = self.sizes
            w = max(self.u_w[u], 1e-9)

            def spread(table: _ByUser, col: str, size: int) -> np.ndarray:
                r = table.rows(u)
                vec = np.zeros(size)
                vec[table.k[r]] = table.v[col][r]
                return vec

            sub_w, brand_w = spread(self.by_sub, "w", n_sub)[self.sub], spread(self.by_brand, "w", n_brand)[self.brand]
            attr = np.append(spread(self.by_attr, "w", len(ATTRS)), 0.0)
            attr_w = np.where(self.a1 >= 0, attr[self.a1.clip(0)], 0) + np.where(self.a2 >= 0, attr[self.a2.clip(0)], 0)
            r = self.by_prod.rows(u)
            last_seen = np.full(n, NEVER, dtype="datetime64[ns]")
            last_seen[self.by_prod.k[r]] = self.by_prod.v["last"][r]
            seen_days = (now - last_seen) / np.timedelta64(1, "D")
            buys = spread(self.by_prod, "buys", n)
            out.update({
                "u_events": np.full(n, self.u_events[u]), "u_buys": np.full(n, self.u_buys[u]),
                "u_days_since_last": np.full(n, (now - self.u_last[u]) / np.timedelta64(1, "D")),
                "u_logp_diff": self.logp - self.u_logp[u],
                "u_buy_logp_diff": self.logp - self.u_buy_logp[u] if self.u_buys[u] > 0 else nan,
                "u_sub_w": sub_w, "u_brand_w": brand_w,
                "u_sub_share": sub_w / w, "u_brand_share": brand_w / w,
                "u_cat_share": spread(self.by_cat, "w", n_cat)[self.cat] / w,
                "u_attr_share": attr_w / (2 * w),
                "u_sub_w_7d": spread(self.by_sub, "w7", n_sub)[self.sub],
                "u_sub_buys": spread(self.by_sub, "buys", n_sub)[self.sub],
                "u_brand_buys": spread(self.by_brand, "buys", n_brand)[self.brand],
                "up_views": spread(self.by_prod, "views", n), "up_carts": spread(self.by_prod, "carts", n),
                "up_c_clicks": spread(self.by_prod, "c_clicks", n),
                "up_days_since_seen": np.where(last_seen > NEVER, seen_days, np.nan),
                "previous_purchase_count": buys, "already_purchased": (buys > 0).astype(float),
            })
        missing = [f for f in schema.features if f not in out]
        if missing:
            raise NotImplementedError(f"LiveState no calcula {missing} (esquema {schema.version})")
        return out
