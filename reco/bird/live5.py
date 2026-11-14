"""Features v5 al servir: los mismos recuentos que build_v5.py, pero "hasta ahora" y para TODO el catálogo.

    download(until=None)                 en vivo: historia + shop.events + catalog_updates + nuestro feedback
    download(until="2026-11-07")         evaluación offline: solo lo anterior al corte
    LiveState5.load(until).matrices(req, schema, filas)   → matrices listas para CatBoost
    estado.with_items()                  refresco ligero (segundos): catálogo, promociones, tendencias, feedback

Las FÓRMULAS son las de reco/bird/feat5.py, las mismas funciones que usa el entrenamiento; aquí solo se
consiguen los recuentos. `parity_report()` lo comprueba: features al servir frente a las del dataset.

Diferencias conocidas con el entrenamiento (no hay más):
  - usuario × producto: solo se guardan los pares con carrito, clic, compra o ≥ 2 vistas (como en v2).
  - la historia del usuario es de cuando se descargó el estado; lo posterior no se ve hasta el siguiente
    refresco (las vistas de la sesión sí: llegan en recent_views).
  - ventanas de producto: al servir acaban en el instante del refresco; al entrenar, en el último tramo
    completo (1 h en el histórico, 10 min en vivo).
A diferencia de v2, un usuario marcado como nuevo que ya tiene eventos de hoy SÍ usa su historia.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import features
from ..client import query
from ..config import DATA_DIR, TEAM
from . import feat5
from .build_dataset import BUY
from .build_v5 import catalog_sql, events_sql, neighbor_dict, position_curve

NEVER = np.datetime64("1971-01-01")
CHUNKED, CHUNKS = ("user_sub", "user_brand", "user_cat", "user_attr", "user_prod"), 6
ITEM_TABLES = ("catalog", "item", "item_feedback")     # las que cambian cada minuto (refresco ligero)


def folder(until: str | None) -> Path:
    return DATA_DIR / "bird" / (f"state5_until_{features.tag(until)}" if until else "state5_live")


def queries(until: str | None) -> dict[str, str]:
    now = f"toDateTime64('{until}', 3)" if until else "now64(3)"
    E, C = events_sql(True, "1", until), catalog_sql(until)
    joined = f"{E} AS e INNER JOIN {C} AS c ON c.product_id = e.product_id"
    decay = f"exp(-dateDiff('millisecond', e.ts, {now}) / {feat5.TAU_MS:.1f})"
    cut = f" AND ts::DateTime64(3) < '{until}'" if until else ""
    organic = (f"(SELECT product_id::String AS product_id, event_type::String AS kind, ts::DateTime64(3) AS ts "
               f"FROM shop.history_events WHERE 1{cut} UNION ALL "
               f"SELECT product_id::String, event_type::String, ts::DateTime64(3) FROM shop.events WHERE 1{cut})")
    ctx = lambda key: (   # noqa: E731
        f"SELECT k, p, count() AS n FROM (SELECT {key}::String AS k, product_id::String AS p FROM shop.history_events "
        f"WHERE event_type::String = 'purchase'{cut} UNION ALL SELECT {key}::String, product_id::String "
        f"FROM shop.events WHERE event_type::String = 'purchase'{cut}) GROUP BY k, p")
    return {
        "catalog": f"SELECT product_id, category, subcategory, brand, base_price, price_eur, created_at, attrs FROM {C}",
        "item": f"""SELECT product_id, countIf(kind = 'view') AS views, countIf(kind = 'add_to_cart') AS carts,
                countIf(kind = 'purchase') AS buys,
                countIf(kind = 'view' AND ts >= {now} - INTERVAL 7 DAY) AS views_7d,
                countIf(kind = 'purchase' AND ts >= {now} - INTERVAL 7 DAY) AS buys_7d,
                countIf(kind = 'view' AND ts >= {now} - INTERVAL 1 DAY) AS views_1d,
                countIf(kind = 'purchase' AND ts >= {now} - INTERVAL 1 DAY) AS buys_1d,
                countIf(kind = 'view' AND ts >= {now} - INTERVAL 1 HOUR) AS views_1h
            FROM {organic} GROUP BY product_id""",
        "item_feedback": f"""SELECT x.1 AS p, x.2 AS slot, count() AS impr, countIf(clicked::String = x.1) AS clicks,
                countIf(purchased::String = x.1) AS buys
            FROM (SELECT arrayJoin(arrayZip(shown::Array(String), arrayEnumerate(shown::Array(String)))) AS x,
                         clicked, purchased FROM {TEAM}.feedback WHERE request_id::String != '__init__'{cut})
            GROUP BY p, slot""",
        "item_shown": f"""SELECT arrayJoin(shown::Array(String)) AS product_id, count() AS c_impr
            FROM shop.history_sessions WHERE 1{cut} GROUP BY product_id""",
        "item_clicked": f"""SELECT clicked::String AS product_id, count() AS c_clicks,
                countIf(purchased::String != '') AS c_buys
            FROM shop.history_sessions WHERE clicked::String != ''{cut} GROUP BY product_id""",
        "ctx_device": ctx("device"),
        "ctx_region": ctx("region"),
        "item_neighbors": features.queries(until)["item_neighbors"],
        "user": f"""SELECT e.user_id AS user_id, count() AS u_events, countIf(e.kind IN {BUY}) AS u_buys,
                sum(e.w) AS u_w, sum(e.w * {decay}) AS u_r, avg(c.logp) AS u_logp,
                avgIf(c.logp, e.kind IN {BUY}) AS u_buy_logp, stddevPop(c.logp) AS u_logp_sd
            FROM {joined} GROUP BY user_id""",
        "user_sub": f"""SELECT e.user_id AS user_id, c.subcategory AS k, sum(e.w) AS w, sum(e.w * {decay}) AS r,
                countIf(e.kind IN {BUY}) AS buys FROM {joined} GROUP BY user_id, k""",
        "user_brand": f"""SELECT e.user_id AS user_id, c.brand AS k, sum(e.w) AS w, countIf(e.kind IN {BUY}) AS buys
            FROM {joined} GROUP BY user_id, k""",
        "user_cat": f"""SELECT e.user_id AS user_id, c.category AS k, sum(e.w) AS w, sum(e.w * {decay}) AS r,
                countIf(e.kind IN {BUY}) AS buys FROM {joined} GROUP BY user_id, k""",
        "user_attr": f"""SELECT e.user_id AS user_id, arrayJoin(c.attrs) AS k, sum(e.w) AS w,
                sum(e.w * {decay}) AS r, countIf(e.kind IN {BUY}) AS buys FROM {joined} GROUP BY user_id, k""",
        "user_prod": f"""SELECT e.user_id AS user_id, e.product_id AS k, countIf(e.kind = 'view') AS views,
                countIf(e.kind = 'add_to_cart') AS carts, countIf(e.kind = 'c_click') AS c_clicks,
                countIf(e.kind IN {BUY}) AS buys, toString(max(e.ts)) AS last
            FROM {joined} GROUP BY user_id, k HAVING carts > 0 OR c_clicks > 0 OR buys > 0 OR views >= 2""",
    }


def _fetch(name: str, sql: str) -> pd.DataFrame:
    if name in CHUNKED:   # RawTree devuelve como mucho 100 MiB por consulta: se parte por hash de usuario
        return pd.concat([pd.DataFrame(query(sql.replace(
            "GROUP BY user_id", f"WHERE cityHash64(e.user_id) % {CHUNKS} = {k} GROUP BY user_id"), timeout=900))
            for k in range(CHUNKS)], ignore_index=True)
    return pd.DataFrame(query(sql, timeout=900))


def download(until: str | None = None, verbose: bool = True, only: tuple[str, ...] | None = None) -> Path:
    """Descarga las tablas y las deja en Parquet. Escribe en una carpeta temporal y la cambia al final."""
    out = folder(until)
    tmp = out.with_name(out.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    for name, sql in queries(until).items():
        if only and name not in only:
            continue
        t = time.time()
        df = _fetch(name, sql)
        if name == "catalog":
            df["attrs"] = df["attrs"].apply(lambda v: list(v) if v is not None else [])
        df.to_parquet(tmp / f"{name}.parquet", index=False)
        if verbose:
            print(f"  {name:<14} {len(df):>9,} filas  {time.time() - t:.1f}s")
    out.mkdir(parents=True, exist_ok=True)
    for f in tmp.glob("*.parquet"):
        f.replace(out / f.name)
    tmp.rmdir()
    return out


# ── estado en memoria ────────────────────────────────────────────────────────────────────────────────

class _ByUser:
    """Tabla (usuario, clave, valores) ordenada por usuario: las filas de un usuario son un trozo contiguo."""

    def __init__(self, df: pd.DataFrame, ucode: dict[str, int], kcode: dict[str, int], cols: list[str]) -> None:
        u = df.user_id.map(ucode).to_numpy(dtype=float)
        k = df.k.map(kcode).to_numpy(dtype=float)
        ok = ~(np.isnan(u) | np.isnan(k))
        u, k = u[ok].astype(np.int64), k[ok].astype(np.int64)
        order = np.argsort(u, kind="stable")
        self.k = k[order]
        self.v = {c: df[c].to_numpy()[ok][order] for c in cols}
        self.ptr = np.searchsorted(u[order], np.arange(len(ucode) + 1))

    def rows(self, u: int) -> slice:
        return slice(self.ptr[u], self.ptr[u + 1])

    def remap(self, new_k: np.ndarray) -> _ByUser:
        """La misma tabla con las claves renumeradas (cuando entran productos nuevos en el catálogo)."""
        other = object.__new__(_ByUser)
        other.k, other.v, other.ptr = new_k[self.k], self.v, self.ptr
        return other


class LiveState5:
    until: str | None

    @classmethod
    def load(cls, until: str | None = None) -> LiveState5:
        src = folder(until)
        if not (src / "user_prod.parquet").exists():
            download(until)
        read = lambda name: pd.read_parquet(src / f"{name}.parquet")   # noqa: E731
        self = cls()
        self.until = until
        self._static_ctx = {name: read(name) for name in ("item_shown", "item_clicked", "ctx_device", "ctx_region")}
        self._neighbors = read("item_neighbors")
        self._set_items(read("catalog"), read("item"), read("item_feedback"))

        # usuario
        user = read("user")
        self.ucode = {u: i for i, u in enumerate(user.user_id)}
        self.user_cols = {c: user[c].to_numpy(dtype=float) for c in ("u_events", "u_buys", "u_w", "u_r", "u_logp",
                                                                     "u_buy_logp", "u_logp_sd")}
        it = self.items
        self.by_sub = _ByUser(read("user_sub"), self.ucode, it.codes["sub"], ["w", "r", "buys"])
        self.by_brand = _ByUser(read("user_brand"), self.ucode, it.codes["brand"], ["w", "buys"])
        self.by_cat = _ByUser(read("user_cat"), self.ucode, it.codes["cat"], ["w", "r", "buys"])
        self.by_attr = _ByUser(read("user_attr"), self.ucode, feat5.ATTR_CODE, ["w", "r", "buys"])
        up = read("user_prod")
        up["last"] = pd.to_datetime(up["last"]).to_numpy().astype("datetime64[ns]")
        self.by_prod = _ByUser(up, self.ucode, it.idx, ["views", "carts", "c_clicks", "buys", "last"])
        return self

    def _set_items(self, cat: pd.DataFrame, item: pd.DataFrame, fb: pd.DataFrame) -> None:
        """Todo lo que depende solo del producto: catálogo, promoción, popularidad, carrusel, contexto."""
        cat = cat.sort_values("product_id").reset_index(drop=True)
        old_ids = getattr(self, "ids", None)
        it = self.items = feat5.Items(cat.assign(created_at=pd.to_datetime(cat.created_at)))
        self.ids, self.idx, self.n = it.ids, it.idx, it.n
        same_catalog = old_ids is not None and len(old_ids) == it.n and bool((old_ids == it.ids).all())
        n = it.n

        def per_item(df: pd.DataFrame, key: str, cols: list[str]) -> dict[str, np.ndarray]:
            out = {c: np.zeros((1, n)) for c in cols}
            if len(df):
                i = df[key].map(it.idx).to_numpy(dtype=float)
                ok = ~np.isnan(i)
                for c in cols:
                    np.add.at(out[c][0], i[ok].astype(int), df[c].to_numpy(dtype=float)[ok])
            return out

        counts = per_item(item, "product_id", ["views", "carts", "buys", "views_7d", "buys_7d", "views_1d",
                                               "buys_1d", "views_1h"])
        static = {k: v[0] for k, v in feat5.item_pop(counts, it.sub).items()}
        self.legacy = {"it_buys_7d": counts["buys_7d"][0], "it_views_7d": counts["views_7d"][0],
                       "it_buys_1d": counts["buys_1d"][0]}
        shown = per_item(self._static_ctx["item_shown"], "product_id", ["c_impr"])
        clicked = per_item(self._static_ctx["item_clicked"], "product_id", ["c_clicks", "c_buys"])
        static.update({k: v[0] for k, v in feat5.carousel_rates(shown["c_impr"], clicked["c_clicks"],
                                                                 clicked["c_buys"]).items()})
        # nuestro carrusel: clics y compras reales frente a los esperados por posición
        if len(fb):
            ctr, cvr = position_curve(fb.rename(columns={"p": "p"}))
            fb = fb[fb.p.isin(it.idx) & fb.slot.between(1, 10)]
            slot = fb.slot.to_numpy(dtype=int)
            fb = fb.assign(exp_clicks=fb.impr.to_numpy(dtype=float) * ctr[slot],
                           exp_buys=fb.impr.to_numpy(dtype=float) * cvr[slot])
            f = per_item(fb, "p", ["clicks", "buys", "exp_clicks", "exp_buys"])
        else:
            f = {c: np.zeros((1, n)) for c in ("clicks", "buys", "exp_clicks", "exp_buys")}
        static.update(feat5.feedback_lift(f["clicks"][0], f["exp_clicks"][0], f["buys"][0], f["exp_buys"][0]))
        price = cat.price_eur.to_numpy(dtype=float)
        static.update({"price_eur": price, "it_price_vs_sub": it.price_vs_sub,
                       "it_discount": np.clip(1.0 - price / it.base_price, 0.0, 1.0)})
        inv = {name: np.array(list(code), dtype=object) for name, code in it.codes.items()}
        cats = {"category": inv["cat"][it.cat], "subcategory": inv["sub"][it.sub], "brand": inv["brand"][it.brand],
                "attr1": it.attr1, "attr2": it.attr2}
        self.static = {**cats, **static}
        self.static_bytes = {c: np.array([str(x).encode() for x in v], dtype=object) for c, v in cats.items()}

        # contexto: log-lift de cada producto para cada dispositivo y cada región
        self.ctx = {}
        for key, prefix, levels in (("device", "dev", feat5.DEV_LEVELS), ("region", "reg", feat5.REG_LEVELS)):
            df = self._static_ctx[f"ctx_{key}"]
            df = df[df.p.isin(it.idx)]
            code = {k: i for i, k in enumerate(sorted(df.k.unique()))}
            m = np.zeros((max(len(code), 1), n))
            if len(df):
                np.add.at(m, (df.k.map(code).to_numpy(), df.p.map(it.idx).to_numpy()), df.n.to_numpy(dtype=float))
            self.ctx[key] = (code, feat5.context_lifts(m, it, prefix, levels))
        if not same_catalog:      # los vecinos van por índice de producto: solo cambian si cambia el catálogo
            self.nb = neighbor_dict(self._neighbors, it)

    def with_items(self, cat: pd.DataFrame | None = None, item: pd.DataFrame | None = None,
                   fb: pd.DataFrame | None = None) -> LiveState5:
        """Copia con el catálogo, las promociones, la popularidad y el feedback de AHORA (3 consultas de
        segundos); la historia de los usuarios se comparte con el estado anterior."""
        q = queries(None)
        cat = _fetch("catalog", q["catalog"]) if cat is None else cat
        item = _fetch("item", q["item"]) if item is None else item
        fb = _fetch("item_feedback", q["item_feedback"]) if fb is None else fb
        new = object.__new__(LiveState5)
        new.__dict__.update(self.__dict__)
        old_ids = self.ids
        new._set_items(cat, item, fb)
        if len(new.ids) != len(old_ids) or (new.ids != old_ids).any():   # productos nuevos: renumerar
            where = np.array([new.idx.get(p, 0) for p in old_ids], dtype=np.int64)
            new.by_prod = self.by_prod.remap(where)
            for name, level in (("by_sub", "sub"), ("by_brand", "brand"), ("by_cat", "cat")):
                old_codes, new_codes = self.items.codes[level], new.items.codes[level]
                table = getattr(self, name)
                where_k = np.array([new_codes.get(k, 0) for k in old_codes], dtype=np.int64)
                setattr(new, name, table.remap(where_k))
        return new

    # ── por petición ─────────────────────────────────────────────────────────────────────────────────

    def user(self, req) -> int | None:
        """Código del usuario, o None si no tiene historia. A diferencia de v2 no se mira is_new_user: un
        comprador nuevo que ya ha navegado hoy tiene historia y se usa."""
        return self.ucode.get(req.get("user_id"))

    def bought(self, u: int | None) -> np.ndarray:
        """Índices de los productos que el usuario ya compró (para no volver a ofrecérselos)."""
        if u is None:
            return np.zeros(0, dtype=int)
        r = self.by_prod.rows(u)
        return self.by_prod.k[r][self.by_prod.v["buys"][r] > 0]

    def features(self, req, schema) -> pd.DataFrame:
        out = self._columns(req, schema)
        return pd.DataFrame({f: out[f] for f in schema.features})

    def matrices(self, req, schema, rows: np.ndarray, out: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(numéricas float32 [n, k], categóricas en bytes [n, c]) en el orden del esquema, sin pasar por pandas."""
        out = self._columns(req, schema) if out is None else out
        num = np.empty((len(rows), len(schema.numeric)), dtype=np.float32)
        for j, f in enumerate(schema.numeric):
            num[:, j] = np.asarray(out[f], dtype=float)[rows]
        cat = np.empty((len(rows), len(schema.categorical)), dtype=object)
        for j, f in enumerate(schema.categorical):
            col = self.static_bytes.get(f)
            cat[:, j] = col[rows] if col is not None else str(out[f][0]).encode()
        return num, cat

    def _columns(self, req, schema) -> dict[str, object]:
        it, n = self.items, self.n
        now = np.datetime64(pd.Timestamp(str(req.get("ts"))[:23]) if req.get("ts")
                            else pd.Timestamp.now("UTC").tz_localize(None), "ns")
        out: dict[str, object] = dict(self.static)
        out["it_age_days"] = (now - it.created) / np.timedelta64(1, "D")
        out["position"] = np.full(n, float(schema.reference_position))
        ctx = req.get("context") or {}
        for key in ("page", "device", "region"):
            out[key] = np.full(n, ctx.get(key) or "UNKNOWN", dtype=object)
        z = np.zeros(n)
        for key in ("device", "region"):
            code, tables = self.ctx[key]
            k = code.get(ctx.get(key))
            for name, m in tables.items():
                out[name] = m[k] if k is not None else z

        # sesión
        views, secs = feat5.parse_views(it, req.get("recent_views"), now)
        out.update(feat5.session_block(it, views, secs, self.nb))

        # usuario
        u = self.user(req)
        agg = {k: z for k in feat5.USER_AGG}
        pair = {"up_views": z, "up_carts": z, "up_c_clicks": z, "up_days_since_seen": np.full(n, np.nan)}
        legacy = {"u_events": z, "u_buys": z, "u_sub_w": z, "u_brand_w": z, "u_sub_buys": z, "u_brand_buys": z}
        if u is not None:
            def spread(table: _ByUser, col: str, size: int) -> np.ndarray:
                r = table.rows(u)
                vec = np.zeros(size + 1)                       # hueco extra: "sin clave"
                vec[table.k[r]] = table.v[col][r]
                return vec

            n_sub, n_brand, n_cat = (len(it.codes[c]) for c in ("sub", "brand", "cat"))
            i1, i2 = np.where(it.a1 >= 0, it.a1, feat5.N_ATTR), np.where(it.a2 >= 0, it.a2, feat5.N_ATTR)
            attr = {c: spread(self.by_attr, c, feat5.N_ATTR) for c in ("w", "r", "buys")}
            agg = {k: np.full(n, v[u]) for k, v in self.user_cols.items()}
            agg.update({
                "u_sub_w": spread(self.by_sub, "w", n_sub)[it.sub], "u_brand_w": spread(self.by_brand, "w", n_brand)[it.brand],
                "u_cat_w": spread(self.by_cat, "w", n_cat)[it.cat],
                "u_attr_w1": attr["w"][i1], "u_attr_w2": attr["w"][i2],
                "u_sub_r": spread(self.by_sub, "r", n_sub)[it.sub], "u_cat_r": spread(self.by_cat, "r", n_cat)[it.cat],
                "u_attr_r": attr["r"][i1] + attr["r"][i2],
                "u_sub_buys": spread(self.by_sub, "buys", n_sub)[it.sub],
                "u_brand_buys": spread(self.by_brand, "buys", n_brand)[it.brand],
                "u_cat_buys": spread(self.by_cat, "buys", n_cat)[it.cat],
                "u_attr_buys": attr["buys"][i1] + attr["buys"][i2],
            })
            r = self.by_prod.rows(u)
            last_seen = np.full(n, NEVER, dtype="datetime64[ns]")
            last_seen[self.by_prod.k[r]] = self.by_prod.v["last"][r]
            pair = {"up_views": spread(self.by_prod, "views", n)[:n], "up_carts": spread(self.by_prod, "carts", n)[:n],
                    "up_c_clicks": spread(self.by_prod, "c_clicks", n)[:n],
                    "up_days_since_seen": np.where(last_seen > NEVER, (now - last_seen) / np.timedelta64(1, "D"), np.nan)}
            legacy = {k: agg[k] for k in legacy}
        out.update(feat5.user_feats(agg, it.logp))
        out.update(pair)
        out.update(legacy)
        out["is_new_user"] = np.full(n, 1.0 if (req.get("is_new_user") or u is None) else 0.0)
        missing = [f for f in schema.features if f not in out]
        if missing:
            raise NotImplementedError(f"LiveState5 no calcula {missing} (esquema {schema.version})")
        return out


# ── paridad entrenamiento ↔ servicio ─────────────────────────────────────────────────────────────────

def parity_report(until: str, minutes: float = 5.0, n: int = 300, verbose: bool = True) -> pd.DataFrame:
    """Compara, para peticiones reales de los `minutes` siguientes a `until`, las features del dataset
    (calculadas al milisegundo) con las que daría el servicio con un estado descargado en `until`.
    Deben coincidir salvo por lo que el usuario hizo entre `until` y la petición."""
    import pickle

    from .build_v5 import OUT, RAW
    from .schema import V5
    ds = pd.read_parquet(OUT / "live_v5.parquet")
    t0 = pd.Timestamp(until, tz="UTC")
    ds = ds[(ds.timestamp >= t0) & (ds.timestamp < t0 + pd.Timedelta(minutes=minutes))]
    sess = pickle.loads((RAW / "live_sessions.pkl").read_bytes()).set_index("request_id")
    ids = ds.request_id.drop_duplicates().head(n)
    state = LiveState5.load(until)
    rows = []
    for rid in ids:
        s = sess.loc[rid]
        req = {"request_id": rid, "user_id": s.user_id, "is_new_user": bool(s.is_new_user), "context": s.context,
               "recent_views": s.recent_views, "ts": s.ts}
        cols = state._columns(req, V5)
        d = ds[ds.request_id == rid]
        at = np.array([state.idx[p] for p in d.product_id])
        for f in V5.features:
            if f in V5.categorical:
                same = np.array([str(a) == str(b) for a, b in zip(np.asarray(cols[f], dtype=object)[at], d[f])])
                rows += [{"feature": f, "serve": float(x), "train": 1.0} for x in same]
            elif f != "position":
                rows += [{"feature": f, "serve": float(a), "train": float(b)}
                         for a, b in zip(np.asarray(cols[f], dtype=float)[at], d[f].to_numpy(dtype=float))]
    r = pd.DataFrame(rows)
    both_nan = r.serve.isna() & r.train.isna()
    r["diff"] = np.where(both_nan, 0.0, (r.serve - r.train).abs())
    scale = r.groupby("feature").train.transform(lambda x: max(x.abs().quantile(0.95), 1e-9))
    r["close"] = (r["diff"] <= 0.02 * scale + 1e-6) | both_nan
    out = r.groupby("feature").agg(n=("diff", "size"), iguales=("close", "mean"), dif_media=("diff", "mean"),
                                   train_media=("train", "mean"), serve_media=("serve", "mean"))
    out["corr"] = r.groupby("feature").apply(lambda g: g[["serve", "train"]].corr().iloc[0, 1], include_groups=False)
    out = out.sort_values("iguales")
    if verbose:
        print(f"{len(ids)} peticiones entre {until} y +{minutes:g} min · estado descargado a fecha {until}")
        print(out.to_string(float_format=lambda v: f"{v:.4f}"))
    return out
