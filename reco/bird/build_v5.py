"""Construye el dataset v5 de Bird: histórico + nuestro feedback en vivo, con las features de reco/bird/feat5.py.

    python scripts/bird_v5.py dataset     # → data/bird/v5/{train,validation,test,live}_v5.parquet

Filas (una por petición y producto enseñado):
  hist   las mismas que v2: shop.history_sessions desde START; todos los carruseles con clic + 1 de cada 25 sin clic
  live   nuestro carrusel de hoy (team_NN.feedback × shop.requests): todos los que tienen clic + 1 de cada 10

Tablas de RawTree que se usan y para qué:
  shop.history_events / shop.events   historia del usuario, popularidad por ventanas, qué compra cada contexto
  shop.history_sessions               filas hist, clics y compras de carruseles anteriores, CTR/CVR por producto
  shop.requests                       contexto y vistas de la sesión de las filas live
  shop.catalog / catalog_updates      atributos, precio base, promociones (precio en el instante), productos nuevos
  team_NN.feedback                    filas live y respuesta de cada producto en nuestro carrusel (COEC)

Sin fugas temporales: cada bloque solo mira lo anterior a la petición.
  usuario × producto   SQL: eventos del usuario con ts < ts de la petición (al milisegundo)
  producto             recuentos acumulados hasta el último tramo COMPLETO (1 h en el histórico, 10 min en vivo)
  contexto             compras por dispositivo / región anteriores al inicio de la semana de la petición
  sesión               recent_views de la petición; vecinos calculados con historia anterior a su semana
"""
from __future__ import annotations

import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .. import features
from ..config import DATA_DIR, TEAM
from ..features import _ATTR_SQL
from . import build_dataset as bd
from . import feat5
from .build_dataset import BUY, CHUNKS, NOCLICK_MOD, SELECTED, START, TEST_FROM, VALID_FROM, WEEKS, _query

ORIGIN = "2026-09-15 00:00:00"          # hora 0 de la rejilla horaria
LIVE_START = "2026-11-14 09:00:00"      # primer evento de shop.events
LIVE_MOD = 10                           # 1 de cada LIVE_MOD carruseles en vivo sin clic
LIVE_CHUNKS = 8
CTX_CUTS = WEEKS + [LIVE_START]
RAW = DATA_DIR / "bird" / "raw5"
OUT = DATA_DIR / "bird" / "v5"
H_LIVE = int((pd.Timestamp(LIVE_START) - pd.Timestamp(ORIGIN)) / pd.Timedelta(hours=1))
B0 = int(pd.Timestamp(LIVE_START).timestamp()) // 600

# catálogo con productos nuevos; precio BASE (el primero que tuvo), no el de la promoción del momento
CATALOG5 = f"""(SELECT product_id, argMin(r_cat, v) AS category, argMin(r_sub, v) AS subcategory,
        argMin(r_brand, v) AS brand, argMin(r_price, v) AS base_price, log(1 + argMin(r_price, v)) AS logp,
        argMax(r_price, v) AS price_eur, min(r_created) AS created_at,
        arrayFilter(a -> position(argMax(r_desc, v), a) > 0, {_ATTR_SQL}) AS attrs
    FROM (SELECT product_id::String AS product_id, category::String AS r_cat, subcategory::String AS r_sub,
                 brand::String AS r_brand, price_eur::Float64 AS r_price, description::String AS r_desc,
                 toString(created_at::DateTime64(3)) AS r_created, '' AS v FROM shop.catalog
          UNION ALL
          SELECT product_id::String, category::String, subcategory::String, brand::String, price_eur::Float64,
                 description::String, toString(created_at::DateTime64(3)), toString(ts::DateTime64(3))
          FROM shop.catalog_updates{{cut}})
    GROUP BY product_id)"""


def catalog_sql(until: str | None = None) -> str:
    return CATALOG5.format(cut=f" WHERE ts::DateTime64(3) < '{until}'" if until else "")


# ── SQL: historia del usuario antes de cada petición ─────────────────────────────────────────────────

def events_sql(live: bool, part: str, until: str | None = None) -> str:
    """Eventos del usuario con su peso (vista 1, carrito 3, compra 5; clic de carrusel 1, compra 5).
    live=False: solo el histórico (las filas hist no pueden ver nada posterior)."""
    cut = f" AND ts::DateTime64(3) < '{until}'" if until else ""
    ev = ("SELECT user_id::String AS user_id, product_id::String AS product_id, ts::DateTime64(3) AS ts, "
          "event_type::String AS kind, multiIf(event_type::String = 'purchase', 5, "
          "event_type::String = 'add_to_cart', 3, 1) AS w FROM {table} WHERE " + part + cut)
    parts = [ev.format(table="shop.history_events"),
             "SELECT user_id::String, clicked::String, ts::DateTime64(3), "
             "if(purchased::String != '', 'c_purchase', 'c_click'), if(purchased::String != '', 5, 1) "
             f"FROM shop.history_sessions WHERE clicked::String != '' AND {part}{cut}"]
    if live:
        fb_cut = f" AND f.ts::DateTime64(3) < '{until}'" if until else ""
        parts += [ev.format(table="shop.events"),
                  "SELECT r.user_id::String, f.clicked::String, f.ts::DateTime64(3), "
                  "if(f.purchased::String != '', 'c_purchase', 'c_click'), if(f.purchased::String != '', 5, 1) "
                  f"FROM {TEAM}.feedback AS f INNER JOIN shop.requests AS r "
                  "ON r.request_id::String = f.request_id::String "
                  "WHERE f.clicked::String != '' AND f.request_id::String != '__init__' AND "
                  + part.replace("user_id::String", "r.user_id::String") + fb_cut]
    return "(" + " UNION ALL ".join(parts) + ")"


def live_selected(until: str) -> str:
    return (f"f.request_id::String != '__init__' AND f.ts::DateTime64(3) < '{until}' AND "
            f"(f.clicked::String != '' OR cityHash64(f.request_id::String) % {LIVE_MOD} = 0)")


ORG_WINDOW = 30        # minutos tras la petición en los que una compra orgánica cuenta como "lo que quería"
ORG_NEG_MOD = 270      # 1 de cada 270 productos del catálogo como negativo de cada petición (~20)


def org_pairs_sql(until: str, part: str = "1") -> str:
    """Filas orgánicas: (petición, producto, pos). No dependen de lo que enseñó nuestro carrusel.
      pos = 1  el usuario compró ese producto POR SU CUENTA (shop.events) en los ORG_WINDOW min siguientes
      pos = 0  ~20 productos del catálogo de ese momento elegidos por hash (los mismos en cada ejecución)
    Solo peticiones con alguna compra orgánica después, y con los ORG_WINDOW min ya observados."""
    r = (f"(SELECT request_id::String AS request_id, user_id::String AS user_id, ts::DateTime64(3) AS ts "
         f"FROM shop.requests WHERE ts::DateTime64(3) >= '{LIVE_START}' "
         f"AND ts::DateTime64(3) < toDateTime64('{until}', 3) - INTERVAL {ORG_WINDOW} MINUTE AND {part})")
    b = (f"(SELECT user_id::String AS user_id, product_id::String AS product_id, ts::DateTime64(3) AS ts "
         f"FROM shop.events WHERE event_type::String = 'purchase' AND ts::DateTime64(3) < '{until}' AND {part})")
    pos = (f"(SELECT DISTINCT r.request_id AS request_id, r.user_id AS user_id, r.ts AS ts, b.product_id AS product_id "
           f"FROM {r} AS r INNER JOIN {b} AS b ON b.user_id = r.user_id "
           f"WHERE b.ts > r.ts AND b.ts <= r.ts + INTERVAL {ORG_WINDOW} MINUTE)")
    neg = (f"(SELECT q.request_id AS request_id, q.user_id AS user_id, q.ts AS ts, c.product_id AS product_id "
           f"FROM (SELECT DISTINCT request_id, user_id, ts FROM {pos}) AS q "
           f"CROSS JOIN (SELECT product_id, created_at FROM {catalog_sql()}) AS c "
           f"WHERE cityHash64(q.request_id, c.product_id) % {ORG_NEG_MOD} = 0 "
           f"AND toDateTime64(c.created_at, 3) <= q.ts)")
    return (f"SELECT request_id, any(user_id) AS user_id, any(ts) AS ts, product_id, max(pos) AS pos FROM ("
            f"SELECT request_id, user_id, ts, product_id, 1 AS pos FROM {pos} UNION ALL "
            f"SELECT request_id, user_id, ts, product_id, 0 AS pos FROM {neg}) GROUP BY request_id, product_id")


def user_item_sql(k: int, chunks: int, live: bool, until: str | None = None, org: bool = False) -> str:
    """Por (petición, producto enseñado): recuentos de la historia del usuario ANTERIOR a la petición.
    De ellos salen las features de usuario, afinidad y usuario × producto (feat5.user_feats).
    org=True: en vez de los productos enseñados, los pares de org_pairs_sql (compras orgánicas + negativos)."""
    part = f"cityHash64(user_id::String) % {chunks} = {k}"
    if org:
        shown = f"SELECT request_id, user_id, ts, product_id FROM ({org_pairs_sql(until, part)})"
    elif live:
        shown = f"""SELECT f.request_id::String AS request_id, r.user_id::String AS user_id,
                           r.ts::DateTime64(3) AS ts, arrayJoin(f.shown::Array(String)) AS product_id
                    FROM {TEAM}.feedback AS f INNER JOIN shop.requests AS r
                      ON r.request_id::String = f.request_id::String
                    WHERE {live_selected(until)} AND {part.replace('user_id::String', 'r.user_id::String')}"""
    else:
        shown = f"""SELECT request_id::String AS request_id, user_id::String AS user_id, ts::DateTime64(3) AS ts,
                           arrayJoin(shown::Array(String)) AS product_id
                    FROM shop.history_sessions WHERE {SELECTED} AND {part}"""
    same = "e.product_id = s.product_id"
    decay = f"exp(-dateDiff('millisecond', e.ts, s.ts) / {feat5.TAU_MS:.1f})"
    attr_n = "(has(ec.attrs, s.a1) + has(ec.attrs, s.a2))"
    return f"""
SELECT s.request_id AS request_id, s.product_id AS product_id,
  countIf({same} AND e.kind = 'view') AS up_views,
  countIf({same} AND e.kind = 'add_to_cart') AS up_carts,
  countIf({same} AND e.kind = 'purchase') AS up_buys,
  countIf({same} AND e.kind = 'c_click') AS up_c_clicks,
  countIf({same} AND e.kind = 'c_purchase') AS up_c_buys,
  toString(maxIf(e.ts, {same})) AS up_last_seen,
  count() AS u_events, countIf(e.kind IN {BUY}) AS u_buys, sum(e.w) AS u_w, sum(e.w * {decay}) AS u_r,
  avg(ec.logp) AS u_logp, avgIf(ec.logp, e.kind IN {BUY}) AS u_buy_logp, stddevPop(ec.logp) AS u_logp_sd,
  sumIf(e.w, ec.subcategory = s.subcategory) AS u_sub_w,
  sumIf(e.w, ec.brand = s.brand) AS u_brand_w,
  sumIf(e.w, ec.category = s.category) AS u_cat_w,
  sumIf(e.w, has(ec.attrs, s.a1)) AS u_attr_w1,
  sumIf(e.w, has(ec.attrs, s.a2)) AS u_attr_w2,
  sumIf(e.w * {decay}, ec.subcategory = s.subcategory) AS u_sub_r,
  sumIf(e.w * {decay}, ec.category = s.category) AS u_cat_r,
  sum(e.w * {decay} * {attr_n}) AS u_attr_r,
  countIf(e.kind IN {BUY} AND ec.subcategory = s.subcategory) AS u_sub_buys,
  countIf(e.kind IN {BUY} AND ec.brand = s.brand) AS u_brand_buys,
  countIf(e.kind IN {BUY} AND ec.category = s.category) AS u_cat_buys,
  sumIf({attr_n}, e.kind IN {BUY}) AS u_attr_buys,
  sumIf(e.w, ec.subcategory = s.subcategory AND e.ts >= s.ts - INTERVAL 7 DAY) AS u_sub_w_7d,
  toString(max(e.ts)) AS u_last_ts
FROM (
  SELECT h.request_id AS request_id, h.user_id AS user_id, h.ts AS ts, h.product_id AS product_id,
         c.category AS category, c.subcategory AS subcategory, c.brand AS brand,
         c.attrs[1] AS a1, c.attrs[2] AS a2
  FROM ({shown}) AS h
  INNER JOIN {catalog_sql()} AS c ON c.product_id = h.product_id
) AS s
INNER JOIN {events_sql(live, part, until)} AS e ON e.user_id = s.user_id
INNER JOIN {catalog_sql()} AS ec ON ec.product_id = e.product_id
WHERE e.ts < s.ts
GROUP BY request_id, product_id"""


# ── SQL: recuentos por producto y tramo de tiempo ────────────────────────────────────────────────────

def item_hourly_sql(k: int, chunks: int) -> str:
    return f"""SELECT p, groupArray(h) AS h, groupArray(v) AS v, groupArray(c) AS c, groupArray(b) AS b FROM (
        SELECT product_id::String AS p, dateDiff('hour', toDateTime64('{ORIGIN}', 3), ts::DateTime64(3)) AS h,
               countIf(event_type::String = 'view') AS v, countIf(event_type::String = 'add_to_cart') AS c,
               countIf(event_type::String = 'purchase') AS b
        FROM shop.history_events WHERE cityHash64(product_id::String) % {chunks} = {k} GROUP BY p, h)
    GROUP BY p"""


ITEM_LIVE10 = """SELECT p, groupArray(t) AS h, groupArray(v) AS v, groupArray(c) AS c, groupArray(b) AS b FROM (
        SELECT product_id::String AS p, intDiv(toUnixTimestamp(ts::DateTime64(3)), 600) AS t,
               countIf(event_type::String = 'view') AS v, countIf(event_type::String = 'add_to_cart') AS c,
               countIf(event_type::String = 'purchase') AS b
        FROM shop.events WHERE ts::DateTime64(3) < '{until}' GROUP BY p, t)
    GROUP BY p"""

FEEDBACK10 = """SELECT x.1 AS p, intDiv(toUnixTimestamp(ts::DateTime64(3)), 600) AS t, x.2 AS slot, count() AS impr,
           countIf(clicked::String = x.1) AS clicks, countIf(purchased::String = x.1) AS buys
    FROM (SELECT arrayJoin(arrayZip(shown::Array(String), arrayEnumerate(shown::Array(String)))) AS x,
                 clicked, purchased, ts
          FROM {team}.feedback WHERE request_id::String != '__init__' AND ts::DateTime64(3) < '{until}')
    GROUP BY p, t, slot"""


def context_sql(key: str, until: str) -> str:
    """Compras orgánicas por (dispositivo | región, producto, tramo entre cortes)."""
    cuts = "[" + ", ".join(f"toDateTime64('{c}', 3)" for c in CTX_CUTS) + "]"
    one = (f"SELECT {key}::String AS k, product_id::String AS p, "
           f"arrayCount(c -> ts::DateTime64(3) >= c, {cuts}) AS bucket FROM {{t}} "
           f"WHERE event_type::String = 'purchase' AND ts::DateTime64(3) < '{until}'")
    return (f"SELECT k, p, bucket, count() AS n FROM ({one.format(t='shop.history_events')} UNION ALL "
            f"{one.format(t='shop.events')}) GROUP BY k, p, bucket")


# ── descargas (con caché en data/bird/raw5/) ─────────────────────────────────────────────────────────

def _cached(name: str, make, refresh: bool = False):
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / name
    if path.exists() and not refresh:
        return pickle.loads(path.read_bytes()) if path.suffix == ".pkl" else pd.read_parquet(path)
    obj = make()
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".pkl":
        tmp.write_bytes(pickle.dumps(obj))
    else:
        obj.to_parquet(tmp, index=False)
    tmp.replace(path)
    return obj


def _chunks(name: str, n: int, sql, refresh: bool, say, workers: int = 4) -> pd.DataFrame:
    def one(k: int) -> pd.DataFrame:
        t = time.time()
        df = _cached(f"{name}_{k:02d}.parquet", lambda: pd.DataFrame(_query(sql(k))), refresh)
        if time.time() - t > 1:
            say(f"   {name} {k + 1}/{n}: {len(df):,} filas  {time.time() - t:.0f}s")
        return df
    with ThreadPoolExecutor(workers) as pool:
        return pd.concat(list(pool.map(one, range(n))), ignore_index=True)


def live_until(refresh: bool = False) -> str:
    """Corte de la parte en vivo: el último tramo completo de 10 min cuando se descargó por primera vez."""
    path = RAW / "live_until.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())["until"]
    last = pd.Timestamp(_query("SELECT toString(max(ts::DateTime64(3))) AS t FROM shop.events")[0]["t"])
    until = str(last.floor("10min") - pd.Timedelta(minutes=10))
    RAW.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"until": until}))
    return until


def load_catalog(refresh: bool = False) -> pd.DataFrame:
    def make() -> pd.DataFrame:
        df = pd.DataFrame(_query(f"SELECT * FROM {catalog_sql()}"))
        df["attrs"] = df["attrs"].apply(lambda v: list(v) if v is not None else [])
        return df.sort_values("product_id").reset_index(drop=True)
    return _cached("catalog.pkl", make, refresh)


def load_updates(refresh: bool = False) -> pd.DataFrame:
    return _cached("updates.parquet", lambda: pd.DataFrame(_query(
        "SELECT product_id::String AS product_id, toString(ts::DateTime64(3)) AS ts, price_eur::Float64 AS price_eur, "
        "change::String AS change FROM shop.catalog_updates ORDER BY ts")), refresh)


def load_live_sessions(until: str, refresh: bool = False) -> pd.DataFrame:
    def make() -> pd.DataFrame:
        # dos consultas: `context` y `recent_views` (objetos) no pasan por un JOIN en RawTree
        fb = pd.DataFrame(_query(f"""SELECT f.request_id::String AS request_id, f.shown AS shown,
                f.clicked::String AS clicked, f.purchased::String AS purchased,
                f.revenue_eur::Float64 AS revenue_eur, f.served_by::String AS served_by
            FROM {TEAM}.feedback AS f WHERE {live_selected(until)}""")).drop_duplicates("request_id")
        rq = pd.DataFrame(_query(f"""SELECT request_id::String AS request_id, toString(ts::DateTime64(3)) AS ts,
                user_id::String AS user_id, is_new_user::Bool AS is_new_user, context, recent_views
            FROM shop.requests WHERE request_id::String IN (
                SELECT f.request_id::String FROM {TEAM}.feedback AS f WHERE {live_selected(until)})"""))
        df = rq.drop_duplicates("request_id").merge(fb, on="request_id")
        df["shown"] = df.shown.apply(list)
        df["recent_views"] = df.recent_views.apply(lambda v: list(v) if v is not None else [])
        return df.sort_values(["ts", "request_id"]).reset_index(drop=True)
    return _cached("live_sessions.pkl", make, refresh)


# ── producto: recuentos acumulados en una rejilla de tiempo (1 h en el histórico, 10 min en vivo) ─────

class Timeline:
    """cum[k][i, p] = eventos de tipo k del producto p ANTERIORES al borde i de la rejilla."""

    def __init__(self, items: feat5.Items, hourly: pd.DataFrame, live10: pd.DataFrame, until: str) -> None:
        n_live = int(pd.Timestamp(until).timestamp()) // 600 - B0            # tramos completos de 10 min
        self.n_buckets = H_LIVE + n_live
        origin = pd.Timestamp(ORIGIN).value
        edges = [origin + h * 3600 * 10**9 for h in range(H_LIVE + 1)]
        edges += [(B0 + j) * 600 * 10**9 for j in range(1, n_live + 1)]
        self.edges = np.array(edges, dtype=np.int64)
        self.cum = {}
        cnt = {k: np.zeros((self.n_buckets, items.n), dtype=np.float32) for k in ("v", "c", "b")}
        for df, offset in ((hourly, 0), (live10, H_LIVE - B0)):
            for p, h, v, c, b in zip(df.p, df.h, df.v, df.c, df.b):
                i = items.idx.get(p)
                if i is None:
                    continue
                t = np.asarray(h, dtype=np.int64) + offset
                ok = (t >= (0 if offset == 0 else H_LIVE)) & (t < (H_LIVE if offset == 0 else self.n_buckets))
                for key, val in (("v", v), ("c", c), ("b", b)):
                    cnt[key][t[ok], i] += np.asarray(val, dtype=np.float32)[ok]
        for key, m in cnt.items():
            self.cum[key] = np.vstack([np.zeros((1, items.n), dtype=np.float64), np.cumsum(m, axis=0, dtype=np.float64)])

    def edge(self, ts_ns: np.ndarray) -> np.ndarray:
        """Último borde ≤ ts: los recuentos hasta ahí son de tramos completos y anteriores a la petición."""
        return np.clip(np.searchsorted(self.edges, ts_ns, side="right") - 1, 0, len(self.edges) - 1)

    def window(self, key: str, at: np.ndarray, seconds: float | None) -> np.ndarray:
        """[len(at), productos]: eventos en los `seconds` anteriores a cada borde (None = todos)."""
        cum = self.cum[key]
        if seconds is None:
            return cum[at]
        lo = np.clip(np.searchsorted(self.edges, self.edges[at] - int(seconds * 1e9), side="right") - 1, 0, None)
        return cum[at] - cum[lo]

    def counts(self, at: np.ndarray) -> dict[str, np.ndarray]:
        day, hour = 86400.0, 3600.0
        return {"views": self.window("v", at, None), "carts": self.window("c", at, None),
                "buys": self.window("b", at, None), "views_7d": self.window("v", at, 7 * day),
                "buys_7d": self.window("b", at, 7 * day), "views_1d": self.window("v", at, day),
                "buys_1d": self.window("b", at, day), "views_1h": self.window("v", at, hour)}


def position_curve(fb: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """CTR y CVR medios por posición de nuestro carrusel (índice 1..10)."""
    by = fb.groupby("slot")[["impr", "clicks", "buys"]].sum().reindex(range(1, 11), fill_value=0)
    impr = by.impr.clip(lower=1).to_numpy(dtype=float)
    ctr, cvr = np.zeros(11), np.zeros(11)
    ctr[1:], cvr[1:] = by.clicks.to_numpy() / impr, by.buys.to_numpy() / impr
    return ctr, cvr


class FeedbackTimeline:
    """Clics y compras reales y esperados (por posición) de cada producto en nuestro carrusel, acumulados
    en la rejilla de 10 min. Borde j = inicio del tramo B0 + j."""

    def __init__(self, items: feat5.Items, fb: pd.DataFrame, until: str) -> None:
        n_live = int(pd.Timestamp(until).timestamp()) // 600 - B0
        self.ctr, self.cvr = position_curve(fb)
        fb = fb[fb.p.isin(items.idx) & fb.slot.between(1, 10)]
        t = (fb.t.to_numpy(dtype=np.int64) - B0).clip(0, max(n_live - 1, 0))
        p = fb.p.map(items.idx).to_numpy()
        slot = fb.slot.to_numpy(dtype=int)
        impr = fb.impr.to_numpy(dtype=float)
        self.cum = {}
        for key, val in (("clicks", fb.clicks.to_numpy(dtype=float)), ("buys", fb.buys.to_numpy(dtype=float)),
                         ("exp_clicks", impr * self.ctr[slot]), ("exp_buys", impr * self.cvr[slot])):
            m = np.zeros((max(n_live, 1), items.n))
            np.add.at(m, (t, p), val)
            self.cum[key] = np.vstack([np.zeros((1, items.n)), np.cumsum(m, axis=0)])

    def lifts(self, edge_live: np.ndarray, flat: np.ndarray) -> dict[str, np.ndarray]:
        """edge_live: borde de 10 min de cada fila (0 = antes de que hubiera feedback)."""
        j = np.clip(edge_live, 0, len(self.cum["clicks"]) - 1)
        g = lambda k: self.cum[k][j, flat]   # noqa: E731
        return feat5.feedback_lift(g("clicks"), g("exp_clicks"), g("buys"), g("exp_buys"))


def context_tables(items: feat5.Items, refresh: bool, until: str) -> dict[str, tuple[dict[str, int], list[dict]]]:
    """Por contexto: (código de cada valor, [lifts a fecha de cada corte de CTX_CUTS])."""
    out = {}
    for key, prefix, levels in (("device", "dev", feat5.DEV_LEVELS), ("region", "reg", feat5.REG_LEVELS)):
        df = _cached(f"ctx_{key}.parquet", lambda: pd.DataFrame(_query(context_sql(key, until))), refresh)
        df = df[df.p.isin(items.idx)]
        code = {k: i for i, k in enumerate(sorted(df.k.unique()))}
        counts = np.zeros((len(CTX_CUTS) + 1, len(code), items.n))
        np.add.at(counts, (df.bucket.to_numpy(dtype=int), df.k.map(code).to_numpy(), df.p.map(items.idx).to_numpy()),
                  df.n.to_numpy(dtype=float))
        as_of = np.cumsum(counts, axis=0)      # as_of[j] = compras anteriores a CTX_CUTS[j]
        out[key] = (code, [feat5.context_lifts(as_of[j], items, prefix, levels) for j in range(len(CTX_CUTS))])
    return out


# ── sesión ───────────────────────────────────────────────────────────────────────────────────────────

def neighbors(until: str, items: feat5.Items) -> dict[str, dict[int, list[tuple[int, float]]]]:
    """Vecinos (vista→compra y co-vista) calculados solo con eventos anteriores a `until`."""
    return neighbor_dict(neighbor_table(until), items)


def neighbor_table(until: str) -> pd.DataFrame:
    """La tabla item_neighbors de reco/features.py a fecha de `until`. Si ya está en data/features_until_*
    se reutiliza; si no, se guarda en raw5/ (una carpeta features_until_* a medias rompería features.load)."""
    path = features.folder(until) / "item_neighbors.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return _cached(f"neighbors_{features.tag(until)}.parquet",
                   lambda: pd.DataFrame(_query(features.queries(until)["item_neighbors"])))


def neighbor_dict(nb: pd.DataFrame, items: feat5.Items) -> dict[str, dict[int, list[tuple[int, float]]]]:
    out = {}
    idx = items.idx
    for kind in ("buy", "view"):
        d = nb[nb.kind == kind]
        d = d.assign(nn=d.n / d.groupby("p").n.transform("max")).sort_values(["p", "nn"], ascending=[True, False])
        out[kind] = {idx[p]: [(idx[q], v) for q, v in zip(g.q, g.nn) if q in idx][:20]
                     for p, g in d.groupby("p", sort=False) if p in idx}
    return out


def live_cuts(until: str) -> list[str]:
    """Cortes de los vecinos para las filas en vivo: cada 4 h desde el inicio del stream."""
    cuts, t = [], pd.Timestamp(LIVE_START)
    while t < pd.Timestamp(until):
        cuts.append(str(t))
        t += pd.Timedelta(hours=4)
    return cuts


def session_features(items: feat5.Items, sess: pd.DataFrame, shown: np.ndarray, cuts: list[str]) -> np.ndarray:
    """[petición, 10, len(feat5.SESSION)] con la MISMA función que usa el servicio (feat5.session_block)."""
    out = np.zeros((len(sess), 10, len(feat5.SESSION)), dtype=np.float32)
    ts = sess.ts.to_numpy().astype("datetime64[ns]")
    which = np.clip(np.searchsorted(np.array(cuts, dtype="datetime64[ns]"), ts, side="right") - 1, 0, None)
    nbs = {w: neighbors(cuts[w], items) for w in np.unique(which)}
    for r, (views, w, now) in enumerate(zip(sess.recent_views, which, ts)):
        v, secs = feat5.parse_views(items, views, now)
        block = feat5.session_block(items, v, secs, nbs[w])
        p = shown[r]
        for j, name in enumerate(feat5.SESSION):
            out[r, :, j] = block[name][p]
    return out


# ── ensamblado ───────────────────────────────────────────────────────────────────────────────────────

def _days_since(ts: pd.Series, past: pd.Series) -> np.ndarray:
    past = pd.to_datetime(past, errors="coerce")
    days = (ts - past).dt.total_seconds() / 86400
    return days.where(past > pd.Timestamp("2000-01-01")).to_numpy(dtype=float)   # 1970 = "nunca"


def _frame(sess: pd.DataFrame, source: str, items: feat5.Items, ui: pd.DataFrame, tl: Timeline,
           fbt: FeedbackTimeline, ctx: dict, daily: pd.DataFrame, updates: pd.DataFrame, cuts: list[str],
           say) -> pd.DataFrame:
    live = source == "live"
    idx = items.idx
    ok = sess.shown.apply(lambda s: len(s) == 10 and len(set(s)) == 10 and all(p in idx for p in s))
    if not ok.all():
        say(f"   [{source}] se descartan {int((~ok).sum())} peticiones con carrusel incompleto o productos desconocidos")
        sess = sess[ok].reset_index(drop=True)
    n = len(sess)
    shown = np.array([[idx[p] for p in s] for s in sess.shown])
    flat = shown.ravel()
    rep = lambda col: np.repeat(sess[col].to_numpy(), 10)   # noqa: E731
    ts_req = sess.ts.to_numpy().astype("datetime64[ns]")

    df = pd.DataFrame({
        "request_id": rep("request_id"), "user_id": rep("user_id"), "product_id": items.ids[flat],
        "timestamp": pd.to_datetime(np.repeat(ts_req, 10), utc=True),
        "position": np.tile(np.arange(1, 11), n).astype("int8"),
    })
    df["target_purchase"] = (df.product_id.to_numpy() == rep("purchased")).astype("int8")
    df["target_click"] = (df.product_id.to_numpy() == rep("clicked")).astype("int8")
    df["propensity"] = 1.0 if live else np.concatenate([np.asarray(p, dtype=float) for p in sess.propensity])
    df["sample_weight"] = np.where(rep("clicked") != "", 1.0, float(LIVE_MOD if live else NOCLICK_MOD))
    df["revenue_eur"] = np.where(df.target_purchase == 1, rep("revenue_eur"), 0.0)
    df["source"] = source
    df["served_by"] = rep("served_by") if live else "old"

    # contexto
    ctx_val = {}
    for key in ("page", "device", "region"):
        ctx_val[key] = sess.context.apply(lambda c: (c or {}).get(key) or "UNKNOWN").to_numpy()
        df[key] = np.repeat(ctx_val[key], 10)

    # producto (catálogo): nivel de precio = precio base; la promoción va aparte en it_discount
    for col, arr in (("category", "cat"), ("subcategory", "sub"), ("brand", "brand")):
        inv = np.array(list(items.codes[arr]), dtype=object)
        df[col] = inv[getattr(items, arr)[flat]]
    df["attr1"], df["attr2"] = items.attr1[flat], items.attr2[flat]
    price = items.base_price[flat].copy()
    if live and len(updates):
        # precio en el instante de la petición: última versión del producto anterior a ella
        up = updates.assign(ts=pd.to_datetime(updates.ts).astype("datetime64[ns]")).sort_values("ts")
        left = pd.DataFrame({"ts": np.repeat(ts_req, 10).astype("datetime64[ns]"), "product_id": items.ids[flat],
                             "row": np.arange(n * 10)})
        m = pd.merge_asof(left.sort_values("ts"), up[["ts", "product_id", "price_eur"]], on="ts", by="product_id",
                          allow_exact_matches=False).sort_values("row")
        got = m.price_eur.to_numpy(dtype=float)
        price = np.where(np.isnan(got), price, got)
    df["price_eur"] = price.astype("float32")
    df["it_discount"] = np.clip(1.0 - price / items.base_price[flat], 0.0, 1.0)
    df["it_price_vs_sub"] = items.price_vs_sub[flat]
    df["it_age_days"] = (np.repeat(ts_req, 10) - items.created[flat]) / np.timedelta64(1, "D")

    # producto (popularidad): recuentos hasta el último tramo completo → cuotas, tasas y tendencias
    edge = tl.edge(ts_req.astype(np.int64))
    uniq, inv = np.unique(edge, return_inverse=True)
    row_u = np.repeat(inv, 10)
    legacy = {}
    pop = {name: np.zeros(n * 10, dtype=np.float32) for name in feat5.ITEM_POP}
    for lo in range(0, len(uniq), 128):                     # por tandas: las matrices [borde, producto] pesan
        at = uniq[lo:lo + 128]
        sel = (row_u >= lo) & (row_u < lo + len(at))
        counts = tl.counts(at)
        block = feat5.item_pop(counts, items.sub)
        for name in feat5.ITEM_POP:
            pop[name][sel] = block[name][row_u[sel] - lo, flat[sel]]
        for name, key in (("it_buys_7d", "buys_7d"), ("it_views_7d", "views_7d"), ("it_buys_1d", "buys_1d")):
            legacy.setdefault(name, np.zeros(n * 10, dtype=np.float32))[sel] = counts[key][row_u[sel] - lo, flat[sel]]
    for name in feat5.ITEM_POP:
        df[name] = pop[name]
    for name, col in legacy.items():                        # recuentos absolutos de v2 (solo para comparar)
        df[name] = col
    df["it_trend"] = np.log1p(df.it_buys_1d * 7) - np.log1p(df.it_buys_7d)

    # producto (carrusel del histórico, hasta el día anterior; las filas en vivo ven todo el histórico)
    days = pd.date_range(bd.HISTORY_FROM, "2026-11-15", freq="D")
    d = daily[daily.product_id.isin(idx)]
    di = pd.to_datetime(d.day).map({x: i for i, x in enumerate(days)}).to_numpy()
    keep = ~np.isnan(di)
    cum = {}
    for col in ("c_impr", "c_clicks", "c_buys"):
        m = np.zeros((len(days), items.n))
        np.add.at(m, (di[keep].astype(int), d.product_id.map(idx).to_numpy()[keep]), d[col].to_numpy(dtype=float)[keep])
        cum[col] = np.vstack([np.zeros((1, items.n)), np.cumsum(m, axis=0)])[:-1]   # lo anterior a cada día
    rates = feat5.carousel_rates(cum["c_impr"], cum["c_clicks"], cum["c_buys"])
    day = np.repeat(np.searchsorted(days.to_numpy(), sess.ts.dt.normalize().to_numpy())
                    if not live else np.full(n, len(days) - 1), 10)
    for name, m in rates.items():
        df[name] = m[day, flat]

    # producto (nuestro carrusel en vivo): COEC hasta el último tramo completo de 10 min
    for name, col in fbt.lifts(np.repeat(edge - H_LIVE, 10), flat).items():
        df[name] = col
    say(f"   [{source}] producto listo")

    # contexto: qué compra cada dispositivo / región (compras anteriores al inicio de la semana)
    cut = np.clip(np.searchsorted(np.array(CTX_CUTS, dtype="datetime64[ns]"), ts_req, side="right") - 1, 0, None)
    for key in ("device", "region"):
        code, tables = ctx[key]
        k = np.array([code.get(v, -1) for v in ctx_val[key]])
        for name in tables[0]:
            col = np.zeros(n * 10, dtype=np.float32)
            for c in np.unique(cut):
                rows = np.flatnonzero((cut == c) & (k >= 0))
                if len(rows):
                    col.reshape(n, 10)[rows] = tables[c][name][k[rows][:, None], shown[rows]]
            df[name] = col

    # sesión
    sf = session_features(items, sess, shown, cuts).reshape(n * 10, -1)
    for i, name in enumerate(feat5.SESSION):
        df[name] = sf[:, i]
    say(f"   [{source}] sesión lista")

    is_new = np.repeat(sess.is_new_user.to_numpy(bool), 10) if live else None
    return _finish(df, ui, items, flat, is_new)


def _finish(df: pd.DataFrame, ui: pd.DataFrame, items: feat5.Items, flat: np.ndarray,
            is_new: np.ndarray | None) -> pd.DataFrame:
    """Parte común a todas las filas: historia del usuario (SQL) → features de usuario, afinidad y par.
    `is_new`: marca is_new_user de la petición por fila; None = histórico (nuevo = sin eventos anteriores)."""
    live = is_new is not None
    # usuario × producto (SQL) → features de usuario y afinidad
    df = df.merge(ui, on=["request_id", "product_id"], how="left")
    counts = ["up_views", "up_carts", "up_buys", "up_c_clicks", "up_c_buys", "u_sub_w_7d",
              *[c for c in feat5.USER_AGG if c not in ("u_logp", "u_buy_logp", "u_logp_sd")]]
    df[counts] = df[counts].fillna(0).astype("float32")
    tsn = df.timestamp.dt.tz_localize(None)
    for name, col in feat5.user_feats({k: df[k].to_numpy() for k in feat5.USER_AGG}, items.logp[flat]).items():
        df[name] = col
    df["is_new_user"] = (is_new if live else (df.u_events == 0).to_numpy()).astype("int8")
    df["previous_purchase_count"] = df.up_buys + df.up_c_buys
    df["already_purchased"] = (df.previous_purchase_count > 0).astype("int8")
    df["up_days_since_seen"] = _days_since(tsn, df.pop("up_last_seen"))
    # como al servir: los pares de una sola vista no están en el estado (feat5.stored_pair)
    lost = ~feat5.stored_pair(df.up_views, df.up_carts, df.up_c_clicks, df.previous_purchase_count)
    df.loc[lost, "up_views"] = 0.0
    df.loc[lost, "up_days_since_seen"] = np.nan
    df["u_days_since_last"] = _days_since(tsn, df.pop("u_last_ts"))      # solo para comparar con v2

    for col in ("request_id", "user_id", "product_id", "category", "subcategory", "brand", "attr1", "attr2",
                "page", "device", "region", "source", "served_by"):
        df[col] = df[col].astype("string")
    for col in df.select_dtypes("float64").columns.difference(["propensity", "sample_weight", "revenue_eur"]):
        df[col] = df[col].astype("float32")
    return df


def load_org(until: str, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(peticiones, pares) de las filas orgánicas. Pares: request_id, product_id, pos."""
    def pairs() -> pd.DataFrame:
        parts = [pd.DataFrame(_query(org_pairs_sql(until, f"cityHash64(user_id::String) % {LIVE_CHUNKS} = {k}")))
                 for k in range(LIVE_CHUNKS)]
        return pd.concat(parts, ignore_index=True)[["request_id", "product_id", "pos"]]

    def requests() -> pd.DataFrame:
        parts = []
        for k in range(LIVE_CHUNKS):   # `context` y `recent_views` no pasan por un JOIN: se filtra con IN
            part = f"cityHash64(user_id::String) % {LIVE_CHUNKS} = {k}"
            parts.append(pd.DataFrame(_query(f"""SELECT request_id::String AS request_id,
                    toString(ts::DateTime64(3)) AS ts, user_id::String AS user_id, is_new_user::Bool AS is_new_user,
                    context, recent_views FROM shop.requests
                WHERE {part} AND request_id::String IN (SELECT request_id FROM ({org_pairs_sql(until, part)}))""")))
        df = pd.concat(parts, ignore_index=True).drop_duplicates("request_id")
        df["recent_views"] = df.recent_views.apply(lambda v: list(v) if v is not None else [])
        return df.sort_values(["ts", "request_id"]).reset_index(drop=True)

    return _cached("org_sessions.pkl", requests, refresh), _cached("org_pairs.parquet", pairs, refresh)


def _frame_org(sess: pd.DataFrame, pairs: pd.DataFrame, items: feat5.Items, ui: pd.DataFrame, tl: Timeline,
               fbt: FeedbackTimeline, ctx: dict, daily: pd.DataFrame, updates: pd.DataFrame, cuts: list[str],
               say) -> pd.DataFrame:
    """Filas orgánicas con las mismas features que las demás. Aquí no hay carrusel: cada petición trae sus
    compras orgánicas (target 1) y ~20 productos al azar (target 0), todos con la posición de referencia."""
    pairs = pairs[pairs.product_id.isin(items.idx) & pairs.request_id.isin(set(sess.request_id))]
    order = {r: i for i, r in enumerate(sess.request_id)}
    ri = pairs.request_id.map(order).to_numpy()
    by = np.argsort(ri, kind="stable")
    pairs, ri = pairs.iloc[by].reset_index(drop=True), ri[by]
    flat = pairs.product_id.map(items.idx).to_numpy()
    n = len(pairs)
    ts_req = sess.ts.to_numpy().astype("datetime64[ns]")
    ts_row = ts_req[ri]
    pos = pairs.pos.to_numpy(dtype="int8")
    df = pd.DataFrame({
        "request_id": pairs.request_id.to_numpy(), "user_id": sess.user_id.to_numpy()[ri],
        "product_id": items.ids[flat], "timestamp": pd.to_datetime(ts_row, utc=True),
        "position": np.ones(n, dtype="int8"), "target_purchase": pos, "target_click": pos,
        "propensity": 1.0, "sample_weight": 1.0, "revenue_eur": 0.0, "source": "org", "served_by": "organic",
    })
    ctx_val = {}
    for key in ("page", "device", "region"):
        ctx_val[key] = sess.context.apply(lambda c: (c or {}).get(key) or "UNKNOWN").to_numpy()
        df[key] = ctx_val[key][ri]
    for col, arr in (("category", "cat"), ("subcategory", "sub"), ("brand", "brand")):
        df[col] = np.array(list(items.codes[arr]), dtype=object)[getattr(items, arr)[flat]]
    df["attr1"], df["attr2"] = items.attr1[flat], items.attr2[flat]
    price = items.base_price[flat].copy()
    if len(updates):   # precio en el instante de la petición
        up = updates.assign(ts=pd.to_datetime(updates.ts).astype("datetime64[ns]")).sort_values("ts")
        left = pd.DataFrame({"ts": ts_row.astype("datetime64[ns]"), "product_id": items.ids[flat], "row": np.arange(n)})
        m = pd.merge_asof(left.sort_values("ts"), up[["ts", "product_id", "price_eur"]], on="ts", by="product_id",
                          allow_exact_matches=False).sort_values("row")
        got = m.price_eur.to_numpy(dtype=float)
        price = np.where(np.isnan(got), price, got)
    df["price_eur"] = price.astype("float32")
    df["it_discount"] = np.clip(1.0 - price / items.base_price[flat], 0.0, 1.0)
    df["it_price_vs_sub"] = items.price_vs_sub[flat]
    df["it_age_days"] = (ts_row - items.created[flat]) / np.timedelta64(1, "D")

    # producto: popularidad hasta el último tramo completo
    edge = tl.edge(ts_req.astype(np.int64))[ri]
    uniq, row_u = np.unique(edge, return_inverse=True)
    cols = {name: np.zeros(n, dtype=np.float32) for name in feat5.ITEM_POP + ("it_buys_7d", "it_views_7d", "it_buys_1d")}
    for lo in range(0, len(uniq), 128):
        at = uniq[lo:lo + 128]
        sel = (row_u >= lo) & (row_u < lo + len(at))
        counts = tl.counts(at)
        block = feat5.item_pop(counts, items.sub)
        for name in feat5.ITEM_POP:
            cols[name][sel] = block[name][row_u[sel] - lo, flat[sel]]
        for name, key in (("it_buys_7d", "buys_7d"), ("it_views_7d", "views_7d"), ("it_buys_1d", "buys_1d")):
            cols[name][sel] = counts[key][row_u[sel] - lo, flat[sel]]
    for name, col in cols.items():
        df[name] = col
    df["it_trend"] = np.log1p(df.it_buys_1d * 7) - np.log1p(df.it_buys_7d)

    # producto: carrusel del histórico (todo) y nuestro carrusel (hasta el último tramo de 10 min)
    days = pd.date_range(bd.HISTORY_FROM, "2026-11-15", freq="D")
    d = daily[daily.product_id.isin(items.idx)]
    tot = d.groupby("product_id")[["c_impr", "c_clicks", "c_buys"]].sum().reindex(items.ids, fill_value=0.0)
    del days
    rates = feat5.carousel_rates(tot.c_impr.to_numpy(), tot.c_clicks.to_numpy(), tot.c_buys.to_numpy())
    for name, m in rates.items():
        df[name] = m[0, flat]
    for name, col in fbt.lifts(edge - H_LIVE, flat).items():
        df[name] = col

    # contexto: compras por dispositivo / región anteriores al inicio del stream
    for key in ("device", "region"):
        code, tables = ctx[key]
        k = np.array([code.get(v, -1) for v in ctx_val[key]])[ri]
        for name, m in tables[len(CTX_CUTS) - 1].items():
            df[name] = np.where(k >= 0, m[np.clip(k, 0, None), flat], 0.0).astype(np.float32)
    say("   [org] producto y contexto listos")

    # sesión: la misma función que el servicio, una vez por petición
    which = np.clip(np.searchsorted(np.array(cuts, dtype="datetime64[ns]"), ts_req, side="right") - 1, 0, None)
    nbs = {w: neighbors(cuts[w], items) for w in np.unique(which)}
    sf = np.zeros((n, len(feat5.SESSION)), dtype=np.float32)
    start = np.searchsorted(ri, np.arange(len(sess) + 1))
    for r, (views, w, now) in enumerate(zip(sess.recent_views, which, ts_req)):
        a, b = start[r], start[r + 1]
        if a == b:
            continue
        v, secs = feat5.parse_views(items, views, now)
        block = feat5.session_block(items, v, secs, nbs[w])
        p = flat[a:b]
        for j, name in enumerate(feat5.SESSION):
            sf[a:b, j] = block[name][p]
    for j, name in enumerate(feat5.SESSION):
        df[name] = sf[:, j]
    say("   [org] sesión lista")
    return _finish(df, ui, items, flat, sess.is_new_user.to_numpy(bool)[ri])


def build(refresh: bool = False, verbose: bool = True, parts: tuple[str, ...] = ("hist", "live")) -> dict[str, pd.DataFrame]:
    say = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    until = live_until(refresh)
    cat = load_catalog(refresh)
    items = feat5.Items(cat.assign(created_at=pd.to_datetime(cat.created_at)))
    updates = load_updates(refresh)
    say(f"catálogo: {items.n:,} productos ({int((items.created >= np.datetime64(LIVE_START)).sum())} nuevos de hoy) · "
        f"parte en vivo hasta {until}  [{time.time() - t0:.0f}s]")
    hourly = _chunks("item_hourly", 4, lambda k: item_hourly_sql(k, 4), refresh, say)
    live10 = _cached("item_live10.parquet", lambda: pd.DataFrame(_query(ITEM_LIVE10.format(until=until))), refresh)
    tl = Timeline(items, hourly, live10, until)
    fb = _cached("feedback10.parquet", lambda: pd.DataFrame(_query(FEEDBACK10.format(team=TEAM, until=until))), refresh)
    fbt = FeedbackTimeline(items, fb, until)
    ctx = context_tables(items, refresh, until)
    daily = bd.load_item_daily()
    say(f"recuentos por producto y contexto listos  [{time.time() - t0:.0f}s]")
    out = {}
    if "hist" in parts:
        sess = bd.load_sessions()
        sess["ts"] = pd.to_datetime(sess.ts)
        ui = _chunks("ui_hist", CHUNKS, lambda k: user_item_sql(k, CHUNKS, live=False), refresh, say)
        say(f"hist: {len(sess):,} peticiones ({(sess.clicked != '').sum():,} con clic, "
            f"{(sess.purchased != '').sum():,} con compra)  [{time.time() - t0:.0f}s]")
        out["hist"] = _frame(sess, "hist", items, ui, tl, fbt, ctx, daily, updates, WEEKS, say)
    if "live" in parts:
        sess = load_live_sessions(until, refresh)
        sess["ts"] = pd.to_datetime(sess.ts)
        ui = _chunks("ui_live", LIVE_CHUNKS, lambda k: user_item_sql(k, LIVE_CHUNKS, live=True, until=until),
                     refresh, say)
        say(f"live: {len(sess):,} peticiones ({(sess.clicked != '').sum():,} con clic, "
            f"{(sess.purchased != '').sum():,} con compra)  [{time.time() - t0:.0f}s]")
        out["live"] = _frame(sess, "live", items, ui, tl, fbt, ctx, daily, updates, live_cuts(until), say)
    if "org" in parts:
        sess, pairs = load_org(until, refresh)
        sess["ts"] = pd.to_datetime(sess.ts)
        ui = _chunks("ui_org", LIVE_CHUNKS,
                     lambda k: user_item_sql(k, LIVE_CHUNKS, live=True, until=until, org=True), refresh, say)
        say(f"org: {len(sess):,} peticiones con compra orgánica en {ORG_WINDOW} min · {len(pairs):,} pares "
            f"({int(pairs.pos.sum()):,} compras)  [{time.time() - t0:.0f}s]")
        out["org"] = _frame_org(sess, pairs, items, ui, tl, fbt, ctx, daily, updates, live_cuts(until), say)
    for name, df in out.items():
        say(f"{name}: {len(df):,} filas × {df.shape[1]} columnas  [{time.time() - t0:.0f}s]")
    return out


def split(parts: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """hist → train / validation / test por fecha (los mismos cortes que v2); live → fichero aparte."""
    out = {}
    if "hist" in parts:
        df = parts["hist"]
        v, t = pd.Timestamp(VALID_FROM, tz="UTC"), pd.Timestamp(TEST_FROM, tz="UTC")
        out.update({"train": df[df.timestamp < v].reset_index(drop=True),
                    "validation": df[(df.timestamp >= v) & (df.timestamp < t)].reset_index(drop=True),
                    "test": df[df.timestamp >= t].reset_index(drop=True)})
    if "live" in parts:
        out["live"] = parts["live"].sort_values(["timestamp", "request_id", "position"]).reset_index(drop=True)
    if "org" in parts:
        out["org"] = parts["org"].sort_values(["timestamp", "request_id"]).reset_index(drop=True)
    return out


def save(parts: dict[str, pd.DataFrame], out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        part.to_parquet(out / f"{name}_v5.parquet", index=False)
    return out


# ── cold start simulado (igual que en v2, con las columnas de v5) ────────────────────────────────────
_COLD_ZERO = ["u_hist_conf", "u_buy_ratio", *feat5.AFFINITY, "up_views", "up_carts", "up_c_clicks",
              "u_events", "u_buys", "u_sub_w", "u_brand_w", "u_sub_w_7d", "u_sub_buys", "u_brand_buys"]
_COLD_NAN = ["u_logp_diff", "u_buy_logp_diff", "u_logp_z", "up_days_since_seen", "u_days_since_last"]


def simulate_cold(df: pd.DataFrame, share: float, seed: int = 0) -> pd.DataFrame:
    """Copia de `df` en la que `share` de las peticiones del HISTÓRICO quedan como si el usuario fuera nuevo
    (en el histórico casi no hay usuarios sin historia; en las filas en vivo los nuevos son de verdad)."""
    if share <= 0:
        return df
    hist = df[df.source == "hist"] if "source" in df.columns else df
    ids = hist.request_id.drop_duplicates()
    hidden = set(ids[np.random.default_rng(seed).random(len(ids)) < share])
    mask = df.request_id.isin(hidden).to_numpy()
    df = df.copy()
    df.loc[mask, [c for c in _COLD_ZERO if c in df.columns]] = 0
    df.loc[mask, [c for c in _COLD_NAN if c in df.columns]] = np.nan
    df.loc[mask, "is_new_user"] = 1
    return df
