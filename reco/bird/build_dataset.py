"""Construye el dataset v2 de Bird: una fila por (petición, producto enseñado) con features a fecha de la petición.

    python scripts/bird.py dataset          # → data/bird/v2/{train,validation,test}_v2.parquet

Qué peticiones: desde START (antes no hay historia suficiente para describir al usuario),
  - TODOS los carruseles con clic (ahí están todas las compras), peso 1
  - 1 de cada NOCLICK_MOD carruseles sin clic (elegidos por hash del request_id), peso NOCLICK_MOD
  `sample_weight` deshace el muestreo: con los pesos, la tasa de compra es la real del histórico.

Sin fugas temporales. Cada bloque de features solo mira el pasado de su petición:
  usuario × producto   SQL en RawTree: eventos del usuario con ts < ts de la petición (exacto al milisegundo)
  producto             contadores diarios acumulados hasta el día ANTERIOR a la petición
  sesión               recent_views de la propia petición (+ vecinos calculados con historia anterior a su semana)
  contexto             page / device / region de la petición

Reparto cronológico: train < VALID_FROM ≤ validation < TEST_FROM ≤ test (TEST_FROM = corte de reco.evaluate).
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import features
from ..client import query
from ..config import DATA_DIR
from ..features import _ATTR_SQL, ATTRS

START, VALID_FROM, TEST_FROM = "2026-09-26", "2026-10-31", "2026-11-07"
HISTORY_FROM, HISTORY_TO = "2026-09-15", "2026-11-14"
NOCLICK_MOD = 25
CHUNKS = 32            # la consulta usuario × producto se parte por hash de usuario
DECAY = 0.6            # peso de cada vista reciente: la última 1, la anterior 0,6, ...
PRIOR_IMPRESSIONS = 200.0
WEEKS = ["2026-09-26", "2026-10-03", "2026-10-10", "2026-10-17", "2026-10-24", "2026-10-31", "2026-11-07"]
RAW = DATA_DIR / "bird" / "raw"
OUT = DATA_DIR / "bird" / "v2"

SELECTED = (f"ts >= '{START}' AND (clicked::String != '' OR cityHash64(request_id::String) % {NOCLICK_MOD} = 0)")

CATALOG = f"""(SELECT product_id::String AS product_id, category::String AS category,
        subcategory::String AS subcategory, brand::String AS brand, price_eur::Float64 AS price_eur,
        log(1 + price_eur::Float64) AS logp, toString(created_at::DateTime64(3)) AS created_at,
        arrayFilter(a -> position(description::String, a) > 0, {_ATTR_SQL}) AS attrs
    FROM shop.catalog)"""

BUY = "('purchase', 'c_purchase')"


def user_item_sql(k: int, chunks: int = CHUNKS) -> str:
    """Historia del usuario ANTES de cada petición, resumida frente a cada producto enseñado.
    Eventos: orgánicos (vista 1, carrito 3, compra 5) + clics y compras de carruseles anteriores."""
    part = f"cityHash64(user_id::String) % {chunks} = {k}"
    events = f"""(
        SELECT user_id::String AS user_id, product_id::String AS product_id, ts::DateTime64(3) AS ts,
               event_type::String AS kind,
               multiIf(event_type::String = 'purchase', 5, event_type::String = 'add_to_cart', 3, 1) AS w
        FROM shop.history_events WHERE {part}
        UNION ALL
        SELECT user_id::String, clicked::String, ts::DateTime64(3),
               if(purchased::String != '', 'c_purchase', 'c_click'), if(purchased::String != '', 5, 1)
        FROM shop.history_sessions WHERE clicked::String != '' AND {part})"""
    return f"""
SELECT s.request_id AS request_id, s.product_id AS product_id,
  countIf(e.product_id = s.product_id AND e.kind = 'view') AS up_views,
  countIf(e.product_id = s.product_id AND e.kind = 'add_to_cart') AS up_carts,
  countIf(e.product_id = s.product_id AND e.kind = 'purchase') AS up_buys,
  countIf(e.product_id = s.product_id AND e.kind = 'c_click') AS up_c_clicks,
  countIf(e.product_id = s.product_id AND e.kind = 'c_purchase') AS up_c_buys,
  toString(maxIf(e.ts, e.product_id = s.product_id AND e.kind IN {BUY})) AS up_last_buy,
  toString(maxIf(e.ts, e.product_id = s.product_id)) AS up_last_seen,
  sumIf(e.w, ec.subcategory = s.subcategory) AS u_sub_w,
  sumIf(e.w, ec.brand = s.brand) AS u_brand_w,
  sumIf(e.w, ec.category = s.category) AS u_cat_w,
  sumIf(e.w, ec.subcategory = s.subcategory AND e.ts >= s.ts - INTERVAL 7 DAY) AS u_sub_w_7d,
  countIf(e.kind IN {BUY} AND ec.subcategory = s.subcategory) AS u_sub_buys,
  countIf(e.kind IN {BUY} AND ec.brand = s.brand) AS u_brand_buys,
  sum(e.w * length(arrayIntersect(ec.attrs, s.attrs))) AS u_attr_w,
  sum(e.w) AS u_w, count() AS u_events, countIf(e.kind IN {BUY}) AS u_buys,
  avg(ec.logp) AS u_logp, avgIf(ec.logp, e.kind IN {BUY}) AS u_buy_logp,
  toString(max(e.ts)) AS u_last_ts
FROM (
  SELECT h.request_id AS request_id, h.user_id AS user_id, h.ts AS ts, h.product_id AS product_id,
         c.category AS category, c.subcategory AS subcategory, c.brand AS brand, c.attrs AS attrs
  FROM (SELECT request_id::String AS request_id, user_id::String AS user_id, ts::DateTime64(3) AS ts,
               arrayJoin(shown::Array(String)) AS product_id
        FROM shop.history_sessions WHERE {SELECTED} AND {part}) AS h
  INNER JOIN {CATALOG} AS c ON c.product_id = h.product_id
) AS s
INNER JOIN {events} AS e ON e.user_id = s.user_id
INNER JOIN {CATALOG} AS ec ON ec.product_id = e.product_id
WHERE e.ts < s.ts
GROUP BY request_id, product_id"""


ITEM_DAILY = {
    "organic": """SELECT product_id::String AS product_id, toString(toDate(ts::DateTime64(3))) AS day,
        countIf(event_type::String = 'view') AS views, countIf(event_type::String = 'add_to_cart') AS carts,
        countIf(event_type::String = 'purchase') AS buys
        FROM shop.history_events GROUP BY product_id, day""",
    "shown": """SELECT arrayJoin(shown::Array(String)) AS product_id, toString(toDate(ts::DateTime64(3))) AS day,
        count() AS c_impr FROM shop.history_sessions GROUP BY product_id, day""",
    "clicked": """SELECT clicked::String AS product_id, toString(toDate(ts::DateTime64(3))) AS day,
        count() AS c_clicks, countIf(purchased::String != '') AS c_buys
        FROM shop.history_sessions WHERE clicked::String != '' GROUP BY product_id, day""",
}


def _cached(name: str, make, refresh: bool = False):
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / name
    if path.exists() and not refresh:
        return pickle.loads(path.read_bytes()) if path.suffix == ".pkl" else pd.read_parquet(path)
    obj = make()
    if path.suffix == ".pkl":
        path.write_bytes(pickle.dumps(obj))
    else:
        obj.to_parquet(path, index=False)
    return obj


def _query(sql: str, tries: int = 3, timeout: float = 900) -> list[dict]:
    for attempt in range(tries):
        try:
            return query(sql, timeout=timeout)
        except Exception as exc:  # red o límite puntual: reintenta
            if attempt == tries - 1:
                raise
            print(f"   reintento ({exc})")
            time.sleep(3)
    return []


def load_sessions(refresh: bool = False) -> pd.DataFrame:
    def make() -> pd.DataFrame:
        rows = _query(f"""SELECT request_id::String AS request_id, toString(ts::DateTime64(3)) AS ts,
            user_id::String AS user_id, context, recent_views, shown, propensity, clicked::String AS clicked,
            purchased::String AS purchased, revenue_eur::Float64 AS revenue_eur
            FROM shop.history_sessions WHERE {SELECTED}""")
        df = pd.DataFrame(rows)
        df["recent_views"] = df.recent_views.apply(lambda v: v or [])
        return df.sort_values(["ts", "request_id"]).reset_index(drop=True)
    return _cached("sessions.pkl", make, refresh)


def load_catalog(refresh: bool = False) -> pd.DataFrame:
    def make() -> pd.DataFrame:
        df = pd.DataFrame(_query(f"SELECT * FROM {CATALOG}"))
        order = {a: i for i, a in enumerate(ATTRS)}
        attrs = df["attrs"].apply(lambda v: sorted(v or [], key=order.get))
        df["attr1"] = attrs.str[0].fillna("UNKNOWN")
        df["attr2"] = attrs.str[1].fillna("UNKNOWN")
        return df.drop(columns="attrs").sort_values("product_id").reset_index(drop=True)
    return _cached("catalog.parquet", make, refresh)


def load_user_item(refresh: bool = False, verbose: bool = True) -> pd.DataFrame:
    parts = []
    for k in range(CHUNKS):
        t = time.time()
        part = _cached(f"user_item_{k:02d}.parquet", lambda: pd.DataFrame(_query(user_item_sql(k))), refresh)
        parts.append(part)
        if verbose and time.time() - t > 1:
            print(f"   usuario × producto {k + 1}/{CHUNKS}: {len(part):,} filas  {time.time() - t:.0f}s")
    return pd.concat(parts, ignore_index=True)


def load_item_daily(refresh: bool = False) -> pd.DataFrame:
    def make() -> pd.DataFrame:
        out = None
        for sql in ITEM_DAILY.values():
            df = pd.DataFrame(_query(sql))
            out = df if out is None else out.merge(df, on=["product_id", "day"], how="outer")
        return out.fillna(0)
    return _cached("item_daily.parquet", make, refresh)


# ── producto: contadores acumulados hasta el día anterior ────────────────────────────────────────────

def item_tables(daily: pd.DataFrame, cat: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DatetimeIndex]:
    """Para cada feature, matriz [día, producto] con el valor que se conocía al EMPEZAR ese día."""
    days = pd.date_range(HISTORY_FROM, HISTORY_TO, freq="D")
    pid = {p: i for i, p in enumerate(cat.product_id)}
    daily = daily[daily.product_id.isin(pid)]
    di = pd.to_datetime(daily.day).map({d: i for i, d in enumerate(days)}).to_numpy()
    pi = daily.product_id.map(pid).to_numpy()
    ok = ~np.isnan(di)
    di, pi = di[ok].astype(int), pi[ok]
    raw = {}
    for col in ("views", "carts", "buys", "c_impr", "c_clicks", "c_buys"):
        m = np.zeros((len(days), len(pid)))
        np.add.at(m, (di, pi), daily[col].to_numpy(dtype=float)[ok])
        raw[col] = m

    def before(m: np.ndarray, window: int | None) -> np.ndarray:
        cum = np.vstack([np.zeros((1, m.shape[1])), np.cumsum(m, axis=0)])   # cum[t] = suma de los días < t
        if window is None:
            return cum[:-1]
        lo = np.maximum(np.arange(len(days)) - window, 0)
        return cum[:-1] - cum[lo]

    views, carts, buys = (before(raw[c], None) for c in ("views", "carts", "buys"))
    impr, clicks, cbuys = (before(raw[c], None) for c in ("c_impr", "c_clicks", "c_buys"))
    buys_7d, buys_1d, views_7d = before(raw["buys"], 7), before(raw["buys"], 1), before(raw["views"], 7)
    share = lambda m: m / np.maximum(m.sum(axis=1, keepdims=True), 1)   # noqa: E731
    g_ctr = clicks.sum(axis=1, keepdims=True) / np.maximum(impr.sum(axis=1, keepdims=True), 1)
    g_cvr = cbuys.sum(axis=1, keepdims=True) / np.maximum(impr.sum(axis=1, keepdims=True), 1)
    sub = cat.subcategory.astype("category").cat.codes.to_numpy()
    sub_rank = np.zeros_like(buys_7d)
    for s in np.unique(sub):
        cols = np.flatnonzero(sub == s)
        sub_rank[:, cols] = (-buys_7d[:, cols]).argsort(axis=1, kind="stable").argsort(axis=1) + 1
    return {
        "it_buy_share": share(buys) * 1e4,          # cuota de las compras orgánicas (×10⁴)
        "it_view_share": share(views) * 1e4,
        "it_buy_rate": (buys + 1) / (views + 50),   # compras por vista
        "it_cart_rate": (carts + 1) / (views + 50),
        "it_buys_7d": buys_7d,
        "it_views_7d": views_7d,
        "it_buys_1d": buys_1d,
        "it_trend": np.log1p(buys_1d * 7) - np.log1p(buys_7d),
        "it_sub_rank": sub_rank,                    # puesto dentro de su subcategoría (ventas 7 días)
        "it_c_impr": np.log1p(impr),                # cuánto lo enseñó el carrusel viejo
        "it_c_ctr": (clicks + PRIOR_IMPRESSIONS * g_ctr) / (impr + PRIOR_IMPRESSIONS),
        "it_c_cvr": (cbuys + PRIOR_IMPRESSIONS * g_cvr) / (impr + PRIOR_IMPRESSIONS),
    }, days


# ── sesión: recent_views frente a cada producto enseñado ─────────────────────────────────────────────

def _neighbors(until: str, pid: dict[str, int]) -> dict[str, dict[int, list[tuple[int, float]]]]:
    if not features.folder(until).exists():
        features.build(until=until, verbose=False)
    nb = pd.read_parquet(features.folder(until) / "item_neighbors.parquet")
    out = {}
    for kind in ("buy", "view"):
        d = nb[nb.kind == kind]
        d = d.assign(nn=d.n / d.groupby("p").n.transform("max")).sort_values(["p", "nn"], ascending=[True, False])
        out[kind] = {pid[p]: [(pid[q], v) for q, v in zip(g.q, g.nn) if q in pid][:20]
                     for p, g in d.groupby("p", sort=False) if p in pid}
    return out


SESSION_COLS = ["n_recent_views", "recently_viewed", "sv_recency", "sv_count", "sv_same_sub_last",
                "sv_same_cat_last", "sv_same_brand_last", "sv_sub_share", "sv_cat_share", "sv_brand_share",
                "sv_attr_share", "sv_logp_diff", "view_to_purchase_score", "coview_score"]


def session_features(sess: pd.DataFrame, shown: np.ndarray, cat: pd.DataFrame) -> np.ndarray:
    """[petición, 10, len(SESSION_COLS)]."""
    pid = {p: i for i, p in enumerate(cat.product_id)}
    sub = cat.subcategory.astype("category").cat.codes.to_numpy()
    cat_ = cat.category.astype("category").cat.codes.to_numpy()
    brand = cat.brand.astype("category").cat.codes.to_numpy()
    logp = cat.logp.to_numpy()
    a1 = cat.attr1.astype("category").cat.codes.to_numpy()
    a2 = cat.attr2.astype("category").cat.codes.to_numpy()
    out = np.zeros((len(sess), 10, len(SESSION_COLS)))
    out[:, :, SESSION_COLS.index("sv_logp_diff")] = np.nan
    week = np.searchsorted(np.array(WEEKS, dtype="datetime64[ns]"), sess.ts.to_numpy().astype("datetime64[ns]"),
                           side="right") - 1
    nbs = {w: _neighbors(WEEKS[w], pid) for w in np.unique(week)}
    for r, (views, w) in enumerate(zip(sess.recent_views, week)):
        v = [pid[x["product_id"]] for x in views if isinstance(x, dict) and x.get("product_id") in pid]
        o = out[r]
        o[:, 0] = len(v)
        if not v:
            continue
        p = shown[r]
        v = np.array(v)
        rev = v[::-1]                               # la última vista primero
        eq = p[:, None] == rev[None, :]
        o[:, 1] = eq.any(axis=1)
        o[:, 2] = np.where(eq.any(axis=1), 1 / (eq.argmax(axis=1) + 1), 0)
        o[:, 3] = eq.sum(axis=1)
        last = rev[0]
        o[:, 4] = sub[p] == sub[last]
        o[:, 5] = cat_[p] == cat_[last]
        o[:, 6] = brand[p] == brand[last]
        o[:, 7] = (sub[p][:, None] == sub[v][None, :]).mean(axis=1)
        o[:, 8] = (cat_[p][:, None] == cat_[v][None, :]).mean(axis=1)
        o[:, 9] = (brand[p][:, None] == brand[v][None, :]).mean(axis=1)
        same = ((a1[p][:, None] == a1[v][None, :]) | (a1[p][:, None] == a2[v][None, :])).astype(float) + \
               ((a2[p][:, None] == a1[v][None, :]) | (a2[p][:, None] == a2[v][None, :]))
        o[:, 10] = same.mean(axis=1) / 2
        o[:, 11] = logp[p] - logp[v].mean()
        for col, kind in ((12, "buy"), (13, "view")):
            score: dict[int, float] = {}
            for i, x in enumerate(rev):
                for q, nn in nbs[w][kind].get(int(x), ()):
                    score[q] = score.get(q, 0.0) + DECAY ** i * nn
            if score:
                o[:, col] = [score.get(int(q), 0.0) for q in p]
    return out


# ── ensamblado ───────────────────────────────────────────────────────────────────────────────────────

def _days_since(ts: pd.Series, past: pd.Series) -> np.ndarray:
    past = pd.to_datetime(past, errors="coerce")
    days = (ts - past).dt.total_seconds() / 86400
    return days.where(past > pd.Timestamp("2000-01-01")).to_numpy(dtype=float)   # 1970 = "nunca"


def build(refresh: bool = False, verbose: bool = True) -> pd.DataFrame:
    say = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    sess = load_sessions(refresh)
    cat = load_catalog(refresh)
    sess["ts"] = pd.to_datetime(sess.ts)
    say(f"peticiones: {len(sess):,} ({(sess.clicked != '').sum():,} con clic, {(sess.purchased != '').sum():,} "
        f"con compra) · catálogo: {len(cat):,}  [{time.time() - t0:.0f}s]")
    pid = {p: i for i, p in enumerate(cat.product_id)}
    ok = sess.shown.apply(lambda s: len(s) == 10 and len(set(s)) == 10 and all(p in pid for p in s))
    if not ok.all():
        say(f"   se descartan {int((~ok).sum())} peticiones con carrusel incompleto o productos fuera de catálogo")
        sess = sess[ok].reset_index(drop=True)
    n = len(sess)
    shown = np.array([[pid[p] for p in s] for s in sess.shown])
    flat = shown.ravel()
    rep = lambda col: np.repeat(sess[col].to_numpy(), 10)   # noqa: E731

    df = pd.DataFrame({
        "request_id": rep("request_id"), "user_id": rep("user_id"),
        "product_id": cat.product_id.to_numpy()[flat],
        "timestamp": pd.to_datetime(rep("ts"), utc=True),
        "position": np.tile(np.arange(1, 11), n).astype("int8"),
    })
    df["target_purchase"] = (df.product_id.to_numpy() == rep("purchased")).astype("int8")
    df["target_click"] = (df.product_id.to_numpy() == rep("clicked")).astype("int8")
    df["propensity"] = np.concatenate([np.asarray(p, dtype=float) for p in sess.propensity])
    df["sample_weight"] = np.where(rep("clicked") != "", 1.0, float(NOCLICK_MOD))
    df["revenue_eur"] = np.where(df.target_purchase == 1, rep("revenue_eur"), 0.0)

    # contexto
    for key in ("page", "device", "region"):
        df[key] = np.repeat(sess.context.apply(lambda c: (c or {}).get(key) or "UNKNOWN").to_numpy(), 10)

    # producto (catálogo)
    for col in ("category", "subcategory", "brand", "attr1", "attr2"):
        df[col] = cat[col].to_numpy()[flat]
    df["price_eur"] = cat.price_eur.to_numpy()[flat].astype("float32")
    logp = cat.logp.to_numpy()
    sub_median = cat.groupby("subcategory").logp.transform("median").to_numpy()
    df["it_price_vs_sub"] = (logp - sub_median)[flat]
    created = pd.to_datetime(cat.created_at).to_numpy()[flat]
    df["it_age_days"] = (rep("ts") - created) / np.timedelta64(1, "D")

    # producto (popularidad hasta el día anterior)
    tables, days = item_tables(load_item_daily(refresh), cat)
    day = np.repeat(np.searchsorted(days.to_numpy(), sess.ts.dt.normalize().to_numpy()), 10)
    for name, m in tables.items():
        df[name] = m[day, flat]
    say(f"producto listo  [{time.time() - t0:.0f}s]")

    # sesión
    sf = session_features(sess, shown, cat).reshape(n * 10, -1)
    for i, name in enumerate(SESSION_COLS):
        df[name] = sf[:, i]
    say(f"sesión lista  [{time.time() - t0:.0f}s]")

    # usuario × producto (SQL)
    ui = load_user_item(refresh, verbose)
    df = df.merge(ui, on=["request_id", "product_id"], how="left")
    counts = ["up_views", "up_carts", "up_buys", "up_c_clicks", "up_c_buys", "u_sub_w", "u_brand_w", "u_cat_w",
              "u_sub_w_7d", "u_sub_buys", "u_brand_buys", "u_attr_w", "u_w", "u_events", "u_buys"]
    df[counts] = df[counts].fillna(0).astype("float32")
    ts = df.timestamp.dt.tz_localize(None)
    df["is_new_user"] = (df.u_events == 0).astype("int8")          # sin ninguna historia antes de la petición
    df["previous_purchase_count"] = df.up_buys + df.up_c_buys
    df["already_purchased"] = (df.previous_purchase_count > 0).astype("int8")
    df["days_since_last_purchase"] = _days_since(ts, df.pop("up_last_buy"))
    df["up_days_since_seen"] = _days_since(ts, df.pop("up_last_seen"))
    df["u_days_since_last"] = _days_since(ts, df.pop("u_last_ts"))
    w = df.u_w.where(df.u_w > 0)
    for part in ("sub", "brand", "cat"):
        df[f"u_{part}_share"] = (df[f"u_{part}_w"] / w).fillna(0)
    df["u_attr_share"] = (df.u_attr_w / (2 * w)).fillna(0)
    lp = np.log1p(df.price_eur.astype(float))
    df["u_logp_diff"] = lp - df.pop("u_logp")                      # precio frente a lo que suele mirar
    df["u_buy_logp_diff"] = lp - df.pop("u_buy_logp").where(df.u_buys > 0)   # ... y frente a lo que compra
    df = df.drop(columns=["u_w", "u_attr_w", "u_cat_w"])

    for col in ("request_id", "user_id", "product_id", "category", "subcategory", "brand", "attr1", "attr2",
                "page", "device", "region"):
        df[col] = df[col].astype("string")
    for col in df.select_dtypes("float64").columns.difference(["propensity", "sample_weight", "revenue_eur"]):
        df[col] = df[col].astype("float32")
    say(f"dataset: {len(df):,} filas × {df.shape[1]} columnas  [{time.time() - t0:.0f}s]")
    return df


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    v, t = pd.Timestamp(VALID_FROM, tz="UTC"), pd.Timestamp(TEST_FROM, tz="UTC")
    return {"train": df[df.timestamp < v].reset_index(drop=True),
            "validation": df[(df.timestamp >= v) & (df.timestamp < t)].reset_index(drop=True),
            "test": df[df.timestamp >= t].reset_index(drop=True)}


def save(parts: dict[str, pd.DataFrame], out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        part.to_parquet(out / f"{name}_v2.parquet", index=False)
    return out


# ── cold start ───────────────────────────────────────────────────────────────────────────────────────
# En el histórico solo el 0,2 % de las peticiones son de usuarios sin historia; en vivo son ~30 % (y en la
# ronda final llegan compradores nuevos). Para que el modelo aprenda a recomendar sin conocer al usuario,
# al ENTRENAR se le esconde la historia en una parte de las peticiones (nunca en validación ni test).
_COLD_ZERO = ["u_events", "u_buys", "u_sub_share", "u_brand_share", "u_cat_share", "u_attr_share", "u_sub_w",
              "u_brand_w", "u_sub_w_7d", "u_sub_buys", "u_brand_buys", "up_views", "up_carts", "up_c_clicks",
              "up_buys", "up_c_buys", "previous_purchase_count", "already_purchased"]
_COLD_NAN = ["u_days_since_last", "u_logp_diff", "u_buy_logp_diff", "up_days_since_seen",
             "days_since_last_purchase"]


def simulate_cold(df: pd.DataFrame, share: float, seed: int = 0) -> pd.DataFrame:
    """Copia de `df` en la que `share` de las peticiones quedan como si el usuario fuera nuevo."""
    if share <= 0:
        return df
    ids = df.request_id.drop_duplicates()
    hidden = set(ids[np.random.default_rng(seed).random(len(ids)) < share])
    mask = df.request_id.isin(hidden).to_numpy()
    df = df.copy()
    df.loc[mask, _COLD_ZERO] = 0
    df.loc[mask, _COLD_NAN] = np.nan
    df.loc[mask, "is_new_user"] = 1
    return df
