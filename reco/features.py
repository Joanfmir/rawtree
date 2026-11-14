"""Tablas preparadas para entrenar y servir. Se calculan con SQL en RawTree y se guardan en data/.

    from reco import features
    features.build(until="2026-11-07")   # offline: solo historia anterior al corte (para evaluar)
    features.build()                     # live: toda la historia + shop.events en vivo
    f = features.load(until="2026-11-07")   # dict nombre -> DataFrame

Tablas:
    item_features    una fila por producto: catálogo + popularidad + tasa de compra
    user_subcat      usuario -> sus 10 subcategorías favoritas (peso: compra 5, carrito 3, vista 1)
    user_brand       usuario -> sus 5 marcas favoritas
    user_purchases   usuario -> productos que ya compró (para no repetirlos)
    item_neighbors   producto -> productos relacionados (kind = 'buy': miró p y compró q en la sesión;
                     kind = 'view': vistos en la misma sesión)
    context_popular  (device, region) -> 50 productos más comprados (para usuarios nuevos)
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .catalog import CURRENT_CATALOG
from .client import query
from .config import DATA_DIR

EVENT_COLS = ("ts::DateTime64(3) AS ts, event_type::String AS event_type, user_id::String AS user_id, "
              "session_id::String AS session_id, product_id::String AS product_id, price_eur::Float64 AS price_eur, "
              "device::String AS device, region::String AS region")


def _events(until: str | None) -> str:
    """Fuente de eventos: historia antes del corte (offline) o historia + stream en vivo (live)."""
    if until:
        return f"(SELECT {EVENT_COLS} FROM shop.history_events WHERE ts < '{until}')"
    return (f"(SELECT {EVENT_COLS} FROM shop.history_events "
            f"UNION ALL SELECT {EVENT_COLS} FROM shop.events)")


def _last_day(until: str | None) -> str:
    return f"toDateTime64('{until}', 3)" if until else "now64(3)"


def queries(until: str | None) -> dict[str, str]:
    E, NOW = _events(until), _last_day(until)
    return {
        "item_features": f"""
            SELECT c.product_id AS product_id, c.category AS category, c.subcategory AS subcategory,
                   c.brand AS brand, c.price_eur AS price_eur, c.is_active AS is_active, c.is_new AS is_new,
                   s.views AS views, s.carts AS carts, s.purchases AS purchases,
                   s.views_7d AS views_7d, s.purchases_7d AS purchases_7d, s.purchases_1d AS purchases_1d,
                   (s.purchases + 1) / (s.views + 50) AS buy_rate
            FROM ({CURRENT_CATALOG}) AS c
            LEFT JOIN (
                SELECT product_id,
                       countIf(event_type = 'view') AS views, countIf(event_type = 'add_to_cart') AS carts,
                       countIf(event_type = 'purchase') AS purchases,
                       countIf(event_type = 'view' AND ts > {NOW} - INTERVAL 7 DAY) AS views_7d,
                       countIf(event_type = 'purchase' AND ts > {NOW} - INTERVAL 7 DAY) AS purchases_7d,
                       countIf(event_type = 'purchase' AND ts > {NOW} - INTERVAL 1 DAY) AS purchases_1d
                FROM {E} GROUP BY product_id
            ) AS s ON s.product_id = c.product_id""",
        "user_subcat": f"""
            SELECT e.user_id AS user_id, c.subcategory::String AS subcategory,
                   sum(multiIf(e.event_type = 'purchase', 5, e.event_type = 'add_to_cart', 3, 1)) AS w
            FROM {E} AS e INNER JOIN ({CURRENT_CATALOG}) AS c ON c.product_id = e.product_id
            GROUP BY user_id, subcategory ORDER BY w DESC LIMIT 10 BY user_id""",
        "user_brand": f"""
            SELECT e.user_id AS user_id, c.brand::String AS brand,
                   sum(multiIf(e.event_type = 'purchase', 5, e.event_type = 'add_to_cart', 3, 1)) AS w
            FROM {E} AS e INNER JOIN ({CURRENT_CATALOG}) AS c ON c.product_id = e.product_id
            GROUP BY user_id, brand ORDER BY w DESC LIMIT 5 BY user_id""",
        "user_purchases": f"""
            SELECT user_id, product_id, count() AS n, max(ts) AS last_ts
            FROM {E} WHERE event_type = 'purchase' GROUP BY user_id, product_id""",
        "item_neighbors": f"""
            SELECT p, q, n, 'buy' AS kind FROM (
                SELECT v.product_id AS p, b.product_id AS q, count() AS n
                FROM (SELECT session_id, product_id, ts FROM {E} WHERE event_type = 'view') AS v
                INNER JOIN (SELECT session_id, product_id, ts FROM {E} WHERE event_type IN ('purchase', 'add_to_cart')) AS b
                  ON v.session_id = b.session_id
                WHERE v.ts <= b.ts
                GROUP BY p, q ORDER BY n DESC LIMIT 30 BY p)
            UNION ALL
            SELECT p, q, n, 'view' AS kind FROM (
                SELECT a.product_id AS p, b.product_id AS q, count() AS n
                FROM (SELECT session_id, product_id FROM {E} WHERE event_type = 'view') AS a
                INNER JOIN (SELECT session_id, product_id FROM {E} WHERE event_type = 'view') AS b
                  ON a.session_id = b.session_id
                WHERE a.product_id != b.product_id
                GROUP BY p, q ORDER BY n DESC LIMIT 30 BY p)""",
        "context_popular": f"""
            SELECT device, region, product_id, count() AS n
            FROM {E} WHERE event_type = 'purchase' AND ts > {NOW} - INTERVAL 14 DAY
            GROUP BY device, region, product_id ORDER BY n DESC LIMIT 50 BY device, region""",
    }


def folder(until: str | None) -> Path:
    return DATA_DIR / (f"features_until_{until}" if until else "features_live")


def build(until: str | None = None, only: list[str] | None = None, verbose: bool = True) -> dict[str, pd.DataFrame]:
    out_dir = folder(until)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name, sql in queries(until).items():
        if only and name not in only:
            continue
        t = time.time()
        df = pd.DataFrame(query(sql, timeout=900))
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        tables[name] = df
        if verbose:
            print(f"  {name:16} {len(df):>9,} filas  {time.time() - t:5.1f}s")
    return tables


def load(until: str | None = None) -> dict[str, pd.DataFrame]:
    d = folder(until)
    if not d.exists():
        raise FileNotFoundError(f"No hay features en {d}. Ejecuta: python scripts/build_features.py"
                                + (f" --until {until}" if until else ""))
    return {p.stem: pd.read_parquet(p) for p in sorted(d.glob("*.parquet"))}
