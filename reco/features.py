"""Tablas preparadas para entrenar y servir. Se calculan con SQL en RawTree y se guardan en data/.

    from reco import features
    features.build(until="2026-11-07")   # offline: solo historia anterior al corte (para evaluar)
    features.build()                     # live: toda la historia + shop.events en vivo
    f = features.load(until="2026-11-07")   # dict nombre -> DataFrame

Tablas:
    item_live        solo live: ventanas cortas, precio/stock actual y trend_score (compras 7d/h)
    item_features    una fila por producto: catálogo + popularidad + tasa de compra
    user_subcat      usuario -> sus 10 subcategorías favoritas (peso: compra 5, carrito 3, vista 1)
    user_brand       usuario -> sus 5 marcas favoritas
    user_purchases   usuario -> productos que ya compró (orgánico + carrusel; la recompra es la señal más fuerte)
    item_neighbors   producto -> productos relacionados (kind = 'buy': miró p y compró q en la sesión;
                     kind = 'view': vistos en la misma sesión)
    user_items       hasta 3 pares útiles por usuario; vistas/carritos/compras/clics por origen
    item_feedback    (solo en vivo) producto × posición en nuestro carrusel: impresiones, clics, compras
    user_stats       usuario -> nº de eventos, nº de compras y nivel de precio (log) que mira / compra
    user_cat         usuario -> peso por categoría (compra 5, carrito 3, vista 1)
    item_attrs       producto -> sus 2 atributos, sacados del texto de la descripción (20 posibles: "bold colours",
                     "designed for kids"...). Los productos nuevos también los tienen: sirve para cold start
    user_attr        usuario -> peso por atributo (compra 5, carrito 3, vista 1)
    context_popular  (device, region) -> 50 productos más comprados (para usuarios nuevos)
"""
from __future__ import annotations

import time
import os
import tempfile
from pathlib import Path

import pandas as pd

from .catalog import CURRENT_CATALOG
from .client import query
from .config import DATA_DIR

EVENT_COLS = ("ts::DateTime64(3) AS ts, event_type::String AS event_type, user_id::String AS user_id, "
              "session_id::String AS session_id, product_id::String AS product_id, price_eur::Float64 AS price_eur, "
              "device::String AS device, region::String AS region")


# Los 20 atributos que aparecen en shop.catalog.description (cada producto tiene 2)
ATTRS = ["compact and lightweight", "a sporty style", "organic ingredients", "a family-size format",
         "wireless connectivity", "smart app control", "a luxury feel", "bold colours", "a minimalist design",
         "built for professionals", "great value for money", "whisper-quiet operation", "premium materials",
         "designed for kids", "made from recycled materials", "a vintage look", "a waterproof shell",
         "extra durability", "a handmade finish", "a travel-ready build"]
_ATTR_SQL = "[" + ", ".join(f"'{a}'" for a in ATTRS) + "]"
# producto -> atributo (una fila por atributo), con la descripción más reciente
ITEM_ATTRS = f"""
SELECT product_id, arrayJoin(arrayFilter(a -> position(description, a) > 0, {_ATTR_SQL})) AS attr
FROM (
    SELECT product_id, argMax(description, v) AS description
    FROM (SELECT product_id::String AS product_id, description::String AS description, '' AS v FROM shop.catalog
          UNION ALL
          SELECT product_id::String, description::String, toString(ts) FROM shop.catalog_updates)
    GROUP BY product_id
)"""


def tag(until: str) -> str:
    """Nombre de carpeta/fichero para un corte: '2026-11-07' → igual; '2026-11-14 15:00:00' → '2026-11-14T150000'."""
    return until.replace(" ", "T").replace(":", "")


def _events(until: str | None) -> str:
    """Fuente de eventos: todo lo anterior al corte (offline) o historia + stream en vivo (live).
    Con corte también entra shop.events (vacía antes del 14/11): así un corte de hoy ve el stream hasta esa hora."""
    if until:
        return (f"(SELECT {EVENT_COLS} FROM shop.history_events WHERE ts < '{until}' "
                f"UNION ALL SELECT {EVENT_COLS} FROM shop.events WHERE ts < '{until}')")
    return (f"(SELECT {EVENT_COLS} FROM shop.history_events "
            f"UNION ALL SELECT {EVENT_COLS} FROM shop.events)")


def _purchases(until: str | None) -> str:
    """Todas las compras conocidas: orgánicas (events) + del carrusel (history_sessions y, en vivo,
    team_NN.feedback cruzado con shop.requests para saber el usuario)."""
    from .config import TEAM
    E = _events(until)
    cut = f" AND ts < '{until}'" if until else ""
    sql = (f"SELECT user_id, product_id, ts, 'organic' AS source FROM {E} WHERE event_type = 'purchase' "
           f"UNION ALL SELECT user_id::String, purchased::String, ts::DateTime64(3), 'carousel' "
           f"FROM shop.history_sessions WHERE purchased::String != ''{cut}")
    fb_cut = f" AND f.ts::DateTime64(3) < '{until}'" if until else ""
    sql += (f" UNION ALL SELECT r.user_id::String, f.purchased::String, f.ts::DateTime64(3), 'carousel' "
            f"FROM {TEAM}.feedback AS f INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String "
            f"WHERE f.purchased::String != ''{fb_cut}")
    return sql


def user_items_query(until: str | None = None) -> str:
    """Tres pares útiles por usuario; contadores separados por origen.

    Tabla truncada para generar candidatos, nunca lista exhaustiva de compras
    (para excluir comprados debe mantenerse user_purchases). Sin consultar al servir.
    El corte offline se aplica a ambas fuentes y no incorpora eventos en vivo.
    """
    from .config import TEAM
    corte = f" AND ts::DateTime64(3) < '{until}'" if until else ""
    sesiones = f"""SELECT user_id::String AS user_id, p AS product_id,
        0 AS vistas, 0 AS carritos, 0 AS compras_organicas,
        countIf(purchased::String = p) AS compras_carrusel,
        countIf(clicked::String = p) AS clics_carrusel, max(ts::DateTime64(3)) AS ultimo_ts
        FROM shop.history_sessions
        ARRAY JOIN arrayDistinct([clicked::String, purchased::String]) AS p
        WHERE p != ''{corte} GROUP BY user_id, product_id"""
    if True:   # nuestro feedback también cuenta con corte (solo lo anterior al corte)
        fb_cut = f" AND f.ts::DateTime64(3) < '{until}'" if until else ""
        sesiones += f""" UNION ALL SELECT r.user_id::String, p, 0, 0, 0,
            countIf(f.purchased::String = p), countIf(f.clicked::String = p), max(f.ts::DateTime64(3))
            FROM {TEAM}.feedback f INNER JOIN shop.requests r
            ON r.request_id::String = f.request_id::String
            ARRAY JOIN arrayDistinct([f.clicked::String, f.purchased::String]) AS p
            WHERE p != '' AND f.request_id::String != '__init__'{fb_cut}
            GROUP BY r.user_id::String, p"""
    return f"""SELECT user_id, product_id, sum(vistas) AS vistas, sum(carritos) AS carritos,
        sum(compras_organicas) AS compras_organicas, sum(compras_carrusel) AS compras_carrusel,
        sum(clics_carrusel) AS clics_carrusel, max(ultimo_ts) AS ultimo_ts
        FROM (
            SELECT user_id, product_id, countIf(event_type = 'view') AS vistas,
                countIf(event_type = 'add_to_cart') AS carritos,
                countIf(event_type = 'purchase') AS compras_organicas,
                0 AS compras_carrusel, 0 AS clics_carrusel, max(ts) AS ultimo_ts
            FROM {_events(until)} GROUP BY user_id, product_id
            UNION ALL {sesiones}
        ) GROUP BY user_id, product_id
        HAVING vistas >= 2 OR carritos > 0 OR compras_organicas > 0
            OR compras_carrusel > 0 OR clics_carrusel > 0
        ORDER BY user_id,
            (compras_carrusel > 0 AND compras_organicas = 0) DESC,
            (carritos > 0 AND compras_organicas + compras_carrusel = 0) DESC,
            (clics_carrusel > 0 AND compras_organicas + compras_carrusel = 0) DESC,
            vistas DESC, ultimo_ts DESC, product_id
        LIMIT 3 BY user_id"""


def _last_day(until: str | None) -> str:
    return f"toDateTime64('{until}', 3)" if until else "now64(3)"


def queries(until: str | None) -> dict[str, str]:
    E, NOW = _events(until), _last_day(until)
    tables = {
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
        "user_items": user_items_query(until),
        "user_purchases": f"""
            SELECT user_id, product_id, count() AS n, max(ts) AS last_ts,
                   countIf(source = 'carousel') AS n_carousel
            FROM ({_purchases(until)})
            GROUP BY user_id, product_id""",
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
        "user_stats": f"""
            SELECT user_id, count() AS n_events, countIf(event_type = 'purchase') AS n_buys,
                   avgIf(log(price_eur + 1), price_eur > 0) AS avg_logp,
                   avgIf(log(price_eur + 1), event_type = 'purchase' AND price_eur > 0) AS buy_logp
            FROM {E} GROUP BY user_id""",
        "user_cat": f"""
            SELECT e.user_id AS user_id, c.category::String AS category,
                   sum(multiIf(e.event_type = 'purchase', 5, e.event_type = 'add_to_cart', 3, 1)) AS w
            FROM {E} AS e INNER JOIN ({CURRENT_CATALOG}) AS c ON c.product_id = e.product_id
            GROUP BY user_id, category""",
        "item_attrs": ITEM_ATTRS,
        "user_attr": f"""
            SELECT e.user_id AS user_id, a.attr AS attr,
                   sum(multiIf(e.event_type = 'purchase', 5, e.event_type = 'add_to_cart', 3, 1)) AS w
            FROM {E} AS e INNER JOIN ({ITEM_ATTRS}) AS a ON a.product_id = e.product_id
            GROUP BY user_id, attr""",
        "context_popular": f"""
            SELECT device, region, product_id, count() AS n
            FROM {E} WHERE event_type = 'purchase' AND ts > {NOW} - INTERVAL 14 DAY
            GROUP BY device, region, product_id ORDER BY n DESC LIMIT 50 BY device, region""",
    }
    if until is None:
        tables["item_live"] = ITEM_LIVE
        tables["item_feedback"] = item_feedback_query()
    return tables


# Tablas que pueden fallar sin tumbar el refresco: si su consulta falla se conserva el Parquet anterior.
OPTIONAL = {"item_feedback"}


def item_feedback_query() -> str:
    """Lo que pasó con cada producto en NUESTRO carrusel, por posición: impresiones, clics y compras.
    Es lo único que mide directamente lo que premia el simulador (la historia orgánica no lo hace)."""
    from .config import TEAM
    return f"""
        SELECT x.1 AS product_id, x.2 AS slot, count() AS impressions,
               countIf(clicked::String = x.1) AS clicks, countIf(purchased::String = x.1) AS purchases
        FROM (SELECT arrayJoin(arrayZip(shown::Array(String), arrayEnumerate(shown::Array(String)))) AS x,
                     clicked, purchased
              FROM {TEAM}.feedback WHERE request_id::String != '__init__')
        GROUP BY product_id, slot"""


def feedback_lift(fb: pd.DataFrame | None, prior: float = 3.0) -> dict[str, float]:
    """{producto: lift}: cuánto más (o menos) de lo esperado convierte un producto en nuestro carrusel.

    valor = clics + 2·compras. Esperado = lo que habría conseguido un producto medio en esas mismas
    posiciones (así no se premia a quien salió arriba). lift = (valor + prior) / (esperado + prior):
    con pocas impresiones se queda en ~1 y solo se aleja cuando hay evidencia. Acotado a [0,3, 3]."""
    if fb is None or not len(fb):
        return {}
    fb = fb.assign(value=fb.clicks.astype(float) + 2.0 * fb.purchases.astype(float),
                   impressions=fb.impressions.astype(float))
    by_slot = fb.groupby("slot")[["value", "impressions"]].sum()
    rate = (by_slot.value / by_slot.impressions.clip(lower=1)).to_dict()
    fb = fb.assign(expected=fb.impressions * fb.slot.map(rate).fillna(0.0))
    agg = fb.groupby("product_id")[["value", "expected"]].sum()
    lift = ((agg.value + prior) / (agg.expected + prior)).clip(0.3, 3.0)
    return lift.to_dict()


# Solo live: instantánea SQL; no consulta por petición. El backtest favorece pop_7d.
ITEM_LIVE = """
WITH now64(3) AS instante,
versiones AS (
    SELECT product_id::String AS product_id, price_eur::Float64 AS precio,
           is_active::Bool AS activo, toDateTime64('1970-01-01', 3) AS ts
    FROM shop.catalog
    UNION ALL
    SELECT product_id::String, price_eur::Float64, is_active::Bool, ts::DateTime64(3)
    FROM shop.catalog_updates WHERE ts::DateTime64(3) <= instante
), secuencia AS (
    SELECT *, lagInFrame(precio, 1, precio) OVER (
        PARTITION BY product_id ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS anterior FROM versiones
), catalogo AS (
    SELECT product_id, argMax(precio, ts) AS price_eur,
           argMin(precio, ts) AS initial_price_eur, argMax(activo, ts) AS in_stock,
           maxIf(ts, precio < anterior) AS ultima_bajada
    FROM secuencia GROUP BY product_id
), recientes AS (
    SELECT product_id::String AS product_id,
           countIf(event_type::String = 'purchase' AND ts::DateTime64(3) > instante - INTERVAL 60 MINUTE) AS purchases_60m,
           countIf(event_type::String = 'view' AND ts::DateTime64(3) > instante - INTERVAL 60 MINUTE) AS views_60m,
           countIf(event_type::String = 'purchase') AS purchases_3h,
           countIf(event_type::String = 'view') AS views_3h
    FROM shop.events
    WHERE ts::DateTime64(3) > instante - INTERVAL 3 HOUR AND ts::DateTime64(3) <= instante
    GROUP BY product_id
), semana AS (
    SELECT product_id, sum(n) AS n FROM (
        SELECT product_id::String AS product_id, count() AS n FROM shop.history_events
        WHERE event_type::String = 'purchase' AND ts::DateTime64(3) > instante - INTERVAL 7 DAY
          AND ts::DateTime64(3) <= instante GROUP BY product_id
        UNION ALL
        SELECT product_id::String, count() FROM shop.events
        WHERE event_type::String = 'purchase' AND ts::DateTime64(3) > instante - INTERVAL 7 DAY
          AND ts::DateTime64(3) <= instante GROUP BY product_id
    ) GROUP BY product_id
)
SELECT c.product_id AS product_id,
       coalesce(r.purchases_60m, 0) AS purchases_60m, coalesce(r.views_60m, 0) AS views_60m,
       coalesce(r.purchases_3h, 0) AS purchases_3h, coalesce(r.views_3h, 0) AS views_3h,
       c.price_eur AS price_eur, c.initial_price_eur AS initial_price_eur,
       c.price_eur / nullIf(c.initial_price_eur, 0) AS price_ratio,
       if(c.ultima_bajada = toDateTime64('1970-01-01', 3), NULL,
          dateDiff('second', c.ultima_bajada, instante) / 60.0) AS minutes_since_price_drop,
       c.in_stock AS in_stock, coalesce(s.n, 0) / 168.0 AS trend_score,
       instante AS as_of
FROM catalogo AS c LEFT JOIN recientes AS r ON c.product_id = r.product_id
LEFT JOIN semana AS s ON c.product_id = s.product_id
"""



def folder(until: str | None) -> Path:
    return DATA_DIR / (f"features_until_{tag(until)}" if until else "features_live")


def build(until: str | None = None, only: list[str] | None = None, verbose: bool = True) -> dict[str, pd.DataFrame]:
    out_dir = folder(until)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name, sql in queries(until).items():
        if only and name not in only:
            continue
        t = time.time()
        try:
            df = pd.DataFrame(query(sql, timeout=900))
        except Exception as e:
            if name not in OPTIONAL:
                raise
            print(f"  {name}: no actualizada ({e})")
            continue
        # Reemplazo atómico: un lector concurrente nunca abre un Parquet a medio escribir.
        fd, tmp = tempfile.mkstemp(prefix=f".{name}-", suffix=".tmp", dir=out_dir)
        os.close(fd)
        try:
            df.to_parquet(tmp, index=False)
            os.replace(tmp, out_dir / f"{name}.parquet")
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
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
