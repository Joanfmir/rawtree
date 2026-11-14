"""Catálogo actual, stock y best sellers. Todo calculado en SQL dentro de RawTree."""
from __future__ import annotations

from .client import query

# shop.catalog + la última versión de cada producto en shop.catalog_updates
CURRENT_CATALOG = """
SELECT product_id,
       argMax(category, v) AS category, argMax(subcategory, v) AS subcategory,
       argMax(brand, v) AS brand, argMax(price_eur, v) AS price_eur,
       argMax(is_active, v) AS is_active, max(is_new) AS is_new
FROM (
    SELECT product_id::String AS product_id, category::String AS category, subcategory::String AS subcategory,
           brand::String AS brand, price_eur::Float64 AS price_eur, is_active::Bool AS is_active,
           0 AS is_new, '' AS v
    FROM shop.catalog
    UNION ALL
    SELECT product_id::String, category::String, subcategory::String, brand::String, price_eur::Float64,
           is_active::Bool, toUInt8(change::String = 'new_product'), toString(ts::DateTime64(3))
    FROM shop.catalog_updates
)
GROUP BY product_id"""

STATIC_CATALOG = """
SELECT product_id::String AS product_id, category::String AS category, subcategory::String AS subcategory,
       brand::String AS brand, price_eur::Float64 AS price_eur, is_active::Bool AS is_active, 0 AS is_new
FROM shop.catalog"""

OUT_OF_STOCK = """
SELECT product_id::String AS product_id
FROM shop.catalog_updates
WHERE change::String IN ('out_of_stock', 'back_in_stock')
GROUP BY product_id
HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock'"""

SALES = """
SELECT product_id::String AS product_id, count() AS n
FROM shop.history_events
WHERE event_type = 'purchase'{cut}
GROUP BY product_id"""


def current_catalog() -> dict[str, dict]:
    """{product_id: fila}. Si catalog_updates aún está vacía (antes del stream), usa solo shop.catalog."""
    try:
        rows = query(CURRENT_CATALOG, timeout=30)
    except RuntimeError as exc:
        # Solo antes del stream puede no existir el esquema de catalog_updates.
        if "Unknown expression identifier" not in str(exc):
            raise
        rows = query(STATIC_CATALOG, timeout=30)
    return {r["product_id"]: r for r in rows}


def out_of_stock() -> set[str]:
    try:
        return {r["product_id"] for r in query(OUT_OF_STOCK)}
    except RuntimeError as exc:
        if "Unknown expression identifier" not in str(exc):
            raise
        return set()


def best_sellers(until: str | None = None) -> list[str]:
    """Productos más vendidos del histórico (= baseline / fallback). `until` para evaluación offline."""
    cut = f" AND ts < '{until}'" if until else ""
    rows = query(SALES.format(cut=cut), timeout=60)
    return [r["product_id"] for r in sorted(rows, key=lambda r: (int(r["n"]), r["product_id"]), reverse=True)]
