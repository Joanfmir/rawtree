SELECT min(ts::DateTime64(3)) AS inicio, max(ts::DateTime64(3)) AS fin FROM team_17.feedback WHERE request_id::String != '__init__';


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
           is_active::Bool, toUInt8(change::String = 'new_product'), toString(ts)
    FROM shop.catalog_updates
)
GROUP BY product_id;

SELECT request_id::String AS id, ts::DateTime64(3) AS ts, shown::Array(String) AS shown, served_by::String AS served_by, fallback_reason::String AS motivo, latency_ms::Float64 AS latencia, clicked::String AS clicked, purchased::String AS purchased FROM team_17.feedback WHERE request_id::String != '__init__' AND ts::DateTime64(3) >= '2026-11-14 10:36:49.934000' AND ts::DateTime64(3) < '2026-11-14 11:36:49.934000';

SELECT request_id::String AS id, recent_views::Array(JSON) AS vistas FROM shop.requests WHERE request_id::String IN (SELECT request_id::String FROM team_17.feedback WHERE request_id::String != '__init__' AND ts::DateTime64(3) >= '2026-11-14 10:36:49.934000' AND ts::DateTime64(3) < '2026-11-14 11:36:49.934000');

SELECT request_id::String AS id, items::Array(String) AS items, model_version::String AS version FROM team_17.submissions WHERE request_id::String IN (SELECT request_id::String FROM team_17.feedback WHERE request_id::String != '__init__' AND ts::DateTime64(3) >= '2026-11-14 10:36:49.934000' AND ts::DateTime64(3) < '2026-11-14 11:36:49.934000');

SELECT request_id::String AS id, ts::DateTime64(3) AS ts, shown::Array(String) AS shown, served_by::String AS served_by, fallback_reason::String AS motivo, latency_ms::Float64 AS latencia, clicked::String AS clicked, purchased::String AS purchased FROM team_17.feedback WHERE request_id::String != '__init__' AND ts::DateTime64(3) >= '2026-11-14 11:36:49.934000' AND ts::DateTime64(3) < '2026-11-14 12:36:49.934000';

SELECT request_id::String AS id, recent_views::Array(JSON) AS vistas FROM shop.requests WHERE request_id::String IN (SELECT request_id::String FROM team_17.feedback WHERE request_id::String != '__init__' AND ts::DateTime64(3) >= '2026-11-14 11:36:49.934000' AND ts::DateTime64(3) < '2026-11-14 12:36:49.934000');

SELECT request_id::String AS id, items::Array(String) AS items, model_version::String AS version FROM team_17.submissions WHERE request_id::String IN (SELECT request_id::String FROM team_17.feedback WHERE request_id::String != '__init__' AND ts::DateTime64(3) >= '2026-11-14 11:36:49.934000' AND ts::DateTime64(3) < '2026-11-14 12:36:49.934000');

SELECT request_id::String AS id, ts::DateTime64(3) AS ts, shown::Array(String) AS shown, served_by::String AS served_by, fallback_reason::String AS motivo, latency_ms::Float64 AS latencia, clicked::String AS clicked, purchased::String AS purchased FROM team_17.feedback WHERE request_id::String != '__init__' AND ts::DateTime64(3) >= '2026-11-14 12:36:49.934000' AND ts::DateTime64(3) < '2026-11-14 13:05:36.592000';

SELECT request_id::String AS id, recent_views::Array(JSON) AS vistas FROM shop.requests WHERE request_id::String IN (SELECT request_id::String FROM team_17.feedback WHERE request_id::String != '__init__' AND ts::DateTime64(3) >= '2026-11-14 12:36:49.934000' AND ts::DateTime64(3) < '2026-11-14 13:05:36.592000');

SELECT request_id::String AS id, items::Array(String) AS items, model_version::String AS version FROM team_17.submissions WHERE request_id::String IN (SELECT request_id::String FROM team_17.feedback WHERE request_id::String != '__init__' AND ts::DateTime64(3) >= '2026-11-14 12:36:49.934000' AND ts::DateTime64(3) < '2026-11-14 13:05:36.592000');
