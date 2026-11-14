"""Cómo vamos en vivo (team_NN.feedback, últimos N minutos).   python scripts/monitor.py [minutos]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reco import config  # noqa: E402
from reco.client import query_df  # noqa: E402

config.check()
minutes = int(sys.argv[1]) if len(sys.argv) > 1 else 30
T = config.TEAM
print(f"── Últimos {minutes} min ──")
print(query_df(f"""
SELECT served_by::String AS served_by, fallback_reason::String AS reason, count() AS requests,
       round(100 * countIf(clicked != '') / count(), 2)   AS ctr_pct,
       round(100 * countIf(purchased != '') / count(), 2) AS conv_pct,
       round(sum(revenue_eur::Float64) / count(), 3)      AS rev_per_req,
       quantile(0.5)(latency_ms::Float64) AS p50_ms, quantile(0.95)(latency_ms::Float64) AS p95_ms
FROM {T}.feedback
WHERE request_id != '__init__' AND ts::DateTime64(3) > now() - INTERVAL {minutes} MINUTE
GROUP BY served_by, reason ORDER BY requests DESC""", timeout=30).to_string(index=False))
print("\n── Por versión de modelo ──")
print(query_df(f"""
SELECT s.model_version::String AS model, count() AS requests,
       round(100 * countIf(f.purchased::String != '') / count(), 2) AS conv_pct,
       round(100 * countIf(f.served_by::String = 'team') / count(), 1) AS on_time_pct
FROM {T}.feedback AS f INNER JOIN {T}.submissions AS s ON s.request_id::String = f.request_id::String
WHERE f.ts::DateTime64(3) > now() - INTERVAL {minutes} MINUTE
GROUP BY model ORDER BY requests DESC""", timeout=30).to_string(index=False))
