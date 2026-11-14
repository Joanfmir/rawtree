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
print("\n── Por versión de modelo (test A/B) ──")
ab = query_df(f"""
SELECT s.model_version::String AS model, count() AS requests,
       countIf(f.purchased::String != '') AS purchases,
       round(100 * countIf(f.clicked::String != '') / count(), 2) AS ctr_pct,
       round(100 * countIf(f.purchased::String != '') / count(), 2) AS conv_pct,
       round(sum(f.revenue_eur::Float64) / count(), 3) AS rev_per_req,
       round(100 * countIf(f.served_by::String = 'team') / count(), 1) AS on_time_pct
FROM {T}.feedback AS f INNER JOIN {T}.submissions AS s ON s.request_id::String = f.request_id::String
WHERE f.ts::DateTime64(3) > now() - INTERVAL {minutes} MINUTE
GROUP BY model ORDER BY requests DESC""", timeout=30)
if len(ab):
    n, p = ab.requests.astype(float), ab.purchases.astype(float) / ab.requests.astype(float)
    # margen de error al 95 %: la conversión real está en conv_pct ± conv_err con esa confianza
    ab.insert(5, "conv_err", (196 * (p * (1 - p) / n) ** 0.5).round(2))
print(ab.to_string(index=False))
if len(ab) >= 2:
    # ¿la diferencia entre las dos versiones con más tráfico es real o puede ser azar?
    (n1, x1, m1), (n2, x2, m2) = [(float(r.requests), float(r.purchases), r.model) for r in ab.head(2).itertuples()]
    pool = (x1 + x2) / (n1 + n2)
    se = (pool * (1 - pool) * (1 / n1 + 1 / n2)) ** 0.5
    z = (x1 / n1 - x2 / n2) / se if se else 0.0
    best = m1 if z > 0 else m2
    verdict = (f"gana {best} con claridad" if abs(z) >= 2.6 else
               f"probablemente gana {best}, pero conviene esperar más datos" if abs(z) >= 2 else
               "todavía no hay diferencia fiable: puede ser azar")
    print(f"\n{m1} vs {m2}: diferencia de {100 * (x1 / n1 - x2 / n2):+.2f} puntos de conversión (z = {z:+.2f}) → {verdict}")
