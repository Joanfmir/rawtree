"""Compara versiones en vivo con nuestro feedback (el simulador real).

    python scripts/ab.py                       # todas las versiones, desde que conviven las dos últimas
    python scripts/ab.py Leattner.v1 Leattner.v2
    python scripts/ab.py --since "2026-11-14 13:00:00"   # hora UTC

Solo cuenta peticiones servidas por nosotros (served_by = team). El intervalo es el del 95 % de la
diferencia de conversión contra la primera versión (aproximación normal): si no incluye el 0, es fiable.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402

from reco import config  # noqa: E402
from reco.client import query_df  # noqa: E402

config.check()
args = sys.argv[1:]
since = None
if "--since" in args:
    i = args.index("--since")
    since = args[i + 1]
    args = args[:i] + args[i + 2:]
T = config.TEAM
JOIN = f"""FROM {T}.feedback AS f
INNER JOIN {T}.submissions AS s ON s.request_id::String = f.request_id::String
INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
WHERE f.served_by::String = 'team' AND f.request_id::String != '__init__'"""

if not args:   # las dos versiones más recientes
    v = query_df(f"SELECT s.model_version::String AS v, max(f.ts::DateTime64(3)) AS last, min(f.ts::DateTime64(3)) AS first {JOIN} GROUP BY v ORDER BY last DESC, first DESC LIMIT 2")
    args = sorted(v.v.tolist())
if since is None:   # desde que la versión más nueva empezó a servir
    since = str(query_df(f"SELECT max(first) AS t FROM (SELECT min(f.ts::DateTime64(3)) AS first {JOIN} AND s.model_version::String IN ({', '.join(repr(a) for a in args)}) GROUP BY s.model_version::String)").t[0])
vers = ", ".join(repr(a) for a in args)
df = query_df(f"""
SELECT s.model_version::String AS version,
       multiIf(length(r.recent_views) = 0, 'home (sin vistas)', r.is_new_user::Bool, 'nuevo', 'conocido') AS segmento,
       count() AS n, countIf(f.clicked::String != '') AS clics, countIf(f.purchased::String != '') AS compras,
       sum(f.revenue_eur::Float64) AS ingresos
{JOIN} AND s.model_version::String IN ({vers}) AND f.ts::DateTime64(3) >= '{since}'
GROUP BY version, segmento""", timeout=120)
if df.empty:
    raise SystemExit("Sin datos todavía.")
tot = df.groupby("version", as_index=False)[["n", "clics", "compras", "ingresos"]].sum().assign(segmento="TODOS")
df = pd.concat([tot, df])
print(f"Versiones: {args} · desde {since} UTC\n")
for seg, g in df.groupby("segmento", sort=False):
    g = g.set_index("version").reindex(args).dropna()
    if len(g) < 1:
        continue
    out = pd.DataFrame({"n": g.n.astype(int), "compras": g.compras.astype(int), "conv %": 100 * g.compras / g.n,
                        "ctr %": 100 * g.clics / g.n, "€/petición": g.ingresos / g.n})
    base = g.iloc[0]
    p0 = base.compras / base.n
    lo, hi = [], []
    for _, r in g.iterrows():
        p1 = r.compras / r.n
        se = math.sqrt(p0 * (1 - p0) / base.n + p1 * (1 - p1) / r.n)
        lo.append(100 * (p1 - p0 - 1.96 * se)); hi.append(100 * (p1 - p0 + 1.96 * se))
    out["dif conv (IC 95 %)"] = [f"{(a + b) / 2:+.2f} [{a:+.2f}, {b:+.2f}]" for a, b in zip(lo, hi)]
    out.iloc[0, -1] = "(referencia)"
    print(f"── {seg}\n{out.to_string(float_format=lambda x: f'{x:.3f}')}\n")
