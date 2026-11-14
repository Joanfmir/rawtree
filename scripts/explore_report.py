"""Cruza exploración y feedback real; tasas IPS normalizadas (SNIPS).

Sin el número total de oportunidades elegibles no se puede calcular una tasa
Horvitz–Thompson por petición. SNIPS con 1/p da el promedio sobre pares
producto-slot del soporte registrado, ponderando contextos por su soporte.
No estima el efecto causal de sustituir ni el valor de una lista alternativa.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reco import config
from reco.client import query


def estimar(rows):
    """Resumen por grupo; sin clipping que cambiaría el estimando."""
    import pandas as pd
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    summaries = []
    for keys in ([], ["slot"], ["category"], ["slot", "category"]):
        groups = [((), df)] if not keys else df.groupby(keys, dropna=False)
        for key, g in groups:
            vals = key if isinstance(key, tuple) else (key,)
            w = 1 / g.propensity.astype(float)
            summaries.append(dict(grupo="+".join(keys) or "total",
                                  detalle=str(dict(zip(keys, vals))), n=len(g),
                                  clics=int(g.click.sum()), compras=int(g.buy.sum()),
                                  ctr_pct=100 * g.click.mean(), conversion_pct=100 * g.buy.mean(),
                                  ctr_snips_pct=100 * (w * g.click).sum() / w.sum(),
                                  conversion_snips_pct=100 * (w * g.buy).sum() / w.sum(),
                                  ess=w.sum() ** 2 / (w * w).sum()))
    return pd.DataFrame(summaries)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=("local", "scratch"), default="local")
    ap.add_argument("--log", type=Path, default=config.LOGS_DIR / "explore.jsonl")
    ap.add_argument("--since", default="2026-11-14 00:00:00", help="UTC; ISO 8601")
    ap.add_argument("--until", help="UTC; límite exclusivo")
    ap.add_argument("--eps", type=float, default=0.10, help="Para el cálculo de coste")
    ap.add_argument("--slot3-purchase-pct", type=float,
                    help="Compra por impresión en slot 3, en porcentaje")
    args = ap.parse_args()
    if not 0 <= args.eps <= 1:
        ap.error("eps debe estar entre 0 y 1")
    print(f"Coste directo si el sustituto nunca vende: slots 8–10, "
          f"0–{args.eps * 0.05:.5f} compras/100 peticiones "
          "(tasa base 0–0,05 % por impresión).")
    if args.slot3_purchase_pct is not None:
        print(f"Slot 3: {args.eps * args.slot3_purchase_pct:.5f} compras/100 "
              "si el sustituto nunca vende; aproximación sin interacciones.")
    from datetime import datetime, timezone
    def fecha(value):
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    since = fecha(args.since)
    until = fecha(args.until) if args.until else None
    cut = f"ts::DateTime64(3) >= '{since}'"
    if until:
        cut += f" AND ts::DateTime64(3) < '{until}'"
    if args.source == "scratch":
        fields = ["request_id::String AS request_id", "slot::Int64 AS slot",
                  "product_id::String AS product_id", "propensity::Float64 AS propensity",
                  "category::String AS category", "dry_run::Bool AS dry_run",
                  "shown::Array(String) AS shown"]
        logs = query(f"SELECT {', '.join(fields)} FROM {config.TEAM}.scratch "
                     f"WHERE kind::String = 'explore' AND {cut} LIMIT 499999", timeout=60)
        if len(logs) == 499999:
            raise SystemExit("Ventana demasiado grande: acótala con --since/--until")
    else:
        logs = []
        if args.log.exists():
            for line in args.log.read_text().splitlines():
                try:
                    r = json.loads(line)
                    ts = fecha(r["ts"])
                    if ts >= since and (not until or ts < until):
                        logs.append(r)
                except (ValueError, KeyError):
                    print("Aviso: línea incompleta o inválida omitida", file=sys.stderr)
    live = [r for r in logs if not r.get("dry_run", False)]
    print(f"Registros: {len(logs)}; dry-run excluidos: {len(logs) - len(live)}")
    if not live:
        print("Sin exploración real: no se pueden estimar CTR ni conversión.")
        return
    # Lotes pequeños de IDs: solo SELECT, nunca se descarga todo el feedback.
    ids = sorted({str(r["request_id"]) for r in live})
    feedback = {}
    for start in range(0, len(ids), 500):
        quoted = ",".join("'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'" for s in ids[start:start + 500])
        rows = query(f"SELECT request_id::String AS request_id, shown::Array(String) AS shown, "
                     f"clicked::String AS clicked, purchased::String AS purchased, served_by::String AS served_by "
                     f"FROM {config.TEAM}.feedback WHERE request_id::String IN ({quoted}) "
                     "AND request_id::String != '__init__' LIMIT 499999", timeout=60)
        for r in rows:
            feedback.setdefault(r["request_id"], []).append(r)
    matched, seen = [], set()
    for r in live:
        rid = r["request_id"]
        fs = feedback.get(rid, [])
        if rid in seen or len(fs) != 1:
            continue
        f = fs[0]
        slot, p = int(r["slot"]), r["product_id"]
        if (f["served_by"] != "team" or not 1 <= slot <= len(f["shown"])
                or f["shown"][slot - 1] != p or f["shown"] != r.get("shown")
                or not 0 < float(r["propensity"]) <= 1):
            continue
        seen.add(rid)
        matched.append({**r, "click": int(f["clicked"] == p), "buy": int(f["purchased"] == p)})
    print(f"Exposiciones verificadas: {len(matched)} / {len(live)} (sin coincidencia se excluyen)")
    result = estimar(matched)
    print(result.to_string(index=False) if not result.empty else "Sin feedback atribuible todavía.")
    print("SNIPS: pesos 1/propensity; soporte limitado a los populares elegibles. ESS mide muestra efectiva.")
    print("No identifica uplift ni compras perdidas; hacen falta control aleatorio y más muestras.")


if __name__ == "__main__":
    main()
