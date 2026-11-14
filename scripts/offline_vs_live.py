"""Evaluación offline contrastada con el leaderboard: ¿es fiable? ¿merece un modelo nuevo ir al vivo?

    python scripts/offline_vs_live.py                               # fiabilidad con las versiones servidas
    python scripts/offline_vs_live.py --candidates mi_politica      # dónde cae una política nueva
    python scripts/offline_vs_live.py --candidates a b --vs bird_v2 # contra una política concreta
    make offline C="mi_politica"

Flujo con una branch de modelo nuevo (ver CLAUDE.md, "Evaluar un modelo nuevo"):
  1. Traer su código (git merge) y sus modelos de data/ entrenados con --until 2026-11-07.
  2. Registrar la política en reco/policies/__init__.py.
  3. Ejecutar este script con --candidates <política>.
  4. Si va al vivo, añadir su model_version a LIVE_VERSIONS (reco/offline_live.py).

La primera ejecución calcula las listas de todas las versiones (~1 min por política); después solo
recalcula las políticas cuyo código o modelos hayan cambiado.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from reco import config  # noqa: E402
from reco import offline_live as ol  # noqa: E402
from reco.evaluate import DEFAULT_UNTIL  # noqa: E402
from reco.policies import POLICIES  # noqa: E402

F2 = lambda x: f"{x:.2f}"  # noqa: E731
F3 = lambda x: f"{x:.3f}"  # noqa: E731


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", nargs="*", default=[], help="políticas nuevas (nombres de POLICIES)")
    ap.add_argument("--vs", help="política de referencia para los candidatos (por defecto, la mejor en vivo)")
    ap.add_argument("--until", default=DEFAULT_UNTIL)
    ap.add_argument("--offline-cache", action="store_true",
                    help="usar la copia guardada del leaderboard sin consultarlo")
    args = ap.parse_args()
    config.check()

    unknown = [c for c in args.candidates + ([args.vs] if args.vs else []) if c not in POLICIES]
    if unknown:
        raise SystemExit(f"Políticas no registradas en reco/policies/__init__.py: {unknown}")

    # ── leaderboard
    if args.offline_cache:
        saved = json.loads((ol.CACHE / f"leaderboard_{config.TEAM}.json").read_text())
        lb = ol.combine(pd.DataFrame(saved["rows"]))
    else:
        lb = ol.fetch_leaderboard(config.TEAM)
    missing = [v for v in lb.index if v not in ol.LIVE_VERSIONS]
    lb["policy"] = [ol.LIVE_VERSIONS.get(v) for v in lb.index]
    bad = sorted({p for p in lb.policy.dropna() if p not in POLICIES})
    if bad:
        print(f"[aviso] en LIVE_VERSIONS pero no en POLICIES (no cuentan): {bad}")
    usable = lb[lb.policy.notna() & lb.policy.isin(POLICIES)]

    # ── offline (listas en caché por política)
    print(f"Test offline: history_sessions desde {args.until}, sin recompras")
    test = ol.TestSet.load(args.until)
    print(f"  {len(test.df)} carruseles con clic, {(test.df.purchased != '').sum()} con compra")
    names = list(dict.fromkeys(list(usable.policy) + args.candidates + ([args.vs] if args.vs else [])))
    lists, failed = {}, {}
    for p in names:
        try:
            lists[p] = ol.policy_lists(p, test)
        except Exception as e:   # p. ej. falta el modelo en data/
            failed[p] = f"{type(e).__name__}: {e}"
    for p, e in failed.items():
        print(f"[aviso] {p} no se pudo evaluar offline: {e[:200]}")
    off = pd.DataFrame({p: ol.scores(lists[p], test.df) for p in lists}).T

    # ── tabla por versión en vivo
    table = lb.join(off, on="policy")
    show = table[["policy", "requests", "conv", "se", "cold_conv", "ips_ndcg", "ips_hit", "hit@10", "nuevo_ips_ndcg"]]
    print("\n── Versiones en vivo: leaderboard (conversión esperada por 100 peticiones) frente a offline")
    print(show.rename(columns={"conv": "conv_lb", "cold_conv": "conv_nuevos_lb"})
          .to_string(float_format=F3, na_rep="—"))
    if missing:
        print(f"\n[aviso] versiones en vivo sin mapear en LIVE_VERSIONS: {missing}")

    # ── fiabilidad
    ev = table.dropna(subset=["ips_ndcg"])
    report = {"generated_at": time.time(), "until": args.until, "versions": list(ev.index), "metrics": {}}
    print(f"\n── Fiabilidad: ¿ordena la métrica offline como el leaderboard? ({len(ev)} versiones)")
    rows = []
    for met, target, se in (("ips_ndcg", "conv", "se"), ("ips_hit", "conv", "se"), ("hit@10", "conv", "se"),
                            ("nuevo_ips_ndcg", "cold_conv", None)):
        sub = ev.dropna(subset=[target])
        if len(sub) < 3:
            continue
        s = sub[se].to_numpy() if se else 100 * np.sqrt(sub[target] / 100 / sub.cold_requests)
        a = ol.agreement(list(sub.index), sub[met].to_numpy(), sub[target].to_numpy(), np.asarray(s))
        report["metrics"][met] = {**a, "wrong": [list(w) for w in a["wrong"]]}
        rows.append({"métrica offline": met, "frente a": "conv nuevos" if target == "cold_conv" else "conv",
                     "spearman": a["spearman"], "parejas bien": "%d/%d" % a["pairs"],
                     "parejas claras bien": "%d/%d" % a["clear_pairs"]})
    print(pd.DataFrame(rows).to_string(index=False, float_format=F2))
    wrong = report["metrics"].get("ips_ndcg", {}).get("wrong", [])
    if wrong:
        print("\n  Parejas que ips_ndcg ordena al revés (* = el leaderboard las separa con claridad):")
        for a, b, clear in wrong:
            print(f"    {'*' if clear else ' '} {a} ({ev.loc[a, 'ips_ndcg']:.3f} offline, {ev.loc[a, 'conv']:.2f} lb)"
                  f"  vs  {b} ({ev.loc[b, 'ips_ndcg']:.3f}, {ev.loc[b, 'conv']:.2f})")

    # calibración: recta conv ~ ips_ndcg; los residuos dicen a quién sobrevalora la métrica
    slope, icpt = np.polyfit(ev.ips_ndcg, ev.conv, 1)
    ev = ev.assign(pred=icpt + slope * ev.ips_ndcg)
    ev["residuo"] = ev.conv - ev.pred
    print(f"\n── Calibración: conv_lb ≈ {icpt:.2f} + {slope:.2f} · ips_ndcg"
          "   (residuo < 0: offline la sobrevalora)")
    print(ev[["policy", "ips_ndcg", "conv", "pred", "residuo"]].sort_values("residuo")
          .to_string(float_format=F3))
    resid_sd = float(ev.residuo.std(ddof=2)) if len(ev) > 2 else float("nan")
    report["calibration"] = {"intercept": icpt, "slope": slope, "resid_sd": resid_sd}

    # ── candidatos
    if args.candidates:
        ref = args.vs or ev.sort_values("conv", ascending=False).policy.iloc[0]
        ref_ver = next((v for v, p in ol.LIVE_VERSIONS.items() if p == ref and v in lb.index), None)
        print(f"\n── Candidatos frente a {ref}" + (f" ({ref_ver}, {lb.loc[ref_ver, 'conv']:.2f} en el lb)" if ref_ver else ""))
        crow = []
        for c in args.candidates:
            if c not in lists:
                continue
            d, lo, hi = ol.bootstrap_diff(lists[c], lists[ref], test.df)
            dn, lon, hin = ol.bootstrap_diff(lists[c], lists[ref], test.df, hide=True)
            pred = icpt + slope * off.loc[c, "ips_ndcg"]
            rank = 1 + int((ev.ips_ndcg > off.loc[c, "ips_ndcg"]).sum())
            if hi < 0:
                verdict = "PEOR offline: descartar (la métrica acierta casi todas las parejas claras)"
            elif lo > 0:
                verdict = ("MEJOR offline: candidata, pero CONFIRMAR EN VIVO (A/B simultáneo); "
                           "el único fallo claro fue un caso así")
            else:
                verdict = "EMPATE offline: la evaluación no lo separa; solo el vivo decide"
            crow.append({"política": c, "ips_ndcg": off.loc[c, "ips_ndcg"], "Δ vs ref": d,
                         "IC 95 %": f"[{lo:+.3f}, {hi:+.3f}]", "Δ nuevos": dn, "IC nuevos": f"[{lon:+.3f}, {hin:+.3f}]",
                         f"puesto (de {len(ev) + 1})": rank, "conv_lb estimada": f"{pred:.2f} ± {2 * resid_sd:.2f}",
                         "veredicto": verdict})
            report.setdefault("candidates", {})[c] = {**off.loc[c].to_dict(), "diff": [d, lo, hi],
                                                      "diff_new": [dn, lon, hin], "ref": ref, "verdict": verdict}
        out = pd.DataFrame(crow).set_index("política")
        print(out.drop(columns="veredicto").to_string(float_format=F3))
        for c, v in out.veredicto.items():
            print(f"  {c}: {v}")

    ol.CACHE.mkdir(parents=True, exist_ok=True)
    path = ol.CACHE / f"report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nInforme → {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
