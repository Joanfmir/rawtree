"""Evaluación offline contrastada con el leaderboard: ¿es fiable? ¿merece un modelo nuevo ir al vivo?

    python scripts/offline_vs_live.py                               # fiabilidad con las versiones servidas
    python scripts/offline_vs_live.py --candidates mi_politica      # dónde cae una política nueva
    python scripts/offline_vs_live.py --candidates a b --vs bird_v2 # contra una política concreta
    python scripts/offline_vs_live.py --tests organic               # solo un test
    make offline C="mi_politica"

Dos tests (ver reco/offline_live.py):
  history   carruseles del histórico desde el 07/11 (sin recompras); métrica ips_ndcg
  organic   peticiones reales de hoy desde --organic-cut; objetivo: lo que el usuario compró por su
            cuenta en los --window minutos siguientes; métrica ndcg@10. Más datos y usuarios nuevos reales.

Flujo con una branch de modelo nuevo (ver CLAUDE.md, "Evaluar un modelo nuevo"):
  1. Traer su código (git merge) y sus modelos de data/ entrenados con datos ANTERIORES a los cortes.
  2. Registrar la política en reco/policies/__init__.py.
  3. Ejecutar este script con --candidates <política>.
  4. Si va al vivo, añadir su model_version a LIVE_VERSIONS (reco/offline_live.py).

La primera ejecución calcula las listas de todas las versiones (unos minutos); después solo recalcula
las políticas cuyo código o modelos hayan cambiado.
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
ORGANIC_CUT = "2026-11-14 12:00:00"
# Fin fijo: todo el equipo evalúa con el mismo test y la caché no caduca cada hora. "latest" = última hora completa.
ORGANIC_END = "2026-11-14 20:00:00"


def load_tests(args) -> list:
    tests = []
    for kind in args.tests:
        if kind == "history":
            t = ol.TestSet.load(args.until)
            print(f"[history] carruseles de history_sessions desde {args.until}, sin recompras: "
                  f"{len(t.df)} con clic, {t.n_labels} con compra")
        else:
            t = ol.OrganicTestSet.load(args.organic_cut, window=args.window,
                                      end=None if args.organic_end == "latest" else args.organic_end)
            print(f"[organic] peticiones de {args.organic_cut} a {t.end} con compra orgánica nueva en "
                  f"{args.window} min: {len(t.df)} ({int(t.df.is_new_user.sum())} de usuarios nuevos)")
        tests.append(t)
    return tests


def reliability(test, ev: pd.DataFrame, report: dict) -> None:
    m, mn = test.main, f"nuevo_{test.main}"
    rows = []
    for met, target, se in ((m, "conv", "se"), (mn, "cold_conv", None)):
        sub = ev.dropna(subset=[met, target])
        if len(sub) < 3:
            continue
        s = sub[se].to_numpy() if se else 100 * np.sqrt(sub[target] / 100 / sub.cold_requests)
        a = ol.agreement(list(sub.index), sub[met].to_numpy(), sub[target].to_numpy(), np.asarray(s))
        report["metrics"][f"{test.kind}:{met}"] = {**a, "wrong": [list(w) for w in a["wrong"]]}
        rows.append({"test": test.kind, "métrica offline": met,
                     "frente a": "conv nuevos" if target == "cold_conv" else "conv",
                     "spearman": a["spearman"], "parejas bien": "%d/%d" % a["pairs"],
                     "parejas claras bien": "%d/%d" % a["clear_pairs"]})
        if met == m and a["wrong"]:
            rows[-1]["al revés (* = claras)"] = "; ".join(f"{'*' if c else ''}{x}<>{y}" for x, y, c in a["wrong"])
    report.setdefault("_rows", []).extend(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", nargs="*", default=[], help="políticas nuevas (nombres de POLICIES)")
    ap.add_argument("--vs", help="política de referencia para los candidatos (por defecto, la mejor en vivo)")
    ap.add_argument("--tests", nargs="+", choices=["history", "organic"], default=["history", "organic"])
    ap.add_argument("--until", default=DEFAULT_UNTIL, help="corte del test histórico")
    ap.add_argument("--organic-cut", default=ORGANIC_CUT, help="corte del test orgánico (UTC)")
    ap.add_argument("--organic-end", default=ORGANIC_END,
                    help=f"fin de las peticiones del test orgánico (por defecto {ORGANIC_END}; 'latest' = última hora completa)")
    ap.add_argument("--window", type=int, default=30, help="minutos tras la petición en los que mirar compras")
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

    # ── offline
    tests = load_tests(args)
    names = list(dict.fromkeys(list(usable.policy) + args.candidates + ([args.vs] if args.vs else [])))
    lists = {t.kind: {} for t in tests}
    for t in tests:
        for p in names:
            try:
                lists[t.kind][p] = ol.policy_lists(p, t)
            except Exception as e:   # p. ej. falta el modelo en data/
                print(f"[aviso] {p} [{t.kind}] no se pudo evaluar: {type(e).__name__}: {str(e)[:200]}")
    off = {t.kind: pd.DataFrame({p: ol.scores(t, ls) for p, ls in lists[t.kind].items()}).T for t in tests}

    # ── tabla por versión en vivo
    table = lb.copy()
    cols = ["policy", "requests", "conv", "cold_conv"]
    for t in tests:
        o = off[t.kind][[t.main, f"nuevo_{t.main}"]].add_prefix(f"{t.kind[:4]}:")
        table = table.join(o, on="policy")
        cols += list(o.columns)
    print("\n── Versiones en vivo: leaderboard (conversión esperada por 100 peticiones) frente a offline")
    print(table[cols].rename(columns={"conv": "conv_lb", "cold_conv": "conv_nuevos_lb"})
          .to_string(float_format=F3, na_rep="—"))
    if missing:
        print(f"\n[aviso] versiones en vivo sin mapear en LIVE_VERSIONS: {missing}")

    # ── fiabilidad y calibración por test
    report = {"generated_at": time.time(), "tests": [t.tag for t in tests], "metrics": {}, "calibration": {}}
    calib = {}
    for t in tests:
        ev = lb.join(off[t.kind], on="policy").dropna(subset=[t.main])
        reliability(t, ev, report)
        if len(ev) > 2:
            slope, icpt = np.polyfit(ev[t.main], ev.conv, 1)
            sd = float((ev.conv - icpt - slope * ev[t.main]).std(ddof=2))
            calib[t.kind] = (icpt, slope, sd)
            report["calibration"][t.kind] = {"intercept": icpt, "slope": slope, "resid_sd": sd}
    print(f"\n── Fiabilidad: ¿ordena la métrica offline como el leaderboard? ({len(usable)} versiones mapeadas)")
    rel = pd.DataFrame(report.pop("_rows", []))
    if len(rel):
        print(rel.fillna("").to_string(index=False, float_format=F2))
    for k, (i, s, sd) in calib.items():
        print(f"  calibración {k}: conv_lb ≈ {i:.2f} + {s:.2f} · {next(t.main for t in tests if t.kind == k)}"
              f"  (error típico {sd:.2f})")

    # ── candidatos
    if args.candidates:
        ev0 = lb.join(off[tests[0].kind], on="policy").dropna(subset=[tests[0].main])
        ref = args.vs or ev0.sort_values("conv", ascending=False).policy.iloc[0]
        ref_ver = next((v for v, p in ol.LIVE_VERSIONS.items() if p == ref and v in lb.index), None)
        print(f"\n── Candidatos frente a {ref}" + (f" ({ref_ver}, {lb.loc[ref_ver, 'conv']:.2f} en el lb)" if ref_ver else ""))
        crow = []
        for c in args.candidates:
            verdicts = []
            for t in tests:
                L = lists[t.kind]
                if c not in L or ref not in L:
                    continue
                d, lo, hi = ol.bootstrap_diff(t, L[c], L[ref])
                dn, lon, hin = ol.bootstrap_diff(t, L[c], L[ref], new=True)
                v = "PEOR" if hi < 0 else "MEJOR" if lo > 0 else "EMPATE"
                verdicts.append(v)
                i, s, sd = calib.get(t.kind, (np.nan, np.nan, np.nan))
                crow.append({"política": c, "test": t.kind, "métrica": t.main, "valor": off[t.kind].loc[c, t.main],
                             "Δ vs ref": d, "IC 95 %": f"[{lo:+.3f}, {hi:+.3f}]", "Δ nuevos": dn,
                             "IC nuevos": f"[{lon:+.3f}, {hin:+.3f}]",
                             "conv_lb estimada": f"{i + s * off[t.kind].loc[c, t.main]:.2f} ± {2 * sd:.2f}",
                             "veredicto": v})
                report.setdefault("candidates", {}).setdefault(c, {})[t.kind] = {
                    "score": off[t.kind].loc[c].to_dict(), "diff": [d, lo, hi], "diff_new": [dn, lon, hin],
                    "ref": ref, "verdict": v}
            if verdicts:
                if all(v == "PEOR" for v in verdicts):
                    final = "PEOR en todos los tests: descartar"
                elif all(v == "MEJOR" for v in verdicts):
                    final = "MEJOR en todos los tests: candidata; confirmar con A/B simultáneo antes de producción"
                elif "PEOR" in verdicts and "MEJOR" in verdicts:
                    rho = {t.kind: report["metrics"].get(f"{t.kind}:{t.main}", {}).get("spearman", -1) for t in tests}
                    best = max(rho, key=rho.get)
                    final = (f"los tests se contradicen: manda {best} (el que mejor ordena el leaderboard, "
                             f"Spearman {rho[best]:.2f}), que dice {verdicts[[t.kind for t in tests].index(best)]}")
                else:
                    final = "sin diferencia clara: solo el vivo decide"
                print(f"  {c}: {final}")
                report["candidates"][c]["final"] = final
        if crow:
            print(pd.DataFrame(crow).set_index(["política", "test"]).to_string(float_format=F3, na_rep="—"))

    ol.CACHE.mkdir(parents=True, exist_ok=True)
    path = ol.CACHE / f"report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nInforme → {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
