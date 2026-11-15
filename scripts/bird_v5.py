"""Bird v5: dataset con features nuevas (histórico + feedback en vivo), entrenamiento y comparación con v2.

    python scripts/bird_v5.py dataset                 # data/bird/v5/{train,validation,test,live}_v5.parquet
    python scripts/bird_v5.py dataset --refresh-live  # vuelve a bajar solo lo de hoy (el histórico no cambia)
    python scripts/bird_v5.py compare                 # mismas filas, features de v2 frente a las de v5
    python scripts/bird_v5.py train --name v5_fresh   # modelo para evaluar offline (solo histórico < 07/11)
    python scripts/bird_v5.py train --name v5_live --final   # modelo de producción (histórico + todo el vivo)
    python scripts/bird_v5.py state [--until CORTE]   # estado para servir (reco/bird/live5.py)
    python scripts/bird_v5.py parity                  # las features al servir coinciden con las del dataset

Las features y su justificación están en docs/DATASET_V5.md; las fórmulas, en reco/bird/feat5.py.
"""
import argparse
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from reco.bird import dataset, metrics  # noqa: E402
from reco.bird.model import BirdModel, train  # noqa: E402
from reco.bird.schema import SCHEMAS, V2_NOREBUY, V5  # noqa: E402
from reco.config import DATA_DIR  # noqa: E402

BIRD = DATA_DIR / "bird"
V5_DIR = BIRD / "v5"
PARAMS = {"depth": 6, "learning_rate": 0.1, "iterations": 800, "thread_count": 6}


def load(*names: str) -> pd.DataFrame:
    return pd.concat([dataset.load(V5_DIR / f"{n}_v5.parquet") for n in names], ignore_index=True)


def cmd_dataset(args) -> None:
    from reco.bird import build_v5 as b5
    if args.refresh_live:   # lo de hoy cambia cada minuto; el histórico no
        for f in list(b5.RAW.glob("ui_live_*")) + list(b5.RAW.glob("ui_org_*")) + [b5.RAW / n for n in (
                "live_until.json", "live_sessions.pkl", "org_sessions.pkl", "org_pairs.parquet", "item_live10.parquet", "feedback10.parquet", "catalog.pkl",
                "updates.parquet", "ctx_device.parquet", "ctx_region.parquet")]:
            f.unlink(missing_ok=True)
    parts = b5.split(b5.build(refresh=args.refresh, parts=tuple(args.parts)))
    if "train" in parts:
        dataset.validate(parts["train"], parts["validation"], V5)
        dataset.validate(parts["validation"], parts["test"], V5)
    if "live" in parts:
        for p in dataset.check(parts["live"], V5, "live"):
            print(f"[contrato] {p}")
    out = b5.save(parts)
    for name, part in parts.items():
        print(f"{name:<11} {dataset.describe(part, V5)}")
    if "org" in parts:
        o = parts["org"]
        print(f"org: {o.target_purchase.mean():.3f} de las filas son compras · ya comprado antes: "
              f"{o[o.target_purchase == 1].already_purchased.mean():.3f} de las compras · compra de algo recién visto: "
              f"{o[o.target_purchase == 1].recently_viewed.mean():.3f}")
    print(f"→ {out}  ({len(V5.features)} features)")


def no_rebuy(df: pd.DataFrame) -> pd.DataFrame:
    """Como al servir: lo que el usuario ya compró nunca se recomienda, así que no entra al entrenar."""
    return df[df.already_purchased == 0].reset_index(drop=True)


def live_split(live: pd.DataFrame, valid_share: float, test_share: float):
    """Parte en vivo por tiempo: train / validation (parada temprana) / test (lo último del día)."""
    rest, te = dataset.split_by_time(live, test_share)
    tr, va = dataset.split_by_time(rest, valid_share / (1 - test_share))
    return tr, va, te


def rank_report(te: pd.DataFrame, score: np.ndarray, target: str = "target_purchase") -> dict[str, float]:
    """Dónde queda el producto comprado (o pulsado) entre los 10 del carrusel según `score`."""
    d = pd.DataFrame({"r": te.request_id.to_numpy(), "y": te[target].to_numpy(), "s": score})
    d["rank"] = d.groupby("r", sort=False).s.rank(ascending=False, method="first")
    k = d.loc[d.y == 1, "rank"].to_numpy(float)
    return {"n": len(k), "hit@1": (k <= 1).mean(), "hit@3": (k <= 3).mean(), "mrr": (1 / k).mean(),
            "ndcg": (1 / np.log2(k + 1)).mean()}


def paired_ndcg(te: pd.DataFrame, a: np.ndarray, b: np.ndarray, target: str, n: int = 2000) -> tuple[float, float, float]:
    """NDCG de b − a en las mismas peticiones; IC 95 % remuestreando usuarios."""
    def gains(score):
        d = pd.DataFrame({"r": te.request_id.to_numpy(), "u": te.user_id.to_numpy(), "y": te[target].to_numpy(), "s": score})
        d["rank"] = d.groupby("r", sort=False).s.rank(ascending=False, method="first")
        d = d[d.y == 1]
        return d.assign(g=1 / np.log2(d["rank"] + 1)).set_index("r")[["u", "g"]]
    ga, gb = gains(a), gains(b)
    diff = (gb.g - ga.g)
    per_user = pd.DataFrame({"d": diff, "u": ga.u}).groupby("u").d.agg(["sum", "count"]).to_numpy()
    idx = np.random.default_rng(0).integers(0, len(per_user), (n, len(per_user)))
    means = per_user[idx, 0].sum(axis=1) / per_user[idx, 1].sum(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fit(schema, tr: pd.DataFrame, va: pd.DataFrame, cold: float, live_weight: float, iterations: int | None = None,
        threads: int = 6, ips_clip: float = 0.0, org_weight: float = 1.0) -> BirdModel:
    from reco.bird.build_v5 import simulate_cold
    tr = simulate_cold(tr, cold)
    org = (tr.source == "org").to_numpy()
    if live_weight != 1.0 or org.any():
        w = np.where(tr.source == "live", live_weight, 1.0)
        tr = tr.assign(sample_weight=tr.sample_weight * w)
    if org.any():
        from reco.bird.build_v5 import ORG_NEG_MOD
        # Filas orgánicas: dos correcciones para que no choquen con las del carrusel.
        #  1. Cada negativo es 1 de cada ORG_NEG_MOD productos del catálogo: pesa ORG_NEG_MOD. Sin esto la
        #     tasa de compra de estas filas (6,7 %) es ~60 veces la real y descalibra todo el modelo.
        #  2. No tienen hueco de carrusel: position va vacía. CatBoost trata el vacío como un valor aparte,
        #     así que el modelo puede separar "orgánico" de "carrusel" sin una feature nueva; al servir,
        #     position = 1 y manda lo aprendido del carrusel.
        w = np.where(org, org_weight * np.where(tr.target_purchase.to_numpy() == 0, float(ORG_NEG_MOD), 1.0), 1.0)
        tr = tr.assign(sample_weight=tr.sample_weight * w,
                       position=np.where(org, np.nan, tr.position.to_numpy(dtype=float)))
    if ips_clip > 0:
        # El sistema viejo enseñaba sobre todo best sellers. Pesar cada impresión del histórico por
        # 1/propensity (acotado y normalizado a media 1) la acerca a "todos los productos por igual", que
        # es como se usa el modelo: puntuando el catálogo entero.
        hist = (tr.source == "hist").to_numpy()
        w = np.clip(1.0 / np.clip(tr.propensity.to_numpy(dtype=float), 1e-3, None), None, ips_clip)
        w = np.where(hist, w / w[hist].mean(), 1.0)
        tr = tr.assign(sample_weight=tr.sample_weight * w)
    params = {**PARAMS, "thread_count": threads, **({"iterations": iterations} if iterations else {})}
    return train(tr, va, schema, params=params, early_stopping=None if iterations else 50)


def cmd_compare(args) -> None:
    """Mismas filas y mismas etiquetas; lo único que cambia es la lista de features (v2_norebuy frente a v5).

    A. solo histórico (train < 31/10, parada temprana 31/10–07/11) → test histórico 07–14/11
    B. histórico + vivo(primeras horas) → test = últimas horas de NUESTRO carrusel de hoy (el simulador de verdad)
    En el test todos los candidatos van con la posición de referencia, como al servir."""
    hist_tr, hist_va, hist_te, live = (no_rebuy(load(n)) for n in ("train", "validation", "test", "live"))
    live_tr, live_va, live_te = live_split(live, args.valid_share, args.test_share)
    schemas = {"v2_norebuy": V2_NOREBUY, "v5": V5}
    for extra in args.schemas:
        schemas[extra] = SCHEMAS[extra]
    print(f"hist  train {dataset.describe(hist_tr, V5)}\n      valid {dataset.describe(hist_va, V5)}\n"
          f"      test  {dataset.describe(hist_te, V5)}")
    print(f"live  train {dataset.describe(live_tr, V5)}\n      valid {dataset.describe(live_va, V5)}\n"
          f"      test  {dataset.describe(live_te, V5)}")
    rows, scores = [], {}
    setups = {
        "A hist→hist": (hist_tr, hist_va, hist_te, 1.0),
        "A hist→live": (hist_tr, hist_va, live_te, 1.0),
        "B hist+live→live": (pd.concat([hist_tr, hist_va, hist_te, live_tr], ignore_index=True), live_va, live_te,
                             args.live_weight),
    }
    models = {}
    for sname, schema in schemas.items():
        for setup, (tr, va, te, lw) in setups.items():
            key = (sname, setup.split("→")[0])
            if key not in models:
                t = time.time()
                models[key] = fit(schema, tr, va, args.cold_share, lw, threads=args.threads)
                print(f"  {sname:<11} {key[1]:<12} {models[key].info['trees']} árboles  {time.time() - t:.0f}s")
            m = models[key]
            te1 = te.assign(position=schema.reference_position)
            p = m.predict(te1)
            for target in ("target_purchase", "target_click"):
                scores[(sname, setup, target)] = p
                r = rank_report(te, p, target)
                rows.append({"features": sname, "protocolo": setup, "objetivo": target.split("_")[1], **r})
    print("\n" + pd.DataFrame(rows).set_index(["protocolo", "objetivo", "features"]).sort_index().round(4).to_string())
    print("\nΔNDCG frente a v2_norebuy (IC 95 % por bootstrap de usuarios):")
    for setup, (_, _, te, _) in setups.items():
        for target in ("target_purchase", "target_click"):
            for sname in schemas:
                if sname == "v2_norebuy":
                    continue
                d, lo, hi = paired_ndcg(te, scores[("v2_norebuy", setup, target)], scores[(sname, setup, target)], target)
                verdict = "MEJOR" if lo > 0 else "PEOR" if hi < 0 else "empate"
                print(f"  {setup:<18} {target.split('_')[1]:<9} {sname:<12} {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {verdict}")
    best = models[("v5", "B hist+live")]
    print("\n── importancia de las features de v5 (modelo B, %)")
    print(best.importance().round(2).head(args.top).to_string())


def org_before(org: pd.DataFrame, cut) -> pd.DataFrame:
    """Filas orgánicas cuya ventana de compra (30 min) acaba antes de `cut`: no ven nada del periodo de validación."""
    from reco.bird.build_v5 import ORG_WINDOW
    return org[org.timestamp < cut - pd.Timedelta(minutes=ORG_WINDOW)]


def cmd_compare_org(args) -> None:
    """¿Ayuda añadir las compras orgánicas al entrenamiento? Mismo modelo y mismas features; lo único que
    cambia son las filas. Se mide donde importa: ordenar los 10 productos de NUESTRO carrusel en las últimas
    horas (que ningún modelo ha visto). Las filas orgánicas usadas son anteriores a ese periodo."""
    hist = pd.concat([no_rebuy(load(n)) for n in ("train", "validation", "test")], ignore_index=True)
    live, org = no_rebuy(load("live")), no_rebuy(load("org"))
    live_tr, live_va, live_te = live_split(live, args.valid_share, args.test_share)
    org_tr = org_before(org, live_va.timestamp.min())
    org_te = org[org.timestamp >= live_te.timestamp.min()]
    print(f"live  train {dataset.describe(live_tr, V5)}\n      valid {dataset.describe(live_va, V5)}\n"
          f"      test  {dataset.describe(live_te, V5)}")
    print(f"org   train {len(org_tr):,} filas · {org_tr.request_id.nunique():,} peticiones · "
          f"{int(org_tr.target_purchase.sum()):,} compras · hasta {org_tr.timestamp.max():%H:%M}\n"
          f"      test  {len(org_te):,} filas · {int(org_te.target_purchase.sum()):,} compras")
    base_tr = pd.concat([hist, live_tr], ignore_index=True)
    te1 = live_te.assign(position=V5.reference_position)
    scores, rows = {}, []
    setups = {"sin orgánicas": None, **{f"orgánicas ×{w:g}": w for w in args.org_weights}}
    for name, w in setups.items():
        t = time.time()
        tr = base_tr if w is None else pd.concat([base_tr, org_tr], ignore_index=True)
        m = fit(V5, tr, live_va, args.cold_share, args.live_weight, threads=args.threads, org_weight=w or 1.0)
        scores[name] = m.predict(te1)
        print(f"  {name:<16} {m.info['trees']} árboles  {time.time() - t:.0f}s")
        for target in ("target_purchase", "target_click"):
            rows.append({"modelo": name, "test": f"carrusel · {target.split('_')[1]}", **rank_report(live_te, scores[name], target)})
        if len(org_te):
            rows.append({"modelo": name, "test": "orgánico · compra", **rank_report(org_te, m.predict(org_te), "target_purchase")})
    print("\n" + pd.DataFrame(rows).set_index(["test", "modelo"]).sort_index().round(4).to_string())
    print("\nΔNDCG en nuestro carrusel frente a «sin orgánicas» (IC 95 % por bootstrap de usuarios):")
    for name in list(setups)[1:]:
        for target in ("target_purchase", "target_click"):
            d, lo, hi = paired_ndcg(live_te, scores["sin orgánicas"], scores[name], target)
            verdict = "MEJOR" if lo > 0 else "PEOR" if hi < 0 else "empate"
            print(f"  {name:<16} {target.split('_')[1]:<9} {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {verdict}")


def cmd_train(args) -> None:
    schema = SCHEMAS[args.schema]
    hist_tr, hist_va, hist_te = (no_rebuy(load(n)) for n in ("train", "validation", "test"))
    if args.final:
        # producción: todo el histórico + todo el vivo. Nº de árboles: el que eligió la parada temprana con
        # las últimas horas del vivo como validación, escalado a los datos de más.
        live = no_rebuy(load("live"))
        live_tr, live_va = dataset.split_by_time(live, args.valid_share)
        extra_tr = extra_all = []
        if args.org_weight > 0:   # compras orgánicas: en la pasada de prueba, solo las anteriores a la validación
            org = no_rebuy(load("org"))
            extra_tr, extra_all = [org_before(org, live_va.timestamp.min())], [org]
            print(f"filas orgánicas: {len(org):,} ({int(org.target_purchase.sum()):,} compras), peso ×{args.org_weight:g}")
        if args.trees:   # nº de árboles fijo: sin pasada de prueba (la parada temprana a veces corta en decenas)
            trees = args.trees
            print(f"nº de árboles fijado a mano: {trees} (sin pasada de prueba)")
        else:
            probe = fit(schema, pd.concat([hist_tr, hist_va, hist_te, live_tr, *extra_tr], ignore_index=True), live_va,
                        args.cold_share, args.live_weight, threads=args.threads, ips_clip=args.ips_clip,
                        org_weight=args.org_weight or 1.0)
            trees = max(int(probe.info["trees"] * 1.05), 50)
            print(f"parada temprana (validación: últimas horas en vivo): {probe.info['trees']} árboles → final {trees}")
            print(metrics.show(probe.info["metrics"]))
        bird = fit(schema, pd.concat([hist_tr, hist_va, hist_te, live, *extra_all], ignore_index=True), live_va,
                   args.cold_share, args.live_weight, iterations=trees, threads=args.threads, ips_clip=args.ips_clip,
                   org_weight=args.org_weight or 1.0)
        bird.info["all_data"], bird.info["org_weight"] = True, args.org_weight
    else:
        # evaluación offline (make offline): solo datos anteriores al 07/11, como v2_fresh
        bird = fit(schema, hist_tr, hist_va, args.cold_share, 1.0, threads=args.threads, ips_clip=args.ips_clip)
        print(metrics.show(bird.info["metrics"]))
    bird.info["cold_share"], bird.info["live_weight"], bird.info["ips_clip"] = args.cold_share, args.live_weight, args.ips_clip
    folder = bird.save(BIRD / args.name)
    print(f"{bird.info['trees']} árboles → {folder}")
    print("── importancia de las features (%)")
    print(bird.importance().round(2).head(args.top).to_string())


def cmd_offline(args) -> None:
    """Los dos tests de `make offline` (reco/offline_live.py), solo para las políticas indicadas: la primera
    es la referencia. Más rápido que `make offline`, que además recalcula todas las versiones servidas."""
    from reco import offline_live as ol
    from reco.safety import finalize
    tests = []
    if "history" in args.tests:
        tests.append(ol.TestSet.load(args.until))
    if "organic" in args.tests:
        tests.append(ol.OrganicTestSet.load(args.organic_cut, window=30, end=args.organic_end))
    ref = args.policies[0]
    for t in tests:
        lists = {p: ol.policy_lists(p, t) for p in args.policies}
        table = pd.DataFrame({p: ol.scores(t, ls) for p, ls in lists.items()}).T
        print(f"\n── test {t.kind} ({t.n_labels} compras) · métrica principal: {t.main}")
        print(table.to_string(float_format=lambda v: f"{v:.4f}"))
        for p in args.policies[1:]:
            for new, label in ((False, "todos"), (True, "nuevos")):
                d, lo, hi = ol.bootstrap_diff(t, lists[p], lists[ref], new=new)
                verdict = "MEJOR" if lo > 0 else "PEOR" if hi < 0 else "EMPATE"
                print(f"   {p} − {ref} [{label:<6}] Δ{t.main} {d:+.4f}  IC 95 % [{lo:+.4f}, {hi:+.4f}]  → {verdict}")
    if args.bench:   # latencia de recommend + finalize con peticiones reales del último test
        from reco.policies import POLICIES
        t = tests[-1]
        rows = list(t.df.itertuples(index=False))[:args.bench]
        for p in args.policies:
            policy = POLICIES[p](until=t.until)
            policy.fit(t.state)
            ms = []
            for row in rows:
                t0 = time.perf_counter()
                finalize(policy.recommend(t.request(row, False), t.state), t.state)
                ms.append((time.perf_counter() - t0) * 1000)
            print(f"   latencia {p}: p50 {np.percentile(ms, 50):.1f} ms · p95 {np.percentile(ms, 95):.1f} ms · "
                  f"máx {max(ms):.1f} ms ({len(ms)} peticiones)")


def cmd_state(args) -> None:
    from reco.bird import live5
    print(f"→ {live5.download(args.until, verbose=True)}")


def cmd_parity(args) -> None:
    from reco.bird import live5
    live5.parity_report(args.until, minutes=args.minutes, n=args.n, base=args.base, delta=not args.no_delta)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("--refresh", action="store_true", help="volver a descargar todo")
    d.add_argument("--refresh-live", action="store_true", help="volver a descargar solo la parte de hoy")
    d.add_argument("--parts", nargs="+", default=["hist", "live"], choices=["hist", "live", "org"],
                   help="org = compras orgánicas tras cada petición + negativos al azar (no depende del carrusel)")
    d.set_defaults(fn=cmd_dataset)
    c = sub.add_parser("compare")
    c.add_argument("--schemas", nargs="*", default=[], help="esquemas extra a comparar")
    c.set_defaults(fn=cmd_compare)
    t = sub.add_parser("train")
    t.add_argument("--name", required=True, help="carpeta del modelo en data/bird/")
    t.add_argument("--schema", default="v5", choices=sorted(SCHEMAS))
    t.add_argument("--final", action="store_true", help="modelo de producción: histórico + vivo")
    t.add_argument("--trees", type=int, default=0, help="con --final: nº de árboles fijo, sin parada temprana")
    t.add_argument("--ips-clip", type=float, default=0.0,
                   help="pesar el histórico por 1/propensity acotado a este valor (0 = no)")
    t.add_argument("--org-weight", type=float, default=0.0,
                   help="con --final: añadir las filas orgánicas con este peso (0 = no usarlas)")
    t.set_defaults(fn=cmd_train)
    co = sub.add_parser("compare-org", help="mismo modelo con y sin filas orgánicas, medido en nuestro carrusel")
    co.add_argument("--org-weights", type=float, nargs="+", default=[0.1, 0.03])
    co.set_defaults(fn=cmd_compare_org)
    for p in (c, t, co):
        p.add_argument("--cold-share", type=float, default=0.3)
        p.add_argument("--live-weight", type=float, default=3.0, help="peso de las filas en vivo frente a las del histórico")
        p.add_argument("--valid-share", type=float, default=0.15)
        p.add_argument("--test-share", type=float, default=0.25)
        p.add_argument("--threads", type=int, default=6)
        p.add_argument("--top", type=int, default=40)
    o = sub.add_parser("offline")
    o.add_argument("--policies", nargs="+", default=["bird_v2", "bird_v5"], help="la primera es la referencia")
    o.add_argument("--tests", nargs="+", default=["history", "organic"], choices=["history", "organic"])
    o.add_argument("--until", default="2026-11-07")
    o.add_argument("--organic-cut", default="2026-11-14 12:00:00")
    o.add_argument("--organic-end", default="2026-11-14 20:00:00")
    o.add_argument("--bench", type=int, default=300, help="peticiones para medir la latencia (0 = no medir)")
    o.set_defaults(fn=cmd_offline)
    s = sub.add_parser("state")
    s.add_argument("--until")
    s.set_defaults(fn=cmd_state)
    q = sub.add_parser("parity")
    q.add_argument("--until", default="2026-11-14 20:00:00")
    q.add_argument("--minutes", type=float, default=5.0)
    q.add_argument("--n", type=int, default=300)
    q.add_argument("--base", help="descarga completa anterior a --until; se le suma el delta hasta --until")
    q.add_argument("--no-delta", action="store_true", help="con --base: sin delta (estado viejo, como antes)")
    q.set_defaults(fn=cmd_parity)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
