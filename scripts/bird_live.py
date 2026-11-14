"""Bird v2 (mismo CatBoost, mismas 56 features, mismos parámetros) con los datos de HOY.

    python scripts/bird.py dataset     # antes: actualiza data/bird/v2/live_v2.parquet
    python scripts/bird_live.py        # compara en las últimas horas de hoy
    python scripts/bird_live.py --save # además guarda los modelos entrenados con TODO el día (para servir)

Parte nuestro carrusel en vivo por tiempo (train / validation para la parada temprana / test) y compara,
para el clasificador (v2-bird) y para el ranker (v3-bird):
    actual     el modelo de producción, entrenado solo con el histórico (v2_live / v2_rank_live)
    solo_hoy   el mismo modelo entrenado desde cero solo con las filas de hoy
    continuado el modelo actual + árboles nuevos ajustados con las filas de hoy
Como al servir: sin productos ya comprados y con todos los candidatos en la posición de referencia.
`servido` es el orden con el que salió el carrusel (parte con ventaja: lo de arriba se compra más).
Modelos de --save en data/bird/v2_hoy, v2_rank_hoy, v2_cont y v2_rank_cont. No toca RawTree.
"""
import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import CatBoost, Pool  # noqa: E402

from reco.bird import dataset  # noqa: E402
from reco.bird.dataset import prepare  # noqa: E402
from reco.bird.model import DEFAULT_PARAMS, RANK_PARAMS, BirdModel, _describe, _rank_pool, train  # noqa: E402
from reco.config import DATA_DIR  # noqa: E402

BIRD = DATA_DIR / "bird"
CURRENT = {"binary": "v2_live", "rank": "v2_rank_live"}
OUT = {"binary": ("v2_hoy", "v2_cont"), "rank": ("v2_rank_hoy", "v2_rank_cont")}


def pool(df: pd.DataFrame, schema, objective: str) -> Pool:
    if objective == "rank":
        return _rank_pool(df, schema)
    return Pool(prepare(df, schema), label=df[schema.target].to_numpy(), cat_features=list(schema.categorical),
                weight=df["sample_weight"].to_numpy())


def continuar(base: BirdModel, tr: pd.DataFrame, va: pd.DataFrame, objective: str, params: dict,
              iterations: int | None = None) -> BirdModel:
    """Parte del modelo actual y le añade árboles ajustados con las filas de hoy."""
    params = {**DEFAULT_PARAMS, **(RANK_PARAMS if objective == "rank" else {}), **params,
              "iterations": iterations or 300}
    model = CatBoost(params)
    model.fit(pool(tr, base.schema, objective), eval_set=pool(va, base.schema, objective), init_model=base.model,
              early_stopping_rounds=None if iterations else 50, use_best_model=False, verbose=0)
    bird = BirdModel(model, base.schema, {"objective": objective})
    if not iterations:   # parada temprana a mano: use_best_model no se combina bien con init_model
        metric = "NDCG:top=10;type=Base" if objective == "rank" else "Logloss"
        curve = model.get_evals_result()["validation"]
        curve = np.array(curve.get(metric) or next(iter(curve.values())))
        best = int(curve.argmax() if objective == "rank" else curve.argmin()) + 1
        return continuar(base, tr, va, objective, params, iterations=best)
    return _describe(bird, tr, va, params, {"objective": objective, "continua": True,
                                            "arboles_nuevos": int(model.tree_count_ - base.model.tree_count_)})


def rangos(te: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    d = pd.DataFrame({"request_id": te.request_id.to_numpy(), "user_id": te.user_id.to_numpy(),
                      "y": te.target_purchase.to_numpy(), "p": score})
    d["rank"] = d.groupby("request_id", sort=False).p.rank(ascending=False, method="first")
    return d[d.y == 1].set_index("request_id")[["user_id", "rank"]]


def resumen(r: pd.DataFrame) -> dict[str, float]:
    k = r["rank"].to_numpy(float)
    return {"hit@1": (k <= 1).mean(), "hit@3": (k <= 3).mean(), "mrr": (1 / k).mean(),
            "ndcg": (1 / np.log2(k + 1)).mean()}


def bootstrap(a: pd.DataFrame, b: pd.DataFrame, n: int = 4000) -> tuple[float, float, float]:
    """NDCG de b − a en las mismas peticiones; IC 95 % remuestreando usuarios."""
    diff = 1 / np.log2(b["rank"] + 1) - 1 / np.log2(a["rank"] + 1)
    per_user = pd.DataFrame({"d": diff, "u": a.user_id}).groupby("u").d.agg(["sum", "count"]).to_numpy()
    idx = np.random.default_rng(0).integers(0, len(per_user), (n, len(per_user)))
    means = per_user[idx, 0].sum(axis=1) / per_user[idx, 1].sum(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(BIRD / "v2" / "live_v2.parquet"))
    ap.add_argument("--valid-share", type=float, default=0.15)
    ap.add_argument("--test-share", type=float, default=0.20)
    ap.add_argument("--save", action="store_true", help="guardar los modelos finales, entrenados con todo el día")
    args = ap.parse_args()

    df = dataset.load(args.data)
    total = df.request_id.nunique()
    df = df[df.already_purchased == 0].reset_index(drop=True)   # igual que al entrenar y al servir Bird v2
    rest, te = dataset.split_by_time(df, args.test_share)
    tr, va = dataset.split_by_time(rest, args.valid_share / (1 - args.test_share))
    schema = BirdModel.load(BIRD / CURRENT["binary"]).schema
    print(f"en vivo: {total:,} peticiones; sin lo ya comprado quedan {df.request_id.nunique():,}")
    for name, part in (("train", tr), ("validation", va), ("test", te)):
        print(f"{name:<11} {dataset.describe(part, schema)}")
    te1 = te.assign(position=schema.reference_position)   # como al servir: el modelo no sabe el hueco
    served = rangos(te, -te.position.to_numpy(float))
    table = {"servido": resumen(served)}
    print(f"\nTest: {len(served)} peticiones con compra")

    for objective in ("binary", "rank"):
        base = BirdModel.load(BIRD / CURRENT[objective])
        params = {k: base.info["params"][k] for k in ("depth", "learning_rate")}
        solo = train(tr, va, base.schema, params=params, objective=objective)
        cont = continuar(base, tr, va, objective, params)
        models = {"actual": base, "solo_hoy": solo, "continuado": cont}
        ranks = {name: rangos(te, m.predict(te1)) for name, m in models.items()}
        print(f"\n═══ {objective}: actual {base.model.tree_count_} árboles · solo_hoy {solo.model.tree_count_} · "
              f"continuado +{cont.info['arboles_nuevos']}")
        for name, r in ranks.items():
            table[f"{objective}/{name}"] = resumen(r)
        for label, a, b in (("solo_hoy − actual", "actual", "solo_hoy"), ("continuado − actual", "actual", "continuado")):
            d, lo, hi = bootstrap(ranks[a], ranks[b])
            verdict = "mejora" if lo > 0 else "empeora" if hi < 0 else "sin diferencia clara"
            print(f"ΔNDCG {label:<22} {d:+.4f}  IC95 [{lo:+.4f}, {hi:+.4f}]  → {verdict}")
        for name in ("actual", "solo_hoy", "continuado"):
            d, lo, hi = bootstrap(served, ranks[name])
            print(f"ΔNDCG {name + ' − servido':<22} {d:+.4f}  IC95 [{lo:+.4f}, {hi:+.4f}]")
        print("importancia solo_hoy (%): " + ", ".join(f"{k} {v:.0f}" for k, v in solo.importance().head(8).items()))
        if args.save:   # modelos finales: todo el día, con los árboles que eligió la validación
            trees = max(int(solo.model.tree_count_ * len(df) / len(tr)), 30)
            final = train(df, te, base.schema, params={**params, "iterations": trees}, early_stopping=None,
                          objective=objective)
            final.info["all_data"] = True
            final_cont = continuar(base, df, te, objective, params, iterations=max(cont.info["arboles_nuevos"], 1))
            final_cont.info["all_data"] = True
            for folder, model in zip(OUT[objective], (final, final_cont)):
                print(f"guardado → {model.save(BIRD / folder)}")

    print("\n" + pd.DataFrame(table).T.round(3).to_string())


if __name__ == "__main__":
    main()
