"""Bird (CatBoost): datos sintéticos, entrenamiento, evaluación y latencia.

    python scripts/bird.py dataset                        # dataset REAL v2 desde RawTree (data/bird/v2/)
    python scripts/bird.py train --schema v2              # entrena con él (v2_norebuy: sin features de recompra)
    python scripts/bird.py eval --schema v2 --data data/bird/v2/test_v2.parquet
    python scripts/bird.py synth --requests 20000         # dataset de prueba con el esquema del contrato (v1)
    python scripts/bird.py train --train data/bird/synth/train_v1.parquet --valid data/bird/synth/validation_v1.parquet
    python scripts/bird.py eval                           # métricas del modelo guardado sobre su validación
    python scripts/bird.py bench                          # latencia de recommend() (sin red)

Con los datos reales: los mismos `train` y `eval` apuntando a los parquet del equipo de datos.
`--until YYYY-MM-DD` guarda el modelo como data/bird/<versión>_until_<fecha>/ (el que usa `reco.evaluate --until`).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from reco.bird import dataset, metrics  # noqa: E402
from reco.bird.model import IMBALANCE, BirdModel, model_dir, train  # noqa: E402
from reco.bird.schema import SCHEMAS  # noqa: E402
from reco.config import DATA_DIR  # noqa: E402

SYNTH_DIR = DATA_DIR / "bird" / "synth"


def local_items() -> pd.DataFrame | None:
    """Catálogo real guardado en local (el más reciente), si lo hay."""
    folders = [DATA_DIR / "features_live"] + sorted(DATA_DIR.glob("features_until_*"), reverse=True)
    for folder in folders:
        if (folder / "item_features.parquet").exists():
            return pd.read_parquet(folder / "item_features.parquet")
    return None


def cmd_synth(args) -> None:
    items = local_items()
    df = dataset.synthetic(args.requests, seed=args.seed, items=items)
    tr, va = dataset.split_by_time(df, args.valid_share)
    schema = SCHEMAS["v1"]
    dataset.validate(tr, va, schema)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tr.to_parquet(out / "train_v1.parquet", index=False)
    va.to_parquet(out / "validation_v1.parquet", index=False)
    print(f"Catálogo: {'real (' + str(len(items)) + ' productos)' if items is not None else 'inventado'}")
    print(f"train       {dataset.describe(tr, schema)}\nvalidation  {dataset.describe(va, schema)}\n→ {out}")
    print("Datos SINTÉTICOS: sirven para probar el pipeline, no para sacar conclusiones del modelo.")


def data_path(args, split: str) -> Path:
    """v1 → sintético; v2 y sus variantes → data/bird/v2/."""
    if args.schema == "v1":
        return SYNTH_DIR / f"{split}_v1.parquet"
    return DATA_DIR / "bird" / "v2" / f"{split}_v2.parquet"


def cmd_dataset(args) -> None:
    from reco.bird import build_dataset as bd
    parts = bd.split(bd.build(refresh=args.refresh))
    schema = SCHEMAS["v2"]
    dataset.validate(parts["train"], parts["validation"], schema)
    dataset.validate(parts["validation"], parts["test"], schema)
    out = bd.save(parts)
    for name, part in parts.items():
        print(f"{name:<11} {dataset.describe(part, schema)}")
    print(f"→ {out}  ({len(schema.features)} features; columnas: {parts['train'].shape[1]})")


def cmd_state(args) -> None:
    """Resumen de la historia de cada usuario que usa bird_v2 al servir (reco/bird/live.py)."""
    from reco.bird import live
    print(f"→ {live.download(args.until, verbose=True)}")


def cmd_train(args) -> None:
    schema = SCHEMAS[args.schema]
    args.train, args.valid = args.train or data_path(args, "train"), args.valid or data_path(args, "validation")
    tr, va = dataset.load(args.train), dataset.load(args.valid)
    if args.all_data:
        # Modelo de producción: train + validation + test, con el nº de árboles que eligió la parada temprana
        # del modelo validado. Las métricas que salen aquí son sobre datos vistos: NO sirven para comparar.
        if not args.iterations:
            raise SystemExit("--all-data necesita --iterations (usa la 'mejor iteración' del modelo validado)")
        te = dataset.load(data_path(args, "test"))
        va, tr = te, __import__("pandas").concat([tr, va, te], ignore_index=True)
        args.force = True
    problems = dataset.validate(tr, va, schema, strict=not args.force)
    for p in problems:
        print(f"[contrato] {p}")
    print(f"train       {dataset.describe(tr, schema)}\nvalidation  {dataset.describe(va, schema)}")
    if args.drop_purchased:
        # En vivo lo ya comprado convierte ~9x menos y nunca se recomienda: el modelo aprende a elegir
        # entre lo que el usuario todavía no tiene (igual que el ranker LightGBM).
        tr, va = tr[tr.already_purchased == 0], va[va.already_purchased == 0]
        print(f"sin productos ya comprados: train {dataset.describe(tr, schema)}")
    cold = args.cold_share if args.cold_share is not None else (0.0 if args.schema == "v1" else 0.3)
    if cold > 0:
        from reco.bird.build_dataset import simulate_cold
        tr = simulate_cold(tr, cold)
        print(f"cold start: en el {cold:.0%} de las peticiones de train se esconde la historia del usuario")
    params = {k: v for k, v in (("iterations", args.iterations), ("depth", args.depth),
                                ("learning_rate", args.learning_rate), ("thread_count", args.threads))
              if v is not None}
    t = time.time()
    bird = train(tr, va, schema, params=params, imbalance=args.imbalance, verbose=args.verbose,
                 early_stopping=None if args.all_data else 50)
    bird.info["all_data"] = bool(args.all_data)
    bird.info["data"] = {"train": str(args.train), "validation": str(args.valid)}
    folder = bird.save(Path(args.out) if args.out else model_dir(schema.version, args.until))
    print(f"\nEntrenado en {time.time() - t:.1f}s · {bird.info['trees']} árboles (mejor iteración "
          f"{bird.info['best_iteration']}) · desbalanceo: {args.imbalance}")
    print(metrics.show(bird.info["metrics"]))
    print("── importancia de las features (%)")
    print(bird.importance().round(1).head(args.top).to_string())
    # el artefacto recargado tiene que dar exactamente lo mismo
    again = BirdModel.load(folder)
    same = np.allclose(bird.predict(va.head(5000)), again.predict(va.head(5000)))
    print(f"\nModelo → {folder}  (recargado e idéntico: {'sí' if same else 'NO'})")
    if not same:
        raise SystemExit("El modelo recargado no reproduce las predicciones")


def cmd_eval(args) -> None:
    bird = BirdModel.load(Path(args.model) if args.model else model_dir(args.schema, args.until))
    path = args.data or bird.info.get("data", {}).get("validation")
    if not path:
        raise SystemExit("Indica el dataset con --data")
    df = dataset.load(path)
    for p in dataset.check(df, bird.schema, "datos"):
        print(f"[contrato] {p}")
    print(f"Modelo {bird.schema.version} ({bird.info.get('trees', '?')} árboles) sobre {path}")
    print(dataset.describe(df, bird.schema))
    p = bird.predict(df)
    print(metrics.show(metrics.report(df, p, bird.schema)))
    print("── calibración por deciles de probabilidad predicha")
    w = df["sample_weight"].to_numpy() if "sample_weight" in df.columns else None
    print(metrics.calibration_table(df[bird.schema.target].to_numpy(), p, w=w)
          .to_string(float_format=lambda x: f"{x:.5f}"))
    print("\nEl ranking de arriba es SOLO entre los 10 productos enseñados; sobre el catálogo completo:\n"
          "  python -m reco.evaluate --policies bird_v1 personal")


def cmd_bench(args) -> None:
    from reco.policies.bird import Bird
    from reco.safety import finalize
    from reco.state import State

    items = local_items()
    if items is None:
        raise SystemExit("No hay catálogo local (data/features_*/item_features.parquet). Ejecuta scripts/build_features.py")
    rows = items[["product_id", "category", "subcategory", "brand", "price_eur"]].to_dict("records")
    catalog = {r["product_id"]: {**r, "is_active": True} for r in rows}
    best = items.sort_values("purchases", ascending=False).product_id.head(50).tolist()
    state = State(catalog=catalog, unavailable=set(items.product_id.sample(100, random_state=0)), best=best)
    folder = Path(args.model) if args.model else model_dir(args.schema, args.until)

    def run(per_request: bool, n: int) -> tuple[np.ndarray, float, list[str]]:
        policy = Bird()
        policy.folder, policy.force_per_request = folder, per_request
        t = time.perf_counter()
        policy.fit(state)
        fit_s = time.perf_counter() - t
        times, recs = [], []
        for i in range(n):
            req = {"request_id": f"bench_{i}", "user_id": f"u_{i}", "is_new_user": False, "recent_views": [],
                   "context": {}}
            t = time.perf_counter()
            recs = finalize(policy.recommend(req, state), state)
            times.append(time.perf_counter() - t)
            assert len(recs) == 10 and len(set(recs)) == 10 and all(state.is_valid(p) for p in recs)
        return np.array(times) * 1000, fit_s, recs

    print(f"Catálogo: {len(catalog)} productos ({len(state.unavailable)} sin stock) · modelo: {folder}")
    for label, per_request, n in (("estático (el que usa V1 en vivo)", False, args.n),
                                  ("por petición (cuando haya features de usuario)", True, max(args.n // 10, 20))):
        ms, fit_s, recs = run(per_request, n)
        print(f"── {label}: {n} peticiones · fit {fit_s:.2f}s")
        print(f"   recommend+finalize  p50 {np.percentile(ms, 50):.2f} ms · p95 {np.percentile(ms, 95):.2f} ms · "
              f"máx {ms.max():.2f} ms")
    print(f"Ejemplo de respuesta: {recs}")
    print("Límite del reto: 3.000 ms por petición, incluida la red (lectura de shop.requests + inserción).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="genera train/validation sintéticos con el esquema V1")
    s.add_argument("--requests", type=int, default=20000)
    s.add_argument("--valid-share", type=float, default=0.2)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default=str(SYNTH_DIR))
    s.set_defaults(fn=cmd_synth)

    d = sub.add_parser("dataset", help="construye el dataset real v2 desde RawTree")
    d.add_argument("--refresh", action="store_true", help="volver a descargar todo (ignora data/bird/raw/)")
    d.set_defaults(fn=cmd_dataset)

    st = sub.add_parser("state", help="descarga el estado de usuarios para servir bird_v2")
    st.add_argument("--until", help="corte para la evaluación offline (YYYY-MM-DD); sin él, en vivo")
    st.set_defaults(fn=cmd_state)

    t = sub.add_parser("train", help="entrena y guarda el modelo")
    t.add_argument("--train", help="parquet de entrenamiento (por defecto, el del esquema)")
    t.add_argument("--valid", help="parquet de validación (por defecto, el del esquema)")
    t.add_argument("--cold-share", type=float,
                   help="parte de las peticiones de train tratadas como usuario nuevo (por defecto 0,3 en v2)")
    t.add_argument("--drop-purchased", action="store_true",
                   help="descartar las impresiones de productos que el usuario ya había comprado")
    t.add_argument("--threads", type=int)
    t.add_argument("--all-data", action="store_true",
                   help="modelo final: entrena con train+validation+test y --iterations fijas")
    t.add_argument("--top", type=int, default=25, help="cuántas features enseñar en la importancia")
    t.add_argument("--imbalance", choices=sorted(IMBALANCE), default="none",
                   help="reponderar las compras (descalibra las probabilidades; por defecto no)")
    t.add_argument("--iterations", type=int)
    t.add_argument("--depth", type=int)
    t.add_argument("--learning-rate", type=float)
    t.add_argument("--out", help="carpeta del modelo (por defecto data/bird/<versión>[_until_<fecha>])")
    t.add_argument("--force", action="store_true", help="entrenar aunque el dataset no cumpla el contrato")
    t.add_argument("--verbose", type=int, default=0, help="imprimir el progreso cada N iteraciones")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval", help="métricas del modelo guardado")
    e.add_argument("--data", help="parquet a evaluar (por defecto, la validación con la que se entrenó)")
    e.add_argument("--model", help="carpeta del modelo")
    e.set_defaults(fn=cmd_eval)

    b = sub.add_parser("bench", help="latencia de recommend() con el catálogo local, sin red")
    b.add_argument("-n", type=int, default=2000)
    b.add_argument("--model", help="carpeta del modelo")
    b.set_defaults(fn=cmd_bench)

    for p in (t, e, b):
        p.add_argument("--schema", choices=sorted(SCHEMAS), default="v1", help="versión del dataset/modelo")
        p.add_argument("--until", help="corte de la evaluación offline (YYYY-MM-DD)")
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
