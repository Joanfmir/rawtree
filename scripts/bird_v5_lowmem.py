"""Reentrenar el modelo de v5-bird / v6-bird con el vivo al día, en una máquina con poca memoria.

    python scripts/bird_v5_lowmem.py live                   # vuelve a bajar lo de hoy → data/bird/v5/live_v5.parquet
    python scripts/bird_v5_lowmem.py train --name v5_live2  # mismo procedimiento que `bird_v5.py train --final`

`live`: lo mismo que `bird_v5.py dataset --refresh-live --parts live`, por lotes de peticiones (el mismo
`build_v5._frame`). El histórico no cambia: se reutilizan {train,validation,test}_v5.parquet.
`train`: las filas pasan del Parquet a un TSV por lotes y CatBoost lee el TSV, en vez de cargar 1,3 M de filas.
  1. prueba: histórico + vivo menos sus últimas peticiones (`--valid-share`), parada temprana con ellas
  2. final:  todo, con el nº de árboles de la prueba × 1,05
Sin recompra, cold start simulado en el histórico y filas en vivo × `--live-weight`, como v5_live.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from reco.bird import build_v5 as b5  # noqa: E402
from reco.bird.schema import V5  # noqa: E402
from reco.config import DATA_DIR  # noqa: E402

OUT = DATA_DIR / "bird" / "v5"
LIVE_CACHE = ("live_until.json", "live_sessions.pkl", "item_live10.parquet", "feedback10.parquet", "catalog.pkl",
              "updates.parquet", "ctx_device.parquet", "ctx_region.parquet")


def cmd_live(args) -> None:
    if not args.keep_cache:   # lo de hoy cambia cada minuto: fuera la caché de la parte en vivo
        for f in list(b5.RAW.glob("ui_live_*")) + [b5.RAW / n for n in LIVE_CACHE]:
            f.unlink(missing_ok=True)
    until = b5.live_until()
    cat = b5.load_catalog()
    items = b5.feat5.Items(cat.assign(created_at=pd.to_datetime(cat.created_at)))
    updates = b5.load_updates()
    print(f"corte en vivo {until} · catálogo {items.n:,} productos "
          f"({int((items.created >= np.datetime64(b5.LIVE_START)).sum())} nuevos del stream)", flush=True)
    hourly = b5._chunks("item_hourly", 4, lambda k: b5.item_hourly_sql(k, 4), False, print, workers=1)
    live10 = b5._cached("item_live10.parquet", lambda: pd.DataFrame(b5._query(b5.ITEM_LIVE10.format(until=until))))
    tl = b5.Timeline(items, hourly, live10, until)
    del hourly, live10
    fb = b5._cached("feedback10.parquet", lambda: pd.DataFrame(b5._query(b5.FEEDBACK10.format(team=b5.TEAM, until=until))))
    fbt = b5.FeedbackTimeline(items, fb, until)
    del fb
    ctx = b5.context_tables(items, False, until)
    daily = b5.bd.load_item_daily()
    sess = b5.load_live_sessions(until)
    sess["ts"] = pd.to_datetime(sess.ts)
    ui = b5._chunks("ui_live", b5.LIVE_CHUNKS, lambda k: b5.user_item_sql(k, b5.LIVE_CHUNKS, live=True, until=until),
                    False, print, workers=args.workers).set_index("request_id")
    print(f"live: {len(sess):,} peticiones ({(sess.clicked != '').sum():,} con clic, {(sess.purchased != '').sum():,} "
          f"con compra) de {sess.ts.min()} a {sess.ts.max()}", flush=True)
    gc.collect()

    cache, original = {}, b5.neighbors          # los vecinos de un corte se construyen una vez, no en cada lote

    def neighbors(cut, it):
        if cut not in cache:
            cache[cut] = original(cut, it)
        return cache[cut]
    b5.neighbors = neighbors

    OUT.mkdir(parents=True, exist_ok=True)
    tmp, writer, rows = OUT / "live_v5.parquet.tmp", None, 0
    for i in range(0, len(sess), args.batch):
        batch = sess.iloc[i:i + args.batch].copy()
        part = ui.loc[ui.index.intersection(batch.request_id)].reset_index()
        frame = b5._frame(batch, "live", items, part, tl, fbt, ctx, daily, updates, b5.live_cuts(until), lambda *a, **k: None)
        frame = frame.sort_values(["timestamp", "request_id", "position"])
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
        writer.write_table(table.cast(writer.schema))
        rows += len(frame)
        print(f"live: {min(i + args.batch, len(sess)):,}/{len(sess):,} peticiones", flush=True)
        del frame, batch, part, table
        gc.collect()
    writer.close()
    tmp.replace(OUT / "live_v5.parquet")
    print(f"live  {rows:,} filas → {OUT / 'live_v5.parquet'}")


def cmd_train(args) -> None:
    from catboost import CatBoostClassifier, Pool

    from reco.bird import metrics
    from reco.bird.dataset import prepare
    from reco.bird.model import DEFAULT_PARAMS, BirdModel
    schema = V5
    work = OUT / "tsv"
    work.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(OUT / "live_v5.parquet", columns=["request_id", "timestamp", "already_purchased"])
    first = meta[meta.already_purchased == 0].groupby("request_id").timestamp.min().sort_values()
    cut = first.iloc[int(len(first) * (1 - args.valid_share))]
    del meta, first
    feats, cats = list(schema.features), set(schema.categorical)
    cd = ["0\tLabel\t" + schema.target] + [f"{i}\t{'Categ' if f in cats else 'Num'}\t{f}" for i, f in enumerate(feats, 1)]
    cd.append(f"{len(feats) + 1}\tWeight\tsample_weight")
    (work / "pool.cd").write_text("\n".join(cd) + "\n", encoding="utf-8")
    need = list(dict.fromkeys(feats + ["request_id", "timestamp", "source", "already_purchased", "sample_weight", schema.target]))
    zero, nan = [c for c in b5._COLD_ZERO if c in feats], [c for c in b5._COLD_NAN if c in feats]
    stats = {k: {"filas": 0, "compras": 0, "desde": None, "hasta": None} for k in ("fit", "valid")}
    weight = {"hist": 0.0, "live": 0.0}
    valid_parts = []
    with (work / "fit.tsv").open("w", encoding="utf-8", newline="") as f_fit, \
            (work / "valid.tsv").open("w", encoding="utf-8", newline="") as f_val:
        for part in ("train", "validation", "test", "live"):
            for batch in pq.ParquetFile(OUT / f"{part}_v5.parquet").iter_batches(batch_size=20000, columns=need):
                d = batch.to_pandas()
                d = d[d.already_purchased == 0].copy()                 # como al servir: nunca lo ya comprado
                if not len(d):
                    continue
                live = (d.source == "live").to_numpy()
                h = pd.util.hash_pandas_object(d.request_id.astype("string"), index=False).to_numpy(dtype="uint64")
                cold = ~live & ((h % 10000) < int(args.cold_share * 10000))   # cold start simulado (por petición)
                d.loc[cold, zero] = 0
                d.loc[cold, nan] = np.nan
                d.loc[cold, "is_new_user"] = 1
                d["sample_weight"] = d.sample_weight * np.where(live, args.live_weight, 1.0)
                weight["live"] += float(d.sample_weight[live].sum())
                weight["hist"] += float(d.sample_weight[~live].sum())
                is_valid = live & (d.timestamp >= cut).to_numpy()
                for name, mask, out in (("fit", ~is_valid, f_fit), ("valid", is_valid, f_val)):
                    if not mask.any():
                        continue
                    g = d[mask]
                    x = prepare(g, schema)
                    x.insert(0, schema.target, g[schema.target].astype(int))
                    x["sample_weight"] = g.sample_weight
                    x.to_csv(out, sep="\t", header=False, index=False, na_rep="nan", lineterminator="\n")
                    st = stats[name]
                    st["filas"] += len(g)
                    st["compras"] += int(g[schema.target].sum())
                    st["desde"] = min(filter(None, [st["desde"], str(g.timestamp.min())]))
                    st["hasta"] = max(filter(None, [st["hasta"], str(g.timestamp.max())]))
                    if name == "valid":
                        valid_parts.append(g)
            print(f"{part}: {stats}", flush=True)
            gc.collect()
    valid_df = pd.concat(valid_parts, ignore_index=True)
    del valid_parts
    live_share = weight["live"] / (weight["live"] + weight["hist"])
    print(f"peso del vivo en el entrenamiento: {100 * live_share:.1f} % (×{args.live_weight:g})", flush=True)

    params = {**DEFAULT_PARAMS, "depth": 6, "learning_rate": 0.1, "iterations": 800, "thread_count": args.threads}
    pool = lambda name: Pool(str(work / f"{name}.tsv"), column_description=str(work / "pool.cd"),   # noqa: E731
                             thread_count=args.threads)
    if args.trees:
        trees, report = args.trees, {}
    else:
        probe = CatBoostClassifier(**params)
        probe.fit(pool("fit"), eval_set=pool("valid"), early_stopping_rounds=50, use_best_model=True, verbose=50)
        trees = max(int(probe.tree_count_ * 1.05), 50)
        report = metrics.report(valid_df, BirdModel(probe, schema).predict(valid_df), schema, probabilities=True)
        print(f"parada temprana (validación: vivo desde {cut}): {probe.tree_count_} árboles → final {trees}", flush=True)
        del probe
        gc.collect()

    with (work / "all.tsv").open("wb") as out:                          # modelo final: todo
        for name in ("fit", "valid"):
            with (work / f"{name}.tsv").open("rb") as src:
                while chunk := src.read(1 << 24):
                    out.write(chunk)
    final = CatBoostClassifier(**{**params, "iterations": trees})
    final.fit(pool("all"), verbose=50)
    bird = BirdModel(final, schema)
    total = {"filas": stats["fit"]["filas"] + stats["valid"]["filas"], "compras": stats["fit"]["compras"] + stats["valid"]["compras"],
             "desde": stats["fit"]["desde"], "hasta": max(stats["fit"]["hasta"], stats["valid"]["hasta"])}
    bird.info = {"objective": "binary", "imbalance": "none", "version": schema.version, "params": params,
                 "best_iteration": trees, "trees": int(final.tree_count_), "train": total, "validation": stats["valid"],
                 "metrics": report, "importance": bird.importance().round(2).to_dict(), "all_data": True,
                 "cold_share": args.cold_share, "live_weight": args.live_weight, "ips_clip": 0.0,
                 "live_until": b5.live_until(), "live_weight_share": round(live_share, 4)}
    folder = bird.save(DATA_DIR / "bird" / args.name)
    print(json.dumps({k: bird.info[k] for k in ("trees", "train", "validation", "live_until", "live_weight_share")},
                     indent=2, default=str))
    print(f"{bird.info['trees']} árboles → {folder}")
    print("── importancia de las features (%)")
    print(bird.importance().round(2).head(args.top).to_string())
    for name in ("fit", "valid", "all"):
        (work / f"{name}.tsv").unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lv = sub.add_parser("live")
    lv.add_argument("--batch", type=int, default=3000)
    lv.add_argument("--workers", type=int, default=2)
    lv.add_argument("--keep-cache", action="store_true", help="no volver a descargar (reanudar una construcción a medias)")
    lv.set_defaults(fn=cmd_live)
    t = sub.add_parser("train")
    t.add_argument("--name", required=True, help="carpeta del modelo en data/bird/")
    t.add_argument("--cold-share", type=float, default=0.3)
    t.add_argument("--live-weight", type=float, default=3.0)
    t.add_argument("--valid-share", type=float, default=0.15)
    t.add_argument("--trees", type=int, help="nº fijo de árboles (sin la pasada de prueba)")
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--top", type=int, default=30)
    t.set_defaults(fn=cmd_train)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
