"""Dataset v3: features de lista para el modelo ordenador (segunda etapa). Ver reco/bird/stage2.py.

    python scripts/bird_stage2.py folds               # selectores entrenados solo con fechas anteriores (sin fuga)
    python scripts/bird_stage2.py states              # fotos semanales del estado de usuarios
    python scripts/bird_stage2.py part --week 2026-10-10   # features de lista de las peticiones de esa semana
    python scripts/bird_stage2.py assemble            # → data/bird/v3/{train,validation,test}_v3.parquet

Para cada carrusel con clic desde el 10/10: el selector de su semana puntúa el catálogo tal como estaba al empezar
esa semana, elige 50 candidatos y se calculan las features de los 10 productos enseñados frente a esos 50.
  - Selector sin fuga: el de la semana S se entrenó con peticiones anteriores a S.
  - Cold start: en el 30 % de las peticiones de train (fijado por hash) se esconde al usuario en las dos etapas.
  - Las señales directas (grupo E), targets y metadatos vienen del dataset v2 (exactas a fecha de la petición).
Límite conocido: el estado de usuario del selector es de principios de semana (hasta 7 días de retraso).
"""
import argparse
import pickle
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from reco.bird import build_dataset as bd, dataset, live, stage2  # noqa: E402
from reco.bird.model import BirdModel, train  # noqa: E402
from reco.bird.schema import SCHEMAS  # noqa: E402
from reco.config import DATA_DIR  # noqa: E402

WEEKS = ["2026-10-10", "2026-10-17", "2026-10-24", "2026-10-31", "2026-11-07"]
FOLD_WEEKS = WEEKS[:3]          # desde el 31/10 el selector es v2_fresh (entrenado con < 31/10)
FOLDS = DATA_DIR / "bird" / "folds"
OUT = DATA_DIR / "bird" / "v3"
COLD_SHARE = 0.3
SELECTOR_SCHEMA = "v2_norebuy"


def selector_folder(week: str) -> Path:
    return FOLDS / f"sel_{week}" if week in FOLD_WEEKS else DATA_DIR / "bird" / "v2_fresh"


def is_cold(request_id: str) -> bool:
    return zlib.crc32(request_id.encode()) % 100 < COLD_SHARE * 100


def cmd_folds(args) -> None:
    schema = SCHEMAS[SELECTOR_SCHEMA]
    df = dataset.load(DATA_DIR / "bird" / "v2" / "train_v2.parquet")
    df = df[df.already_purchased == 0]
    for week in FOLD_WEEKS:
        t = time.time()
        part = df[df.timestamp < pd.Timestamp(week, tz="UTC")]
        tail = part[part.timestamp >= part.timestamp.max() - pd.Timedelta(days=2)]   # solo para informar
        bird = train(bd.simulate_cold(part, COLD_SHARE), tail, schema, early_stopping=None,
                     params={"iterations": 130, "learning_rate": 0.1, "thread_count": args.threads})
        bird.save(selector_folder(week))
        print(f"selector < {week}: {len(part):,} filas, {int(part.target_purchase.sum()):,} compras  {time.time() - t:.0f}s")


def cmd_states(args) -> None:
    for week in WEEKS:
        if (live.folder(week) / "user_prod.parquet").exists():
            continue
        t = time.time()
        live.download(week, verbose=False)
        print(f"estado hasta {week}: {time.time() - t:.0f}s")


def cmd_part(args) -> None:
    week = args.week
    end = (pd.Timestamp(week) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    sess = bd.load_sessions()
    sess = sess[(sess.ts >= week) & (sess.ts < end) & (sess.clicked != "")].reset_index(drop=True)
    state = live.LiveState.load(week)
    selector = BirdModel.load(selector_folder(week))
    train_split = week < bd.VALID_FROM
    out = {name: [] for name in stage2.LIST}
    rid, pid, cold_col = [], [], []
    t = time.time()
    for n, s in enumerate(sess.itertuples(index=False)):
        rows = np.array([state.idx.get(p, -1) for p in s.shown])
        if (rows < 0).any():
            continue
        cold = train_split and is_cold(s.request_id)
        req = {"request_id": s.request_id, "user_id": None if cold else s.user_id, "is_new_user": cold,
               "context": s.context, "recent_views": s.recent_views, "ts": s.ts}
        allowed = np.ones(state.n, dtype=bool)
        allowed[state.bought(state.user(req))] = False
        cols, p, chosen = stage2.select(state, selector, req, allowed)
        for name, v in stage2.list_features(state, cols, p, chosen, rows).items():
            out[name].append(v)
        rid += [s.request_id] * len(rows)
        pid += list(s.shown)
        cold_col += [int(cold)] * len(rows)
        if args.verbose and n % 2000 == 0:
            print(f"  {week}: {n:,}/{len(sess):,}  {time.time() - t:.0f}s", flush=True)
    df = pd.DataFrame({"request_id": rid, "product_id": pid, "cold_sim": cold_col,
                       **{k: np.concatenate(v).astype("float32") for k, v in out.items()}})
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / f"part_{week}.parquet", index=False)
    print(f"{week}: {len(sess):,} peticiones → {len(df):,} filas  {time.time() - t:.0f}s")


def cmd_assemble(args) -> None:
    schema = SCHEMAS["v3_rank"]
    lists = pd.concat([pd.read_parquet(OUT / f"part_{w}.parquet") for w in WEEKS], ignore_index=True)
    keep = ["request_id", "user_id", "product_id", "timestamp", "position", "target_purchase", "target_click",
            "propensity", "sample_weight", "revenue_eur", "already_purchased", *stage2.DIRECT]
    v2 = pd.concat([dataset.load(DATA_DIR / "bird" / "v2" / f"{s}_v2.parquet")[keep]
                    for s in ("train", "validation", "test")], ignore_index=True)
    df = v2.merge(lists, on=["request_id", "product_id"], how="inner")
    # cold start simulado: las señales directas de usuario también quedan como las de un usuario nuevo
    cold = df.cold_sim == 1
    df.loc[cold, "u_attr_share"] = 0.0
    df.loc[cold, "u_logp_diff"] = np.nan
    df.loc[cold, "is_new_user"] = 1
    df = df.sort_values(["timestamp", "request_id", "position"]).reset_index(drop=True)
    parts = bd.split(df)
    for name, part in parts.items():
        part.to_parquet(OUT / f"{name}_v3.parquet", index=False)
        print(f"{name:<11} {dataset.describe(part, schema)} · cold simulado {part.cold_sim.mean():.0%}")
    for a, b in (("train", "validation"), ("validation", "test")):
        for problem in dataset.validate(parts[a], parts[b], schema, strict=False):
            print(f"[contrato] {problem}")
    print(f"→ {OUT}  ({len(schema.features)} features)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("folds")
    f.add_argument("--threads", type=int, default=4)
    f.set_defaults(fn=cmd_folds)
    sub.add_parser("states").set_defaults(fn=cmd_states)
    p = sub.add_parser("part")
    p.add_argument("--week", required=True, choices=WEEKS)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=cmd_part)
    sub.add_parser("assemble").set_defaults(fn=cmd_assemble)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
