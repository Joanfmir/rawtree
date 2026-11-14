"""Reordenador orgánico de la política joan: dataset y entrenamiento.

    python scripts/joan_ranker.py dataset     # peticiones 31/10–07/11 + compras orgánicas en los 30 min siguientes
    python scripts/joan_ranker.py train       # CatBoost YetiRank → data/joan/organic_ranker/

Datos (todo anterior a los dos tests offline, que empiezan el 07/11 y el 14/11 a las 12:00):
  - peticiones: carruseles de shop.history_sessions del 31/10 al 07/11 (contexto, vistas de la sesión, usuario)
  - objetivo: lo que el usuario compró POR SU CUENTA (history_events) en los 30 min siguientes y no tenía ya
    (mismo objetivo que el test orgánico, el que mejor ordena el leaderboard: Spearman 0,98)
  - candidatos y features: estado de Bird con corte 31/10 (sin ver el futuro) y bird_v2 (v2_fresh, entrenado
    con datos hasta el 30/10): sus K mejores productos, con las 56 features + puntuación, puesto y distancia al 1.º
  - solo grupos con la compra entre los K candidatos (lo único que el reordenador puede arreglar)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from reco import evaluate as ev  # noqa: E402
from reco import features  # noqa: E402
from reco.client import query  # noqa: E402
from reco.config import DATA_DIR  # noqa: E402
from reco.offline_live import organic_labels  # noqa: E402
from reco.policies.bird import BirdV2  # noqa: E402
from reco.policies.joan import EXTRA, candidates  # noqa: E402

WINDOW, K = 30, 100
OUT = DATA_DIR / "joan"
# Fuentes de entrenamiento: (corte del estado, inicio, fin de las peticiones). Todo anterior a la parte del
# test orgánico con la que se evalúa el reordenador (peticiones desde las 13:00 del 14/11).
SOURCES = {
    "history": ("2026-10-31", "2026-10-31", "2026-11-07"),           # carruseles del histórico
    "today": ("2026-11-14 09:00:00", "2026-11-14 09:00:00", "2026-11-14 12:30:00"),   # peticiones reales
}


def _source_rows(name: str, feats) -> pd.DataFrame:
    cut, start, end = SOURCES[name]
    if name == "history":
        req_sql = f"""SELECT request_id::String AS request_id, ts::DateTime64(3) AS ts, user_id::String AS user_id,
                             context, recent_views FROM shop.history_sessions"""
        ev_tab, car = "shop.history_events", f"""SELECT user_id::String, purchased::String, ts::DateTime64(3), 'carousel'
            FROM shop.history_sessions WHERE purchased::String != '' AND ts >= '{start}' AND ts < '{end}'"""
    else:
        from reco.config import TEAM
        req_sql = f"""SELECT request_id::String AS request_id, ts::DateTime64(3) AS ts, user_id::String AS user_id,
                             context, recent_views FROM shop.requests"""
        ev_tab, car = "shop.events", f"""SELECT r.user_id::String, f.purchased::String, f.ts::DateTime64(3), 'carousel'
            FROM {TEAM}.feedback AS f INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
            WHERE f.purchased::String != '' AND f.ts::DateTime64(3) >= '{start}' AND f.ts::DateTime64(3) < '{end}'"""
    reqs = pd.DataFrame(query(f"""{req_sql}
        WHERE ts::DateTime64(3) >= '{start}' AND ts::DateTime64(3) < '{end}'
          AND user_id::String IN (SELECT user_id::String FROM {ev_tab}
                                  WHERE event_type::String = 'purchase' AND ts::DateTime64(3) >= '{start}'
                                    AND ts::DateTime64(3) < toDateTime64('{end}', 3) + INTERVAL {WINDOW} MINUTE)""",
                              timeout=900))
    buys = pd.DataFrame(query(f"""
        SELECT user_id::String AS user_id, product_id::String AS product_id, ts::DateTime64(3) AS ts, 'organic' AS source
        FROM {ev_tab} WHERE event_type::String = 'purchase'
          AND ts::DateTime64(3) >= '{start}' AND ts::DateTime64(3) < toDateTime64('{end}', 3) + INTERVAL {WINDOW} MINUTE
        UNION ALL {car}""", timeout=900))
    df = organic_labels(reqs, buys, feats["user_purchases"], WINDOW)
    print(f"[{name}] peticiones con compra orgánica nueva en {WINDOW} min: {len(df)} de {len(reqs)}")
    return df


def cmd_dataset(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.sources:
        cut = SOURCES[name][0]
        feats = features.load(until=cut) if features.folder(cut).exists() else features.build(until=cut)
        state = ev.make_state(cut, feats)
        df = _source_rows(name, feats)
        if args.limit:
            df = df.sample(min(args.limit, len(df)), random_state=0).sort_values("ts").reset_index(drop=True)
        bird = BirdV2(until=cut)
        bird.fit(state)
        schema = bird.model.schema
        nums, cats, labels, groups, tss = [], [], [], [], []
        kept = known = 0
        t = time.time()
        for g, row in enumerate(df.itertuples(index=False)):
            req = {"request_id": row.request_id, "user_id": row.user_id, "is_new_user": False,
                   "context": row.context, "recent_views": row.recent_views, "ts": row.ts}
            if bird.live.user(req) is None:      # el reordenador es para conocidos
                continue
            known += 1
            idx, num, cat = candidates(bird, req, state, K)
            pos = {bird.live.idx[p] for p in row.labels if p in bird.live.idx}
            y = np.array([i in pos for i in idx], dtype=np.int8)
            if not y.any():
                continue
            kept += 1
            nums.append(num), cats.append(cat), labels.append(y)
            groups.append(np.full(len(idx), f"{name}:{g}")), tss.append(np.full(len(idx), str(row.ts), dtype=object))
        print(f"[{name}] conocidos: {known} · con la compra en el top {K} de bird_v2: {kept} "
              f"({kept / max(known, 1):.1%}) · {time.time() - t:.0f}s")
        names = list(schema.numeric) + list(EXTRA)
        out = pd.DataFrame(np.vstack(nums), columns=names)
        for j, c in enumerate(schema.categorical):
            out[c] = [v.decode("utf-8") if isinstance(v, bytes) else str(v)   # CatBoost codifica en UTF-8
                      for v in np.concatenate([x[:, j] for x in cats])]
        out["label"], out["group"], out["ts"] = np.concatenate(labels), np.concatenate(groups), np.concatenate(tss)
        out["source"] = name
        out.to_parquet(OUT / f"organic_{name}.parquet", index=False)
        (OUT / "organic_dataset.json").write_text(json.dumps({"num": names, "cat": list(schema.categorical), "k": K,
                                                              "sources": SOURCES, "window": WINDOW}))
        print(f"→ {OUT / f'organic_{name}.parquet'}  ({len(out)} filas, {out.group.nunique()} grupos)")


def cmd_train(args) -> None:
    from catboost import CatBoost, Pool
    meta = json.loads((OUT / "organic_dataset.json").read_text())
    df = pd.concat([pd.read_parquet(OUT / f"organic_{n}.parquet") for n in args.sources], ignore_index=True)
    order = df.groupby("group").ts.first().sort_values()
    valid_groups = set(order.index[int(len(order) * (1 - args.valid_share)):])   # los más recientes
    va = df.group.isin(valid_groups).to_numpy()
    cols = meta["num"] + meta["cat"]
    if args.drop:
        cols = [c for c in cols if c not in args.drop]

    def pool(m):
        d = df[m]
        return Pool(d[cols], label=d.label.to_numpy(), group_id=d.group.to_numpy(),
                    cat_features=[c for c in meta["cat"] if c in cols])
    params = dict(loss_function=args.loss, eval_metric="NDCG:top=10", iterations=args.iterations,
                  learning_rate=args.lr, depth=args.depth, random_seed=0, thread_count=-1, verbose=100)
    model = CatBoost(params)
    t = time.time()
    model.fit(pool(~va), eval_set=pool(va), early_stopping_rounds=100, use_best_model=True)
    best = model.get_best_iteration()
    print(f"entrenado en {time.time() - t:.0f}s · mejor iteración {best} · NDCG@10 valid {model.get_best_score()}")
    # NDCG@10 de bird_v2 sin reordenar en la misma validación (orden = bird_rank)
    d = df[va]
    base = d.groupby("group").apply(lambda g: 1 / np.log2(g.bird_rank[g.label == 1].min() + 2)
                                    if (g.bird_rank[g.label == 1] < 10).any() else 0.0, include_groups=False).mean()
    print(f"NDCG@10 de bird_v2 en esa validación (solo grupos con la compra en el top {meta['k']}): {base:.4f}")
    if args.final:   # reentrena con todo, mismas iteraciones
        params.update(iterations=best + 1)
        model = CatBoost(params)
        model.fit(pool(np.ones(len(df), bool)))
    folder = OUT / args.name
    folder.mkdir(parents=True, exist_ok=True)
    model.save_model(str(folder / "model.cbm"))
    num = [c for c in meta["num"] if c in cols]
    cat = [c for c in meta["cat"] if c in cols]
    (folder / "meta.json").write_text(json.dumps({**meta, "num": num, "cat": cat, "best_iteration": best,
                                                  "params": params, "final": bool(args.final)}, indent=1))
    imp = pd.Series(model.get_feature_importance(type="PredictionValuesChange"), index=cols).sort_values(ascending=False)
    print("importancia:\n" + imp.head(15).round(1).to_string())
    print(f"→ {folder}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("--limit", type=int, help="máximo de peticiones (muestra aleatoria)")
    d.add_argument("--sources", nargs="+", choices=sorted(SOURCES), default=sorted(SOURCES))
    t = sub.add_parser("train")
    t.add_argument("--name", default="organic_ranker")
    t.add_argument("--sources", nargs="+", choices=sorted(SOURCES), default=sorted(SOURCES))
    t.add_argument("--loss", default="YetiRank")
    t.add_argument("--iterations", type=int, default=2000)
    t.add_argument("--lr", type=float, default=0.05)
    t.add_argument("--depth", type=int, default=6)
    t.add_argument("--valid-share", type=float, default=0.15)
    t.add_argument("--drop", nargs="*", default=[])
    t.add_argument("--final", action="store_true", help="reentrenar con todos los datos tras elegir iteraciones")
    args = ap.parse_args()
    {"dataset": cmd_dataset, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
