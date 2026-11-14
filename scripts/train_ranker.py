"""Entrena el ranker LightGBM con las impresiones de shop.history_sessions.

    python scripts/train_ranker.py --until 2026-11-07   # para evaluar offline: solo sesiones < 11-07
    python scripts/train_ranker.py                      # producción: usa también la última semana

Cada semana de sesiones se describe con features calculadas SOLO con eventos anteriores a esa semana
(features_until_<inicio>), así el modelo aprende con la misma "frescura" que tendrá en vivo.
Ejemplo de entrenamiento = uno de los 10 productos enseñados en un carrusel con clic.
Etiqueta (lambdarank, grupo = carrusel): compra 2, clic sin compra 1, nada 0.
Sin recompra: se descartan los productos que el usuario ya había comprado antes.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from reco import features  # noqa: E402
from reco.client import query  # noqa: E402
from reco.config import DATA_DIR  # noqa: E402
from reco.ranker import FEATURES, Featurizer  # noqa: E402

PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[10], learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, verbose=-1, num_threads=4)
ROUNDS = 400
COLD_SHARE = 0.3   # en vivo ~30 % de usuarios nuevos: escondemos al usuario en ese % de carruseles


def clicked_sessions() -> pd.DataFrame:
    cache = DATA_DIR / "clicked_sessions_all.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    rows = query("""SELECT request_id::String request_id, ts::DateTime64(3) ts, user_id::String user_id, context,
                    recent_views, shown, propensity, clicked::String clicked, purchased::String purchased,
                    revenue_eur::Float64 revenue_eur
                    FROM shop.history_sessions WHERE clicked::String != ''""", timeout=900)
    df = pd.DataFrame(rows)
    df["recent_views"] = df.recent_views.apply(lambda v: v or [])
    cache.write_bytes(pickle.dumps(df))
    return df


def windows(until: str | None) -> list[tuple[str, str]]:
    """(inicio, fin) semanales; features hasta `inicio` describen las sesiones [inicio, fin)."""
    starts = ["2026-09-26", "2026-10-03", "2026-10-10", "2026-10-17", "2026-10-24", "2026-10-31", "2026-11-07"]
    ends = starts[1:] + ["2026-11-15"]
    return [(a, b) for a, b in zip(starts, ends) if until is None or b <= until]


def noclick_sessions() -> pd.DataFrame:
    """Muestra del 4 % de los carruseles SIN clic (enseñan la probabilidad absoluta de compra)."""
    cache = DATA_DIR / "noclick_sessions_sample.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    rows = query("""SELECT request_id::String request_id, ts::DateTime64(3) ts, user_id::String user_id, context,
                    recent_views, shown, '' AS clicked, '' AS purchased
                    FROM shop.history_sessions
                    WHERE clicked::String = '' AND ts >= '2026-09-26'
                      AND cityHash64(request_id::String) % 25 = 0""", timeout=900)
    df = pd.DataFrame(rows)
    df["recent_views"] = df.recent_views.apply(lambda v: v or [])
    cache.write_bytes(pickle.dumps(df))
    return df


def build_rows(fz: Featurizer, sess: pd.DataFrame, rng: np.random.Generator, cold_share: float,
               keep_empty: bool = False):
    X, y, groups = [], [], []
    for r in sess.itertuples(index=False):
        user = None if rng.random() < cold_share else r.user_id
        bought = fz.bought.get(r.user_id, set())
        keep = [(k, fz.idx[p]) for k, p in enumerate(r.shown) if p in fz.idx and fz.idx[p] not in bought]
        if len(keep) < 2:
            continue
        pos = np.array([k for k, _ in keep])
        cand = np.array([i for _, i in keep])
        lab = np.array([2 if fz.ids[i] == r.purchased else 1 if fz.ids[i] == r.clicked else 0 for i in cand])
        if lab.max() == 0 and not keep_empty:
            continue
        X.append(fz.matrix({"recent_views": r.recent_views, "context": r.context}, user, cand, pos))
        y.append(lab)
        groups.append(len(cand))
    return np.vstack(X), np.concatenate(y), groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", help="entrenar solo con sesiones anteriores (evaluación offline)")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--weeks", type=int, default=6, help="nº de semanas (las más recientes) para entrenar")
    ap.add_argument("--objective", choices=["rank", "binary"], default="rank",
                    help="rank: lambdarank por carrusel · binary: P(compra) por impresión")
    ap.add_argument("--noclick", action="store_true", help="añadir carruseles sin clic (solo con binary)")
    ap.add_argument("--tag", default="", help="sufijo del fichero del modelo (para experimentos)")
    args = ap.parse_args()
    binary = args.objective == "binary"

    df = clicked_sessions()
    if binary and args.noclick:
        df = pd.concat([df, noclick_sessions()], ignore_index=True)
    df["ts"] = pd.to_datetime(df.ts)
    rng = np.random.default_rng(0)
    Xs, ys, gs = [], [], []
    for start, end in windows(args.until)[-args.weeks:]:
        t = time.time()
        if not features.folder(start).exists():
            features.build(until=start, verbose=False)
        fz = Featurizer(features.load(until=start))
        sess = df[(df.ts >= start) & (df.ts < end)]
        X, y, g = build_rows(fz, sess, rng, COLD_SHARE, keep_empty=binary)
        Xs.append(X)
        ys.append(y)
        gs += g
        print(f"  semana {start}: {len(g):,} carruseles, {int((y == 2).sum()):,} compras  {time.time() - t:.0f}s")
    X, y = np.vstack(Xs), np.concatenate(ys)
    if binary:
        params = {**PARAMS, "objective": "binary", "metric": "auc"}
        params.pop("eval_at")
        ds = lgb.Dataset(X, (y == 2).astype(int), feature_name=FEATURES, categorical_feature=["page", "device"])
    else:
        params = PARAMS
        ds = lgb.Dataset(X, y, group=gs, feature_name=FEATURES, categorical_feature=["page", "device"])
    model = lgb.train(params, ds, num_boost_round=args.rounds)
    out = DATA_DIR / ((f"ranker_until_{args.until}" if args.until else "ranker_live") + f"{args.tag}.txt")
    model.save_model(str(out))
    imp = pd.Series(model.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    print(f"Modelo → {out}\nImportancia (gain, %):")
    print((100 * imp / imp.sum()).round(1).head(20).to_string())


if __name__ == "__main__":
    main()
