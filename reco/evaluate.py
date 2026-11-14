"""Evaluación offline de políticas con los carruseles reales de shop.history_sessions.

    python -m reco.evaluate                                  # baseline, coview y personal
    python -m reco.evaluate --policies personal --until 2026-11-07

Cómo funciona:
  - Las features se calculan SOLO con historia anterior a --until (sin ver el futuro).
  - Test: todos los carruseles desde --until en los que hubo clic o compra.
  - Cada carrusel se pasa a la política como si fuera una petición en vivo, y la lista pasa por
    safety.finalize(), exactamente como en producción.

Métricas (sobre los carruseles con compra; "click_" sobre los que tienen clic):
  hit@10    la compra real está en nuestro top 10
  ndcg@10   igual pero premia ponerla arriba (posición 1 = 1, posición 10 = 0,29)
  ips_hit   hit@10 ponderado por 1/propensity: corrige que el sistema viejo enseñaba más unos
            productos que otros (sesgo de popularidad del log)

Límite: en el log solo se pudo comprar lo que el sistema viejo enseñó. Es una medida relativa para
comparar políticas, no una predicción exacta de la conversión en vivo.
"""
from __future__ import annotations

import argparse
import pickle
import time

import numpy as np
import pandas as pd

from . import catalog, features
from .client import query
from .config import DATA_DIR
from .policies import POLICIES
from .safety import finalize
from .state import State

DEFAULT_UNTIL = "2026-11-07"


def load_sessions(until: str = DEFAULT_UNTIL) -> pd.DataFrame:
    """Carruseles de test (con clic o compra) desde `until`. Se guardan en caché en data/."""
    cache = DATA_DIR / f"eval_sessions_{until}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    rows = query(f"""
        SELECT request_id, ts, user_id, context, recent_views, shown, propensity, clicked, purchased, revenue_eur
        FROM shop.history_sessions
        WHERE ts >= '{until}' AND clicked::String != ''""", timeout=600)
    df = pd.DataFrame(rows)
    for col in ("clicked", "purchased"):
        df[col] = df[col].fillna("").astype(str)
    df["recent_views"] = df["recent_views"].apply(lambda v: v or [])
    DATA_DIR.mkdir(exist_ok=True)
    cache.write_bytes(pickle.dumps(df))
    return df


def drop_rebuys(df: pd.DataFrame, user_purchases: pd.DataFrame) -> pd.DataFrame:
    """Quita los carruseles cuya compra (o clic) es un producto que el usuario ya había comprado.
    El simulador en vivo penaliza la recompra, así que no queremos premiar a quien la explota."""
    bought = set(zip(user_purchases.user_id, user_purchases.product_id))
    target = np.where(df.purchased != "", df.purchased, df.clicked)
    return df[[(u, p) not in bought for u, p in zip(df.user_id, target)]].reset_index(drop=True)


def make_state(until: str, feats: dict[str, pd.DataFrame]) -> State:
    """Estado como el de producción, pero con el catálogo y best sellers de antes del corte."""
    items = feats["item_features"]
    cat = {r["product_id"]: {"is_active": True} for r in items[["product_id"]].to_dict("records")}
    return State(catalog=cat, unavailable=set(), best=catalog.best_sellers(until=until))


def _metrics(recs: list[list[str]], df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for target, prefix in (("purchased", ""), ("clicked", "click_")):
        mask = (df[target] != "").to_numpy()
        if not mask.any():
            continue
        tgt = df[target].to_numpy()[mask]
        rr = [r for r, m in zip(recs, mask) if m]
        pos = np.array([r.index(t) if t in r else -1 for r, t in zip(rr, tgt)])
        hit = pos >= 0
        out[f"{prefix}n"] = int(mask.sum())
        out[f"{prefix}hit@10"] = hit.mean()
        out[f"{prefix}ndcg@10"] = np.where(hit, 1 / np.log2(np.maximum(pos, 0) + 2), 0).mean()
        if not prefix:
            # propensity del producto comprado en el slot en el que lo enseñó el sistema viejo
            props = np.array([p[s.index(t)] for p, s, t in
                              zip(df.propensity.to_numpy()[mask], df.shown.to_numpy()[mask], tgt)])
            w = 1 / np.clip(props, 1e-3, None)
            out["ips_hit"] = (w * hit).sum() / w.sum()
            out["ips_ndcg"] = (w * np.where(hit, 1 / np.log2(np.maximum(pos, 0) + 2), 0)).sum() / w.sum()
    return out


def evaluate(policy, df: pd.DataFrame, state: State, known_users: set[str], anon: bool = True) -> pd.DataFrame:
    """Métricas globales y por segmento para una política ya entrenada (fit)."""
    def run(hide_user: bool) -> list[list[str]]:
        recs = []
        for row in df.itertuples(index=False):
            req = {"request_id": row.request_id, "user_id": None if hide_user else row.user_id,
                   "is_new_user": hide_user or row.user_id not in known_users,
                   "context": row.context, "recent_views": row.recent_views, "ts": row.ts}
            recs.append(finalize(policy.recommend(req, state), state))
        return recs

    t = time.time()
    recs = run(hide_user=False)
    ms = (time.time() - t) / len(df) * 1000
    # cold start: mismas peticiones sin saber quién es el usuario (en el test no hay usuarios nuevos)
    recs_anon = run(hide_user=True) if anon else None
    known = df.user_id.isin(known_users).to_numpy()
    has_views = (df.recent_views.str.len() > 0).to_numpy()
    segments = {"todos": np.ones(len(df), bool), "conocidos": known,
                "con vistas": has_views, "sin vistas": ~has_views}
    rows = {}
    for name, m in segments.items():
        rows[name] = _metrics([r for r, k in zip(recs, m) if k], df[m])
    if anon:
        rows["simulados nuevos"] = _metrics(recs_anon, df)
    out = pd.DataFrame(rows).T
    out["ms/req"] = ms
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="*", default=["baseline", "coview", "personal"])
    ap.add_argument("--until", default=DEFAULT_UNTIL)
    ap.add_argument("--with-rebuy", action="store_true",
                    help="incluir las compras que son recompras (por defecto se excluyen del test)")
    args = ap.parse_args()

    if not features.folder(args.until).exists():
        features.build(until=args.until)
    feats = features.load(until=args.until)
    df = load_sessions(args.until)
    if not args.with_rebuy:
        df = drop_rebuys(df, feats["user_purchases"])
    state = make_state(args.until, feats)
    known = set(feats["user_subcat"].user_id)
    print(f"Test desde {args.until}: {len(df)} carruseles con clic, {(df.purchased != '').sum()} con compra\n")

    summary = {}
    for name in args.policies:
        policy = POLICIES[name](until=args.until)
        policy.fit(state)
        res = evaluate(policy, df, state, known)
        summary[name] = res.loc["todos"]
        print(f"── {name}")
        print(res[["n", "hit@10", "ndcg@10", "ips_hit", "ips_ndcg", "click_hit@10", "ms/req"]]
              .to_string(float_format=lambda x: f"{x:.4f}"), "\n")
    print("── Resumen (todos)")
    print(pd.DataFrame(summary).T[["hit@10", "ndcg@10", "ips_hit", "ips_ndcg", "click_hit@10"]]
          .to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
