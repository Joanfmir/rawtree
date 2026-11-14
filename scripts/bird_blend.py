"""Busca offline la mejor forma de mezclar Bird con personal_v5 (catálogo completo, corte de reco.evaluate).

    python scripts/bird_blend.py                  # usa data/bird/v2_fresh y, si existe, data/bird/v2_rank
    python scripts/bird_blend.py --refresh        # vuelve a calcular las listas de cada política

Cada política calcula UNA vez su lista larga (200 productos) para cada carrusel de test, con el usuario visible
y con el usuario oculto (cold start). Las mezclas se prueban después sobre esas listas, sin volver a puntuar.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402

from reco import evaluate as ev, features  # noqa: E402
from reco.bird.blend import fuse  # noqa: E402
from reco.config import DATA_DIR  # noqa: E402
from reco.policies.bird import BirdV2  # noqa: E402
from reco.policies.personal import PersonalV5  # noqa: E402
from reco.safety import finalize  # noqa: E402

K = 200
COLS = ["ips_ndcg", "ips_hit", "ndcg@10", "hit@10"]


def lists(policy, df, state, known, hide_user: bool) -> list[list[str]]:
    out = []
    for row in df.itertuples(index=False):
        req = {"request_id": row.request_id, "user_id": None if hide_user else row.user_id,
               "is_new_user": hide_user or row.user_id not in known, "context": row.context,
               "recent_views": row.recent_views, "ts": row.ts}
        out.append(policy.recommend(req, state))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", default=ev.DEFAULT_UNTIL)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    feats = features.load(until=args.until)
    df = ev.drop_rebuys(ev.load_sessions(args.until), feats["user_purchases"])
    state = ev.make_state(args.until, feats)
    known = set(feats["user_subcat"].user_id)
    print(f"Test desde {args.until}: {len(df)} carruseles, {(df.purchased != '').sum()} con compra")

    cache = DATA_DIR / "bird" / f"blend_lists_{args.until}.pkl"
    got: dict[str, dict[bool, list[list[str]]]] = {}
    if cache.exists() and not args.refresh:
        got = pickle.loads(cache.read_bytes())
    makers = {"personal_v5": lambda: PersonalV5(until=args.until, weights={"top": K})}
    for name in ("v2_fresh", "v2_rank"):
        if (DATA_DIR / "bird" / name / "model.cbm").exists():
            def make(name=name):
                p = BirdV2(until=args.until)
                p.model_name, p.top = name, K
                return p
            makers[f"bird:{name}"] = make
    for name, make in makers.items():
        stamp = (DATA_DIR / "bird" / name.split(":")[1] / "model.cbm").stat().st_mtime if ":" in name else 0
        if name in got and got[name].get("stamp") == stamp:
            continue
        t = time.time()
        policy = make()
        policy.fit(state)
        got[name] = {False: lists(policy, df, state, known, False), True: lists(policy, df, state, known, True),
                     "stamp": stamp}
        cache.write_bytes(pickle.dumps(got))
        print(f"  listas de {name}: {time.time() - t:.0f}s")

    def score(make_list) -> dict[str, float]:
        out = {}
        for hide, prefix in ((False, ""), (True, "nuevo_")):
            recs = [finalize(make_list(i, hide), state) for i in range(len(df))]
            m = ev._metrics(recs, df)
            out.update({f"{prefix}{c}": m[c] for c in COLS})
        return out

    rows = {}
    per = got["personal_v5"]
    rows["personal_v5"] = score(lambda i, h: per[h][i])
    for name in [n for n in got if n.startswith("bird:")]:
        bird = got[name]
        tag = name.split(":")[1]
        rows[tag] = score(lambda i, h: bird[h][i])
        for n in (15, 20, 30, 50, 100):
            rows[f"{tag} rerank n={n}"] = score(lambda i, h: fuse(bird[h][i], per[h][i], "rerank", n=n))
        for alpha in (0.3, 0.5, 0.7):
            for k in (3, 10, 30):
                rows[f"{tag} rrf a={alpha} k={k}"] = score(
                    lambda i, h: fuse(bird[h][i], per[h][i], "rrf", alpha=alpha, k=k))
    table = pd.DataFrame(rows).T
    table["media"] = 0.7 * table["ips_ndcg"] + 0.3 * table["nuevo_ips_ndcg"]   # ~30 % de usuarios nuevos en vivo
    print(table.sort_values("media", ascending=False).to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
