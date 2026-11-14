"""Ajuste de pesos de la política personal (offline).

    python scripts/tune.py ablation     # apaga cada señal y mide
    python scripts/tune.py search       # búsqueda coordenada (un peso cada vez, 2 vueltas) sobre ips_ndcg
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402

from reco import evaluate as ev, features  # noqa: E402
from reco.policies.personal import DEFAULT_WEIGHTS, Personal  # noqa: E402

UNTIL = sys.argv[2] if len(sys.argv) > 2 else ev.DEFAULT_UNTIL
feats = features.load(until=UNTIL)
df = ev.load_sessions(UNTIL)
state = ev.make_state(UNTIL, feats)
known = set(feats["user_subcat"].user_id)
policy = Personal(until=UNTIL)
policy.fit(state)
COLS = ["ips_ndcg", "ips_hit", "ndcg@10", "hit@10"]


def score(weights, anon=False):
    policy.w = {**DEFAULT_WEIGHTS, **weights}
    r = ev.evaluate(policy, df, state, known, anon=anon)
    out = r.loc["todos", COLS].to_dict()
    if anon:
        out["anon_ips_ndcg"] = r.loc["simulados nuevos", "ips_ndcg"]
    return out


mode = sys.argv[1] if len(sys.argv) > 1 else "ablation"
if mode == "ablation":
    runs = {"default": {}, **{f"sin {k}": {k: 0} for k in ("self", "buy", "view", "user", "ctx", "pop", "brand", "rebuy")}}
    print(pd.DataFrame({n: score(w, anon=True) for n, w in runs.items()}).T.to_string(float_format=lambda x: f"{x:.4f}"))
else:
    grid = {"self": [0, 0.15, 0.3, 0.6, 1], "buy": [0.5, 1, 2], "view": [0.1, 0.3, 0.6], "user": [0.25, 0.5, 1],
            "ctx": [0.15, 0.3, 0.6], "pop": [0.1, 0.2, 0.4], "brand": [0.25, 0.5, 1], "rebuy": [1, 2, 4, 8],
            "rebuy_days": [7, 15, 30, 60], "decay": [0.3, 0.6, 0.9]}
    best = {}
    best_score = score(best)["ips_ndcg"]
    print(f"inicio ips_ndcg={best_score:.4f}")
    for rnd in range(2):
        for k, values in grid.items():
            results = {v: score({**best, k: v})["ips_ndcg"] for v in values}
            v = max(results, key=results.get)
            if results[v] > best_score + 1e-4:
                best[k], best_score = v, results[v]
            print(f"vuelta {rnd + 1} · {k:10} {' '.join(f'{x}:{y:.4f}' for x, y in results.items())}  → {best.get(k, DEFAULT_WEIGHTS[k])}")
    print("\nMejores pesos:", {**DEFAULT_WEIGHTS, **best})
    print(pd.DataFrame({"default": score({}, anon=True), "ajustado": score(best, anon=True)}).T
          .to_string(float_format=lambda x: f"{x:.4f}"))
