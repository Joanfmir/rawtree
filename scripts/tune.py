"""Ablación y ajuste de pesos de la política personal (offline).   python scripts/tune.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402

from reco import evaluate as ev, features  # noqa: E402
from reco.policies.personal import DEFAULT_WEIGHTS, Personal  # noqa: E402

UNTIL = ev.DEFAULT_UNTIL
feats = features.load(until=UNTIL)
df = ev.load_sessions(UNTIL)
state = ev.make_state(UNTIL, feats)
known = set(feats["user_subcat"].user_id)
policy = Personal(until=UNTIL)
policy.fit(state)


def score(**changes):
    policy.w = {**DEFAULT_WEIGHTS, **changes}
    r = ev.evaluate(policy, df, state, known)
    return {"ips_hit": r.loc["todos", "ips_hit"], "ndcg@10": r.loc["todos", "ndcg@10"],
            "hit@10": r.loc["todos", "hit@10"], "anon_ips": r.loc["simulados nuevos", "ips_hit"]}


configs = {k: v for k, v in [a.split("=", 1) for a in sys.argv[1:]]} if len(sys.argv) > 1 else None
runs = {"default": {}, "sin user": {"user": 0}, "sin brand": {"brand": 0}, "sin rebuy": {"rebuy": 0},
        "sin self": {"self": 0}, "sin buy": {"buy": 0}, "sin view": {"view": 0}, "sin ctx": {"ctx": 0},
        "sin pop": {"pop": 0}}
res = {name: score(**ch) for name, ch in runs.items()}
print(pd.DataFrame(res).T.to_string(float_format=lambda x: f"{x:.4f}"))
