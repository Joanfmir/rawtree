from __future__ import annotations

from collections import Counter, defaultdict

from ..client import query
from ..config import DATA_DIR
from .base import Policy

COVIEWS = """
SELECT a.product_id AS p, b.product_id AS q, count() AS n
FROM (SELECT session_id::String AS session_id, product_id::String AS product_id FROM shop.history_events WHERE event_type = 'view') AS a
INNER JOIN (SELECT session_id::String AS session_id, product_id::String AS product_id FROM shop.history_events WHERE event_type = 'view') AS b
  ON a.session_id = b.session_id
WHERE a.product_id != b.product_id
GROUP BY p, q ORDER BY n DESC LIMIT 20 BY p"""


class Coview(Policy):
    """Productos vistos en las mismas sesiones que los recent_views (los más recientes pesan más)."""
    name = "coview"

    def __init__(self) -> None:
        self.neighbors: dict[str, list[str]] = {}

    def fit(self, state):
        cache = DATA_DIR / "coviews.csv"
        if cache.exists():
            import pandas as pd
            rows = pd.read_csv(cache).to_dict("records")
        else:
            rows = query(COVIEWS, timeout=600)
            DATA_DIR.mkdir(exist_ok=True)
            import pandas as pd
            pd.DataFrame(rows).to_csv(cache, index=False)
        nb = defaultdict(list)
        for r in rows:
            nb[str(r["p"])].append(str(r["q"]))
        self.neighbors = dict(nb)
        print(f"[coview] {len(self.neighbors)} productos con vecinos")

    def recommend(self, req, state):
        views = req.get("recent_views") or []
        seen = {v["product_id"] for v in views}
        score = Counter()
        for k, v in enumerate(reversed(views)):
            for rank, q in enumerate(self.neighbors.get(v["product_id"], [])):
                if q not in seen:
                    score[q] += 0.7 ** k / (rank + 1)
        return [q for q, _ in score.most_common(10)]
