from __future__ import annotations

from collections import Counter

from .base import Policy


class Coview(Policy):
    """Productos vistos en las mismas sesiones que los recent_views (los más recientes pesan más)."""
    name = "coview"

    def fit(self, state):
        nb = self.load_features()["item_neighbors"]
        nb = nb[nb.kind == "view"].sort_values(["p", "n"], ascending=[True, False])
        self.neighbors = nb.groupby("p").q.apply(list).to_dict()

    def recommend(self, req, state):
        views = req.get("recent_views") or []
        seen = {v["product_id"] for v in views}
        score = Counter()
        for k, v in enumerate(reversed(views)):
            for rank, q in enumerate(self.neighbors.get(v["product_id"], [])[:20]):
                if q not in seen:
                    score[q] += 0.7 ** k / (rank + 1)
        return [q for q, _ in score.most_common(10)]
