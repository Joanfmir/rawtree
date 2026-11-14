"""Bird v2 serving: immutable snapshots, no SQL in the request path.

History uses the training event definitions, including carousel clicks/purchases.
Item aggregates stop at midnight, as in training. User history stops at snapshot
time (so it can be slightly stale, but never includes a future request's outcome).
"""
from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, features
from ..catalog import CURRENT_CATALOG
from ..client import query
from . import build_dataset as bd
from .schema import SESSION, V2_NOATTRS

SNAPSHOT = config.DATA_DIR / "bird" / "live.pkl"


def events(cut: str) -> str:
    organic = " UNION ALL ".join(
        f"SELECT user_id::String AS user_id, product_id::String AS product_id, "
        f"ts::DateTime64(3) AS ts, event_type::String AS kind FROM {table} "
        f"WHERE ts::DateTime64(3) < '{cut}'"
        for table in ("shop.history_events", "shop.events"))
    return f"""({organic} UNION ALL
        SELECT user_id::String, clicked::String, ts::DateTime64(3),
               if(purchased::String != '', 'c_purchase', 'c_click')
        FROM shop.history_sessions WHERE clicked::String != '' AND ts::DateTime64(3) < '{cut}'
        UNION ALL
        SELECT r.user_id::String, f.clicked::String, f.ts::DateTime64(3),
               if(f.purchased::String != '', 'c_purchase', 'c_click')
        FROM {config.TEAM}.feedback f INNER JOIN shop.requests r
        ON f.request_id::String = r.request_id::String
        WHERE f.clicked::String != '' AND f.ts::DateTime64(3) < '{cut}')"""


def build_snapshot() -> None:
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    cut = str(now)
    cat = pd.DataFrame(query(CURRENT_CATALOG, timeout=60))
    original = bd.load_catalog().set_index("product_id")
    attrs = pd.DataFrame(query(features.ITEM_ATTRS, timeout=60))
    order = {a: i for i, a in enumerate(features.ATTRS)}
    attr = attrs.groupby("product_id").attr.apply(lambda a: sorted(set(a), key=order.get)).to_dict()
    created = query("SELECT product_id::String AS product_id, toString(min(ts::DateTime64(3))) AS created "
                    "FROM shop.catalog_updates WHERE change::String = 'new_product' GROUP BY product_id", timeout=60)
    dates = {r["product_id"]: r["created"] for r in created}
    for i in (0, 1):
        cat[f"attr{i + 1}"] = [attr.get(p, [])[i] if len(attr.get(p, [])) > i else "UNKNOWN"
                                 for p in cat.product_id]
    cat["created_at"] = [original.loc[p, "created_at"] if p in original.index else dates.get(p, cut)
                         for p in cat.product_id]
    cat["logp"] = np.log1p(cat.price_eur.astype(float))
    cat = cat.sort_values("product_id").reset_index(drop=True)
    pid = {p: i for i, p in enumerate(cat.product_id)}

    # Same daily formulas as training, extending the calendar to today's midnight.
    daily = bd.load_item_daily().copy()
    live = pd.DataFrame(query(f"""SELECT product_id::String AS product_id,
        toString(toDate(ts::DateTime64(3))) AS day,
        countIf(event_type::String = 'view') AS views,
        countIf(event_type::String = 'add_to_cart') AS carts,
        countIf(event_type::String = 'purchase') AS buys
        FROM shop.events WHERE ts::DateTime64(3) < '{now.normalize()}' GROUP BY product_id, day""", timeout=120))
    feedback = pd.DataFrame(query(f"""SELECT p AS product_id,
        toString(toDate(ts::DateTime64(3))) AS day, count() AS c_impr,
        countIf(clicked::String = p) AS c_clicks, countIf(purchased::String = p) AS c_buys
        FROM {config.TEAM}.feedback ARRAY JOIN shown::Array(String) AS p
        WHERE request_id::String != '__init__' AND ts::DateTime64(3) < '{now.normalize()}'
        GROUP BY product_id, day""", timeout=120))
    daily = pd.concat([daily, live, feedback], ignore_index=True).fillna(0)
    bd.HISTORY_TO = str(now.date())
    tables, _ = bd.item_tables(daily, cat)
    item = cat.copy()
    for name, values in tables.items():
        item[name] = values[-1]
    item["it_price_vs_sub"] = cat.logp - cat.groupby("subcategory").logp.transform("median")
    item["it_age_days"] = (now - pd.to_datetime(cat.created_at)).dt.total_seconds() / 86400
    neighbors = bd._neighbors(bd.WEEKS[-1], pid)
    users = {}
    for k in range(32):
        rows = query(f"""SELECT user_id, product_id,
            countIf(kind = 'view') AS up_views, countIf(kind = 'add_to_cart') AS up_carts,
            countIf(kind = 'purchase') AS up_buys, countIf(kind = 'c_click') AS up_c_clicks,
            countIf(kind = 'c_purchase') AS up_c_buys, count() AS n,
            sum(multiIf(kind IN ('purchase', 'c_purchase'), 5, kind = 'add_to_cart', 3, 1)) AS w,
            sumIf(multiIf(kind IN ('purchase', 'c_purchase'), 5, kind = 'add_to_cart', 3, 1),
                  ts >= toDateTime64('{cut}', 3) - INTERVAL 7 DAY) AS w7,
            toString(max(ts)) AS last_seen,
            toString(maxIf(ts, kind IN ('purchase', 'c_purchase'))) AS last_buy
            FROM {events(cut)} WHERE cityHash64(user_id) % 32 = {k}
            GROUP BY user_id, product_id LIMIT 499999""", timeout=120)
        if len(rows) >= 499999:
            raise RuntimeError("Snapshot user history reached query row limit")
        for r in rows:
            p = pid.get(r["product_id"])
            if p is not None:
                users.setdefault(r["user_id"], []).append((p, *[r[c] for c in (
                    "up_views", "up_carts", "up_buys", "up_c_clicks", "up_c_buys", "n", "w", "w7")],
                    pd.Timestamp(r["last_seen"]).timestamp(), pd.Timestamp(r["last_buy"]).timestamp()))
        print(f"[bird snapshot] users {k + 1}/32", flush=True)
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    tmp = SNAPSHOT.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(pickle.dumps(dict(items=item, users=users, neighbors=neighbors, as_of=cut)))
    os.replace(tmp, SNAPSHOT)
    print(f"[bird snapshot] ready: {len(item)} items, {len(users)} users, as_of={cut}", flush=True)


class LiveFeatures:
    def __init__(self, snapshot: dict):
        self.__dict__.update(snapshot)
        self.pid = {p: i for i, p in enumerate(self.items.product_id)}
        self.item_ids = self.items.product_id.to_numpy()

    def build(self, req: dict) -> pd.DataFrame:
        df = self.items.copy()
        n = len(df)
        context = req.get("context") or {}
        for key in ("page", "device", "region"):
            df[key] = context.get(key) or "UNKNOWN"
        for col in V2_NOATTRS.per_request:
            if col not in df:
                df[col] = 0.0
        now = pd.Timestamp(req.get("ts") or pd.Timestamp.now(tz="UTC"))
        if now.tzinfo is not None:
            now = now.tz_convert("UTC").tz_localize(None)
        if now < pd.Timestamp(self.as_of):
            raise ValueError("request predates feature snapshot")
        ts = now.timestamp()
        df["it_age_days"] = (now - pd.to_datetime(df.created_at)).dt.total_seconds() / 86400
        df["position"] = V2_NOATTRS.reference_position
        views = [self.pid[v["product_id"]] for v in req.get("recent_views", [])
                 if isinstance(v, dict) and v.get("product_id") in self.pid]
        df["n_recent_views"] = len(views)
        df["sv_logp_diff"] = np.nan
        if views:
            rev = views[::-1]
            eq = np.arange(n)[:, None] == np.array(rev)[None, :]
            df["recently_viewed"] = eq.any(axis=1).astype(float)
            df["sv_recency"] = np.where(eq.any(axis=1), 1 / (eq.argmax(axis=1) + 1), 0)
            df["sv_count"] = eq.sum(axis=1)
            for col, label in (("subcategory", "sub"), ("category", "cat"), ("brand", "brand")):
                a = df[col].to_numpy()
                df[f"sv_same_{label}_last"] = (a == a[rev[0]]).astype(float)
                df[f"sv_{label}_share"] = (a[:, None] == a[views][None, :]).mean(axis=1)
            df["sv_logp_diff"] = df.logp - df.logp.iloc[views].mean()
            for kind, col in (("buy", "view_to_purchase_score"), ("view", "coview_score")):
                score = np.zeros(n)
                for j, p in enumerate(rev):
                    for q, value in self.neighbors[kind].get(p, []):
                        score[q] += bd.DECAY ** j * value
                df[col] = score
        for col in ("u_days_since_last", "u_logp_diff", "u_buy_logp_diff", "up_days_since_seen",
                    "days_since_last_purchase"):
            df[col] = np.nan
        history = self.users.get(req.get("user_id"), [])
        df["is_new_user"] = int(not history)
        if history:
            h = np.array(history, dtype=float)
            ix = h[:, 0].astype(int)
            counts, w, w7 = h[:, 6], h[:, 7], h[:, 8]
            buys = h[:, 3] + h[:, 5]
            df["u_events"], df["u_buys"] = counts.sum(), buys.sum()
            df["u_days_since_last"] = (ts - h[:, 9].max()) / 86400
            df["u_logp_diff"] = df.logp - np.average(df.logp.iloc[ix], weights=counts)
            if buys.sum():
                df["u_buy_logp_diff"] = df.logp - np.average(df.logp.iloc[ix], weights=buys)
            for col, label in (("subcategory", "sub"), ("brand", "brand"), ("category", "cat")):
                labels = df[col].iloc[ix].to_numpy()
                def aggregate(weight):
                    return df[col].map(pd.Series(weight).groupby(labels).sum()).fillna(0).to_numpy()
                df[f"u_{label}_share"] = aggregate(w) / w.sum()
                if label != "cat":
                    df[f"u_{label}_w"] = aggregate(w)
                    df[f"u_{label}_buys"] = aggregate(buys)
                if label == "sub":
                    df["u_sub_w_7d"] = aggregate(w7)
            for j, col in enumerate(("up_views", "up_carts", "up_buys", "up_c_clicks", "up_c_buys"), 1):
                df.loc[ix, col] = h[:, j]
            df.loc[ix, "up_days_since_seen"] = (ts - h[:, 9]) / 86400
            db = np.where(h[:, 10] > 946684800, (ts - h[:, 10]) / 86400, np.nan)
            df.loc[ix, "days_since_last_purchase"] = db
        df["previous_purchase_count"] = df.up_buys + df.up_c_buys
        df["already_purchased"] = (df.previous_purchase_count > 0).astype(float)
        return df[list(V2_NOATTRS.features)]


if __name__ == "__main__":
    build_snapshot()
