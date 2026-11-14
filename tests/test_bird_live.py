"""Live v2 feature parity and history semantics, without network access."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from reco.bird import build_dataset as bd
from reco.bird.live import LiveFeatures
from reco.bird.schema import SESSION, V2, V2_NOATTRS


class BirdLiveTest(unittest.TestCase):
    def setUp(self):
        self.items = pd.DataFrame({
            "product_id": ["a", "b", "c"], "category": ["x", "x", "y"],
            "subcategory": ["s", "s", "t"], "brand": ["m", "n", "m"],
            "attr1": ["compact", "compact", "sporty"], "attr2": ["premium", "sporty", "premium"],
            "price_eur": [9., 19., 29.], "created_at": ["2026-09-01"] * 3,
        })
        self.items["logp"] = np.log1p(self.items.price_eur)
        for c in V2_NOATTRS.features:
            if c.startswith("it_"):
                self.items[c] = 0.
        self.neighbors = {"buy": {1: [(2, 0.5)]}, "view": {0: [(1, 1.)]}}
        self.snapshot = dict(items=self.items, users={}, neighbors=self.neighbors, as_of="2026-11-14 10:00:00")
        self.req = {"user_id": "u", "ts": "2026-11-14 11:00:00", "context": {"page": "home"},
                    "recent_views": [{"product_id": "a"}, {"product_id": "b"}, {"product_id": "a"}]}

    def test_session_features_match_training(self):
        fs = LiveFeatures(self.snapshot)
        actual = fs.build(self.req)
        sess = pd.DataFrame({"ts": [pd.Timestamp(self.req["ts"])], "recent_views": [self.req["recent_views"]]})
        shown = np.array([[0, 1, 2, 0, 1, 2, 0, 1, 2, 0]])
        with patch.object(bd, "_neighbors", return_value=self.neighbors):
            expected = bd.session_features(sess, shown, self.items)[0, :3]
        retained = [i for i, name in enumerate(SESSION) if name in V2_NOATTRS.features]
        np.testing.assert_allclose(actual[[SESSION[i] for i in retained]].to_numpy(), expected[:, retained], equal_nan=True)
        self.assertEqual(list(actual.columns), list(V2_NOATTRS.features))
        self.assertTrue((actual.is_new_user == 1).all())
        self.assertTrue(actual.u_logp_diff.isna().all())

    def test_user_history_weights_and_purchase_sources(self):
        seen = pd.Timestamp("2026-11-13 11:00:00").timestamp()
        # a: view + organic purchase; b: cart + carousel purchase.
        self.snapshot["users"] = {"u": [(0, 1, 0, 1, 0, 0, 2, 6, 6, seen, seen),
                                         (1, 0, 1, 0, 0, 1, 2, 8, 8, seen, seen)]}
        x = LiveFeatures(self.snapshot).build(self.req)
        np.testing.assert_allclose(x.u_events, 4)
        np.testing.assert_allclose(x.u_buys, 2)
        np.testing.assert_allclose(x.u_sub_share, [1, 1, 0])
        np.testing.assert_allclose(x.u_brand_w, [6, 8, 6])
        np.testing.assert_allclose(x.previous_purchase_count, [1, 1, 0])
        np.testing.assert_allclose(x.days_since_last_purchase[:2], [1, 1])
        self.assertTrue(np.isnan(x.days_since_last_purchase.iloc[2]))
        np.testing.assert_allclose(x.u_days_since_last, 1)
        np.testing.assert_allclose(x.u_logp_diff, self.items.logp - self.items.logp[:2].mean())

    def test_no_attribute_features_in_training_or_live_schema(self):
        excluded = {"attr1", "attr2", "sv_attr_share", "u_attr_share"}
        self.assertEqual(len(V2_NOATTRS.features), 57)
        self.assertEqual(set(V2.features) - set(V2_NOATTRS.features), excluded)
        self.assertFalse(excluded & set(V2_NOATTRS.categorical))
        self.assertFalse(excluded & set(LiveFeatures(self.snapshot).build(self.req).columns))

    def test_rejects_requests_older_than_snapshot(self):
        self.req["ts"] = "2026-11-14 09:59:59"
        with self.assertRaises(ValueError):
            LiveFeatures(self.snapshot).build(self.req)


if __name__ == "__main__":
    unittest.main()
