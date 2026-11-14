"""reco.offline_live sin red: ventanas del leaderboard, combinación, métricas por fila, acuerdo y caché."""
import unittest

import numpy as np
import pandas as pd

from reco import offline_live as ol
from reco.evaluate import _metrics


class WindowsTest(unittest.TestCase):
    def test_contiguous_and_bounded(self):
        start, end = 1_000_200, 1_000_200 + 8 * 3600 + 600
        w = ol.windows(start, end)
        self.assertEqual(w[0][0], start // 300 * 300)
        self.assertEqual(w[-1][1], end // 300 * 300)
        for (a, b), (c, _) in zip(w, w[1:]):
            self.assertEqual(b, c)                      # sin huecos ni solapes
        for a, b in w:
            self.assertTrue(900 <= b - a <= 10800)
            self.assertEqual(a % 300, 0)

    def test_too_short(self):
        self.assertEqual(ol.windows(0, 600), [])


class CombineTest(unittest.TestCase):
    def test_weighted_by_requests(self):
        rows = pd.DataFrame([
            dict(model_version="a", requests=1000, conversion=1.0, cold_requests=100, cold_conversion=2.0,
                 revenue_per_request=1.0),
            dict(model_version="a", requests=3000, conversion=2.0, cold_requests=300, cold_conversion=4.0,
                 revenue_per_request=0.0),
            dict(model_version="b", requests=500, conversion=0.5, cold_requests=0, cold_conversion=None,
                 revenue_per_request=0.2)])
        out = ol.combine(rows)
        self.assertEqual(list(out.index), ["a", "b"])
        self.assertAlmostEqual(out.loc["a", "conv"], 1.75)
        self.assertAlmostEqual(out.loc["a", "cold_conv"], 3.5)
        self.assertAlmostEqual(out.loc["a", "rev"], 0.25)
        self.assertTrue(np.isnan(out.loc["b", "cold_conv"]))


def _df():
    return pd.DataFrame({
        "purchased": ["p1", "", "p3", "p4"],
        "clicked": ["p1", "p2", "p3", "p4"],
        "shown": [["p1", "x"], ["p2"], ["y", "p3"], ["p4"]],
        "propensity": [[0.5, 0.1], [0.2], [0.3, 0.25], [0.1]],
    })


class MetricsTest(unittest.TestCase):
    def test_per_row_matches_evaluate(self):
        df = _df()
        recs = [["p1"] + [f"z{i}" for i in range(9)], ["p2"] * 10,
                [f"z{i}" for i in range(9)] + ["p3"], [f"q{i}" for i in range(10)]]
        w, g, h = ol.per_row(recs, df)
        ref = _metrics(recs, df)
        self.assertAlmostEqual((w * g).sum() / w.sum(), ref["ips_ndcg"])
        self.assertAlmostEqual((w * h).sum() / w.sum(), ref["ips_hit"])
        self.assertAlmostEqual(h.mean(), ref["hit@10"])

    def test_bootstrap_identical_is_zero(self):
        df = _df()
        recs = [["p1"] * 10, ["p2"] * 10, ["p3"] * 10, ["x"] * 10]
        lists = {False: recs, True: recs}
        d, lo, hi = ol.bootstrap_diff(lists, lists, df, n=200)
        self.assertEqual((d, lo, hi), (0.0, 0.0, 0.0))

    def test_bootstrap_sign(self):
        df = _df()
        good = {False: [["p1"] * 10, ["p2"] * 10, ["p3"] * 10, ["p4"] * 10]}
        bad = {False: [["x"] * 10] * 4}
        d, lo, hi = ol.bootstrap_diff(good, bad, df, n=200)
        self.assertGreater(d, 0)
        self.assertGreaterEqual(lo, 0)


class AgreementTest(unittest.TestCase):
    def test_perfect_and_inverted(self):
        names = ["a", "b", "c", "d"]
        live = np.array([1.0, 2.0, 3.0, 4.0])
        se = np.full(4, 0.1)
        a = ol.agreement(names, np.array([0.1, 0.2, 0.3, 0.4]), live, se)
        self.assertEqual(a["pairs"], (6, 6))
        self.assertEqual(a["clear_pairs"], (6, 6))
        self.assertAlmostEqual(a["spearman"], 1.0)
        b = ol.agreement(names, np.array([0.4, 0.3, 0.2, 0.1]), live, se)
        self.assertEqual(b["pairs"], (0, 6))
        self.assertEqual(len(b["wrong"]), 6)

    def test_clear_only_when_separated(self):
        a = ol.agreement(["a", "b", "c"], np.array([0.2, 0.1, 0.3]), np.array([1.0, 1.05, 3.0]), np.full(3, 0.1))
        self.assertEqual(a["clear_pairs"], (2, 2))   # a-b no está clara: no cuenta como fallo claro
        self.assertEqual(a["wrong"], [("a", "b", False)])


class CacheKeyTest(unittest.TestCase):
    def test_key_depends_on_policy_code(self):
        self.assertNotEqual(ol.policy_key("baseline", "2026-11-07"), ol.policy_key("personal", "2026-11-07"))
        self.assertEqual(ol.policy_key("baseline", "2026-11-07"), ol.policy_key("baseline", "2026-11-07"))

    def test_imports_are_followed(self):
        from reco.config import ROOT
        files = {p.relative_to(ROOT).as_posix() for p in ol._reco_imports(ROOT / "reco" / "policies" / "bird.py")}
        self.assertIn("reco/bird/model.py", files)
        self.assertIn("reco/policies/base.py", files)

    def test_stale_lists_do_not_match_prefixed_policies(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            for n in ("personal_aaaaaaaaaaaa", "personal_v3_bbbbbbbbbbbb", "personal_v4_cccccccccccc"):
                (Path(tmp) / f"lists_2026-11-07_{n}.pkl").touch()
            got = sorted(p.name for p in ol.stale_lists("personal", "2026-11-07", Path(tmp)))
            self.assertEqual(got, ["lists_2026-11-07_personal_aaaaaaaaaaaa.pkl"])
            got = [p.name for p in ol.stale_lists("personal_v3", "2026-11-07", Path(tmp))]
            self.assertEqual(got, ["lists_2026-11-07_personal_v3_bbbbbbbbbbbb.pkl"])

    def test_live_versions_are_registered(self):
        from reco.policies import POLICIES
        for v, p in ol.LIVE_VERSIONS.items():
            if p is not None:
                self.assertIn(p, POLICIES, v)


if __name__ == "__main__":
    unittest.main()
