"""Política joan y descarga del estado de Bird, sin red."""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from reco.bird import live
from reco.policies import POLICIES
from reco.policies.joan import Joan, JoanRerank


class SafeDownloadTest(unittest.TestCase):
    """Dos descargas a la vez del mismo estado no pueden dejar Parquets truncados (pasó el 14/11)."""

    def test_concurrent_downloads_leave_valid_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "state_live"

            def slow_query(sql, timeout=None):
                time.sleep(0.05)                  # da tiempo a que las dos descargas se crucen
                n = int(sql)
                return [{"user_id": f"u{i}", "k": "x" * 50, "v": float(i)} for i in range(n)]

            qs = {f"t{j}": str(2000 + j) for j in range(4)}
            with mock.patch.object(live, "folder", lambda until: out), \
                    mock.patch.object(live, "queries", lambda until: qs), \
                    mock.patch.object(live, "query", slow_query):
                threads = [threading.Thread(target=live.download, kwargs={"verbose": False}) for _ in range(3)]
                for th in threads:
                    th.start()
                for th in threads:
                    th.join()
            for name, n in qs.items():
                self.assertEqual(len(pd.read_parquet(out / f"{name}.parquet")), int(n))
            self.assertEqual([p.name for p in Path(tmp).glob("state_live.tmp-*")], [])   # temporales limpios

    def test_failed_download_keeps_previous_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "state_live"
            out.mkdir()
            pd.DataFrame({"a": [1, 2]}).to_parquet(out / "t0.parquet")

            def broken(sql, timeout=None):
                raise RuntimeError("RawTree caído")
            with mock.patch.object(live, "folder", lambda until: out), \
                    mock.patch.object(live, "queries", lambda until: {"t0": "1"}), \
                    mock.patch.object(live, "query", broken):
                with self.assertRaises(RuntimeError):
                    live.download(verbose=False)
            self.assertEqual(len(pd.read_parquet(out / "t0.parquet")), 2)


class RerankMixTest(unittest.TestCase):
    def _policy(self, ranker_scores, w):
        p = JoanRerank.__new__(JoanRerank)
        p.w, p.top = w, 30
        p.live = mock.Mock(ids=np.array([f"p{i}" for i in range(10)]))
        p.ranker = mock.Mock(k=5, predict=lambda num, cat: np.asarray(ranker_scores, float))
        return p

    def test_mix_of_positions(self):
        # bird ordena p0..p4; el reordenador prefiere p4 > p3 > p2 > p1 > p0
        idx = np.arange(5)
        with mock.patch("reco.policies.joan.candidates", lambda *a: (idx, np.zeros((5, 1)), np.zeros((5, 1)))):
            self.assertEqual(self._policy([0, 1, 2, 3, 4], w=0.0).recommend({}, None), ["p0", "p1", "p2", "p3", "p4"])
            self.assertEqual(self._policy([0, 1, 2, 3, 4], w=1.0).recommend({}, None), ["p4", "p3", "p2", "p1", "p0"])
            # w=0,5: todos empatan en la clave (2) → manda el orden de bird
            self.assertEqual(self._policy([0, 1, 2, 3, 4], w=0.5).recommend({}, None), ["p0", "p1", "p2", "p3", "p4"])
            # el reordenador pone p3 primero (puestos: p3 0, p0 1, p1 2, p2 3, p4 4); claves con bird p0..p4:
            # p0 0,5 · p1 1,5 · p3 1,5 · p2 2,5 · p4 4 → p3 sube del 4.º al 3.º y empata con p1, que gana por bird
            self.assertEqual(self._policy([0.3, 0.2, 0.1, 9, 0], w=0.5).recommend({}, None),
                             ["p0", "p1", "p3", "p2", "p4"])

    def test_no_candidates(self):
        with mock.patch("reco.policies.joan.candidates", lambda *a: (np.zeros(0, int), None, None)):
            self.assertEqual(self._policy([], w=0.5).recommend({}, None), [])


class RouterTest(unittest.TestCase):
    def test_routes_by_bird_history(self):
        j = Joan.__new__(Joan)
        j.known = mock.Mock(live=mock.Mock(user=lambda req: None if req.get("new") else 1))
        j.known.recommend = lambda req, state: ["k1", "k2"]
        j.gulias = mock.Mock(recommend=lambda req, state: ["g1", "g2", "x"])
        j.blend = mock.Mock(recommend=lambda req, state: ["b1", "x", "b2"])
        j.blend.bird = mock.Mock()
        self.assertEqual(j.recommend({"new": False}, None), ["k1", "k2"])
        out = j.recommend({"new": True}, None)
        self.assertEqual(out[0], "x")                 # en las dos listas: sube arriba con RRF
        self.assertEqual(set(out), {"g1", "g2", "x", "b1", "b2"})
        self.assertIs(j.blend.bird.live, j.known.live)   # un solo estado de Bird

    def test_registered(self):
        self.assertIs(POLICIES["joan"], Joan)
        self.assertIs(POLICIES["joan_rerank"], JoanRerank)


if __name__ == "__main__":
    unittest.main()
