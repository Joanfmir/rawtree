"""Serving policy for the trained v2 model, with atomic background refresh."""
import pickle
import subprocess
import sys
import threading
import time

import numpy as np

from ..bird.live import LiveFeatures, SNAPSHOT
from ..bird.model import BirdModel, model_dir
from ..config import ROOT
from .base import Policy


class BirdV2(Policy):
    name = "bird_v2"
    refresh_every = 600.0

    def fit(self, state):
        if self.until:
            raise ValueError("bird_v2 live snapshots cannot be used for historical evaluation")
        self.model = BirdModel.load(model_dir("v2_noattrs"))
        self.fs = LiveFeatures(pickle.loads(SNAPSHOT.read_bytes()))
        self._building = False
        self._loaded_at = time.time()

    def refresh(self, state):
        if self._building or time.time() - self._loaded_at < self.refresh_every:
            return
        self._building = True

        def work():
            try:
                subprocess.run([sys.executable, "-u", "-m", "reco.bird.live"], cwd=ROOT,
                               check=True, timeout=900)
                self.fs = LiveFeatures(pickle.loads(SNAPSHOT.read_bytes()))
                self._loaded_at = time.time()
                print(f"[bird_v2] snapshot updated: {self.fs.as_of}")
            except Exception as exc:
                print(f"[bird_v2] refresh failed; keeping previous snapshot: {exc}")
            finally:
                self._building = False

        threading.Thread(target=work, daemon=True).start()

    def recommend(self, req, state):
        fs = self.fs
        score = self.model.predict(fs.build(req))
        snap = state.snapshot()
        return [fs.item_ids[i] for i in np.argsort(-score, kind="stable")
                if snap.is_valid(fs.item_ids[i])][:30]
