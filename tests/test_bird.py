"""Bird de punta a punta con datos sintéticos, sin red: contrato, entrenamiento, artefacto y política."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from reco.bird import dataset
from reco.bird.builder import FeatureState, build_features
from reco.bird.model import BirdModel, train
from reco.bird.schema import V1, FeatureSchema
from reco.policies.bird import Bird
from reco.safety import finalize
from reco.state import State

FAST = {"iterations": 40, "thread_count": 2}


class BirdTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = dataset.fake_catalog(300, seed=1)
        df = dataset.synthetic(3000, seed=1, items=cls.items)
        cls.train, cls.valid = dataset.split_by_time(df)
        cls.bird = train(cls.train, cls.valid, V1, params=FAST)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = cls.bird.save(Path(cls.tmp.name) / "v1")
        rows = cls.items.to_dict("records")
        cls.catalog = {r["product_id"]: {**r, "is_active": True} for r in rows}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def policy(self, state, per_request=False):
        p = Bird()
        p.folder, p.force_per_request = self.folder, per_request
        p.fit(state)
        return p

    def test_sintetico_cumple_el_contrato(self):
        self.assertEqual(dataset.validate(self.train, self.valid, V1), [])
        self.assertEqual(len(self.train.columns), 12)

    def test_validacion_detecta_incumplimientos(self):
        dup = pd.concat([self.train, self.train.head(1)], ignore_index=True)
        self.assertTrue(any("duplicados" in p for p in dataset.check(dup, V1)))
        bad_pos = self.train.assign(position=11).astype({"position": "int8"})
        self.assertTrue(any("position" in p for p in dataset.check(bad_pos, V1)))
        nulls = self.train.assign(brand=self.train.brand.where(self.train.index > 5))
        self.assertTrue(any("brand" in p for p in dataset.check(nulls, V1)))
        with self.assertRaises(ValueError):   # la misma petición en los dos splits y validación no posterior
            dataset.validate(self.train, self.train, V1)

    def test_prepare_trata_los_ausentes(self):
        raw = pd.DataFrame({"category": ["a", None], "subcategory": ["", "s"], "brand": ["b", "b"],
                            "price_eur": [1.5, None], "position": [1, 2]})
        X = dataset.prepare(raw, V1)
        self.assertEqual(list(X.columns), list(V1.features))
        self.assertEqual(X.category.tolist(), ["a", "UNKNOWN"])
        self.assertEqual(X.subcategory.tolist(), ["UNKNOWN", "s"])
        self.assertTrue(np.isnan(X.price_eur[1]))

    def test_el_esquema_rechaza_fugas(self):
        with self.assertRaises(ValueError):
            FeatureSchema("x", features=("category", "target_click"), categorical=("category",))

    def test_guardar_y_cargar_da_lo_mismo(self):
        again = BirdModel.load(self.folder)
        self.assertEqual(again.schema, V1)
        np.testing.assert_allclose(self.bird.predict(self.valid), again.predict(self.valid))
        p = again.predict(self.valid)
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        self.assertIn("pr_auc", again.info["metrics"]["clasificacion"])

    def test_nueva_feature_solo_cambia_el_esquema(self):
        v2 = FeatureSchema("v2", features=(*V1.features, "already_purchased"), categorical=V1.categorical,
                           per_request=("already_purchased",))
        rng = np.random.default_rng(0)
        tr = self.train.assign(already_purchased=rng.integers(0, 2, len(self.train)))
        va = self.valid.assign(already_purchased=rng.integers(0, 2, len(self.valid)))
        bird = train(tr, va, v2, params=FAST)
        self.assertEqual(list(bird.model.feature_names_), list(v2.features))
        self.assertFalse(v2.is_static)

    def test_builder_respeta_orden_y_posicion(self):
        fs = FeatureState.from_items(self.items)
        ids = ["p_000005", "no_existe", "p_000001"]
        X = build_features({}, ids, fs, V1)
        self.assertEqual(list(X.index), ids)
        self.assertEqual(list(X.columns), list(V1.features))
        self.assertTrue((X.position == V1.reference_position).all())
        self.assertEqual(len(self.bird.predict(X)), 3)   # un producto desconocido no rompe la predicción

    def test_recomienda_diez_validos_por_los_dos_caminos(self):
        out_of_stock = set(self.items.product_id[:150])
        state = State(catalog=self.catalog, unavailable=out_of_stock, best=list(self.items.product_id[-20:]))
        req = {"request_id": "r", "user_id": "u", "is_new_user": False, "recent_views": [], "context": {}}
        fast = self.policy(state).recommend(req, state)
        slow = self.policy(state, per_request=True).recommend(req, state)
        self.assertEqual(fast, slow)
        recs = finalize(fast, state)
        self.assertEqual(recs, fast)
        self.assertEqual(len(set(recs)), 10)
        self.assertFalse(set(recs) & out_of_stock)
        # ordenado de mayor a menor probabilidad
        fs = FeatureState.from_catalog(self.catalog)
        score = self.bird.predict(build_features(req, recs, fs, V1))
        self.assertTrue((np.diff(score) <= 1e-12).all())

    def test_refresh_recoge_cambios_de_catalogo(self):
        state = State(catalog=dict(self.catalog), unavailable=set(), best=list(self.items.product_id[:20]))
        policy = self.policy(state)
        top = policy.recommend({}, state)[0]
        state.catalog = {k: v for k, v in state.catalog.items() if k != top}
        policy.refresh(state)
        self.assertNotIn(top, policy.recommend({}, state))


if __name__ == "__main__":
    unittest.main()
