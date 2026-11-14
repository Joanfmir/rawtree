"""Fórmulas del dataset v5 (reco/bird/feat5.py), sin red: lo que se promete en docs/DATASET_V5.md."""
import unittest

import numpy as np
import pandas as pd

from reco.bird import feat5
from reco.bird.schema import V2_NOREBUY, V5
from reco.features import ATTRS


def catalog(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sub = rng.integers(0, 6, n)
    return pd.DataFrame({
        "product_id": [f"p_{i:04d}" for i in range(n)],
        "category": [f"cat_{s // 2}" for s in sub], "subcategory": [f"sub_{s}" for s in sub],
        "brand": [f"b_{b}" for b in rng.integers(0, 8, n)],
        "attrs": [list(rng.choice(ATTRS, 2, replace=False)) for _ in range(n)],
        "base_price": np.round(np.exp(rng.normal(3, 1, n)), 2),
        "created_at": pd.Timestamp("2026-01-01"),
    })


def counts(n: int, seed: int = 1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    views = rng.integers(50, 5000, (1, n)).astype(float)
    c = {"views": views, "carts": np.round(views * 0.08), "buys": np.round(views * 0.04)}
    for key, share in (("views_7d", 0.12), ("views_1d", 0.02), ("views_1h", 0.001)):
        c[key] = np.round(views * share)
    c["buys_7d"], c["buys_1d"] = np.round(c["views_7d"] * 0.04), np.round(c["views_1d"] * 0.04)
    return c


class Feat5Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = feat5.Items(catalog())

    def test_esquema(self):
        self.assertEqual(len(set(V5.features)), len(V5.features))
        for f in ("u_events", "u_buys", "it_buys_7d", "it_views_7d", "it_buys_1d", "u_days_since_last"):
            self.assertNotIn(f, V5.features)             # recuentos absolutos: dependen del volumen de tráfico
        for group in (feat5.ITEM_POP, feat5.ITEM_FB, feat5.CTX_LIFT, feat5.SESSION, feat5.USER, feat5.AFFINITY):
            self.assertFalse(set(group) - set(V5.features), group)
        self.assertFalse({"already_purchased", "previous_purchase_count", "up_buys"} & set(V5.features))
        self.assertEqual(V5.categorical, V2_NOREBUY.categorical)

    def test_popularidad_no_depende_del_volumen(self):
        """El stream en vivo tiene ~29 veces más eventos por hora que el histórico: con 29 veces más de todo,
        cuotas, tasas y puestos no se mueven, y las tendencias (suavizadas) casi nada."""
        c = counts(self.items.n)
        a = feat5.item_pop(c, self.items.sub)
        b = feat5.item_pop({k: v * 29 for k, v in c.items()}, self.items.sub)
        for name in ("it_buy_share", "it_view_share", "it_buy_share_7d", "it_view_share_7d", "it_buy_share_1d",
                     "it_sub_rank"):
            np.testing.assert_allclose(a[name], b[name], rtol=1e-9, err_msg=name)
        for name in ("it_buy_rate", "it_cart_rate"):
            np.testing.assert_allclose(a[name], b[name], rtol=0.05, err_msg=name)
        self.assertLess(np.abs(a["it_trend_1d"] - b["it_trend_1d"]).max(), 0.7)

    def test_producto_nuevo_hereda_de_su_subcategoria(self):
        c = counts(self.items.n)
        for k in c:
            c[k][0, 0] = 0                                 # producto 0: recién añadido, sin un solo evento
        f = feat5.item_pop(c, self.items.sub)
        same = self.items.sub == self.items.sub[0]
        rate = c["buys"][0, same].sum() / c["views"][0, same].sum()
        self.assertAlmostEqual(f["it_buy_rate"][0, 0], rate, places=9)
        self.assertEqual(f["it_buy_share"][0, 0], 0.0)
        self.assertAlmostEqual(f["it_trend_1h"][0, 0], 0.0, places=9)    # sin evidencia: tendencia neutra
        rng = np.random.default_rng(3)
        by_ctx = np.vstack([c["buys"][0] * rng.uniform(0.5, 1.5, self.items.n) for _ in range(2)])
        lifts = feat5.context_lifts(by_ctx, self.items, "dev", feat5.DEV_LEVELS)
        np.testing.assert_allclose(lifts["ctx_dev_item"][:, 0], lifts["ctx_dev_sub"][:, 0], atol=1e-9)

    def test_tendencia_detecta_un_pico(self):
        c = counts(self.items.n)
        base = feat5.item_pop(c, self.items.sub)["it_trend_1h"][0, 3]
        c["views_1h"][0, 3] += 200
        self.assertGreater(feat5.item_pop(c, self.items.sub)["it_trend_1h"][0, 3], base + 1.0)

    def test_contexto_lift(self):
        n = self.items.n
        m = np.ones((3, n)) * 20
        target = self.items.cat == self.items.cat[0]
        m[0, target] *= 3                                  # el contexto 0 compra el triple de esa categoría
        out = feat5.context_lifts(m, self.items, "dev", feat5.DEV_LEVELS)
        self.assertEqual(set(out), {f"ctx_dev_{x}" for x in feat5.DEV_LEVELS})
        self.assertGreater(out["ctx_dev_cat"][0, target].min(), 0.2)
        self.assertLess(out["ctx_dev_cat"][0, ~target].max(), 0.0)
        self.assertLess(np.abs(out["ctx_dev_cat"][1]).max(), 0.35)
        for v in out.values():
            self.assertEqual(v.shape, (3, n))
            self.assertTrue(np.isfinite(v).all())

    def test_feedback_sin_impresiones_es_neutro(self):
        f = feat5.feedback_lift(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
        np.testing.assert_allclose(f["it_fb_click_lift"], 1.0)
        np.testing.assert_allclose(f["it_fb_buy_lift"], 1.0)
        g = feat5.feedback_lift(np.array([30.0]), np.array([10.0]), np.array([0.0]), np.array([4.0]))
        self.assertGreater(g["it_fb_click_lift"][0], 2.0)
        self.assertLess(g["it_fb_buy_lift"][0], 0.5)

    def test_sesion(self):
        it = self.items
        nb = {"buy": {5: [(7, 1.0), (9, 0.5)]}, "view": {}}
        empty = feat5.session_block(it, np.zeros(0, dtype=int), np.nan, nb)
        self.assertEqual(set(empty), set(feat5.SESSION))
        self.assertTrue(np.isnan(empty["sv_logp_diff"]).all() and (empty["sv_sub_share"] == 0).all())
        s = feat5.session_block(it, np.array([2, 5]), 30.0, nb)        # vio 2 y, lo último, 5
        self.assertEqual(s["n_recent_views"][0], 2)
        self.assertEqual(s["recently_viewed"][5], 1.0)
        self.assertEqual(s["sv_recency"][5], 1.0)
        self.assertEqual(s["sv_recency"][2], 0.5)
        self.assertEqual(s["view_to_purchase_score"][7], 1.0)
        self.assertEqual(s["sv_secs_since_last"][0], 30.0)
        self.assertEqual(s["sv_attr_last"][5], float(len({it.a1[5], it.a2[5]})))
        self.assertAlmostEqual(s["sv_sub_w"][5], (1.0 + (0.6 if it.sub[2] == it.sub[5] else 0.0)) / 1.6)
        self.assertEqual(s["sv_logp_diff_last"][5], 0.0)
        for name in feat5.SESSION:
            self.assertEqual(len(s[name]), it.n, name)

    def test_par_guardado_igual_que_el_estado_de_servicio(self):
        # (vistas, carritos, clics, compras): una sola vista no se guarda; lo demás sí
        got = feat5.stored_pair([1, 2, 1, 1, 1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0])
        self.assertEqual(got.tolist(), [False, True, True, True, True, False])

    def test_mediana_de_precio_solo_con_el_catalogo_inicial(self):
        cat = catalog()
        new = cat.iloc[[0]].assign(product_id="p_9999", base_price=9999.0, created_at=pd.Timestamp("2026-11-14 10:00"))
        both = feat5.Items(pd.concat([cat, new], ignore_index=True).sort_values("product_id").reset_index(drop=True))
        np.testing.assert_allclose(both.price_vs_sub[:-1], self.items.price_vs_sub)   # un producto nuevo no la mueve

    def test_parse_views(self):
        views = [{"product_id": "p_0002", "ts": "2026-11-14 16:01:40.000000000"},
                 {"product_id": "no_existe", "ts": "2026-11-14 16:01:50"},
                 {"product_id": "p_0005", "ts": "2026-11-14T16:01:58Z"}]
        idx, secs = feat5.parse_views(self.items, views, np.datetime64("2026-11-14T16:02:11.482"))
        self.assertEqual(idx.tolist(), [2, 5])
        self.assertAlmostEqual(secs, 13.482, places=3)
        self.assertEqual(feat5.parse_views(self.items, None, None)[0].tolist(), [])

    def test_usuario(self):
        n = 4
        zero = {k: np.zeros(n) for k in feat5.USER_AGG}
        zero.update({k: np.full(n, np.nan) for k in ("u_logp", "u_buy_logp", "u_logp_sd")})
        f = feat5.user_feats(zero, np.ones(n))
        self.assertEqual(set(f), set(feat5.USER + feat5.AFFINITY))
        for name in feat5.AFFINITY + ("u_hist_conf", "u_buy_ratio"):
            self.assertTrue((f[name] == 0).all(), name)             # sin historia: cuotas a 0
        for name in ("u_logp_diff", "u_buy_logp_diff", "u_logp_z"):
            self.assertTrue(np.isnan(f[name]).all(), name)          # ... y precio relativo desconocido
        a = {k: np.full(n, v) for k, v in dict(
            u_events=100, u_buys=10, u_w=200, u_r=20, u_logp=3.0, u_buy_logp=2.5, u_logp_sd=0.5, u_sub_w=50,
            u_brand_w=20, u_cat_w=100, u_attr_w1=80, u_attr_w2=40, u_sub_r=10, u_cat_r=15, u_attr_r=20,
            u_sub_buys=5, u_brand_buys=2, u_cat_buys=8, u_attr_buys=10).items()}
        f = feat5.user_feats(a, np.full(n, 3.5))
        self.assertAlmostEqual(f["u_attr_share"][0], 0.3)
        self.assertAlmostEqual(f["u_attr_max_share"][0], 0.4)
        self.assertAlmostEqual(f["u_attr_min_share"][0], 0.2)
        self.assertAlmostEqual(f["u_sub_share_rec"][0], 0.5)
        self.assertAlmostEqual(f["u_attr_buy_share"][0], 0.5)
        self.assertAlmostEqual(f["u_logp_z"][0], 1.0)
        self.assertAlmostEqual(f["u_hist_conf"][0], 100 / 150)
        b = feat5.user_feats({k: (v * 29 if k not in ("u_logp", "u_buy_logp", "u_logp_sd") else v) for k, v in a.items()},
                             np.full(n, 3.5))
        for name in feat5.AFFINITY + ("u_buy_ratio", "u_logp_diff", "u_logp_z"):
            np.testing.assert_allclose(f[name], b[name], err_msg=name)   # 29 veces más actividad: mismo perfil


if __name__ == "__main__":
    unittest.main()
