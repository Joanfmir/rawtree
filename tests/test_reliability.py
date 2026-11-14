"""Regresiones de fiabilidad con datos sintéticos y sin red."""
import ast
import unittest
import zlib
import tempfile
import threading
from reco import features
import time
from reco.policies.ranker import Ranker
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from reco.state import State
from reco.safety import finalize
from reco.policies.personal import Personal, PersonalV1, _epoch
from reco.ranker import Featurizer
from reco import catalog


def datos():
    ids = [f'p{i}' for i in range(30)]
    items = pd.DataFrame(dict(product_id=ids, category=['a']*15+['b']*15,
        subcategory=['s']*30, brand=['m']*30, price_eur=[10.]*30,
        is_active=[True]*30, is_new=[0]*30, views=[20]*30, carts=[2]*30,
        purchases=list(range(30)), views_7d=[20]*30,
        purchases_7d=list(range(30)), purchases_1d=[1]*30))
    return dict(item_features=items,
        item_neighbors=pd.DataFrame(columns=['p','q','n','kind']),
        user_subcat=pd.DataFrame(columns=['user_id','subcategory','w']),
        user_brand=pd.DataFrame(columns=['user_id','brand','w']),
        user_cat=pd.DataFrame(columns=['user_id','category','w']),
        user_stats=pd.DataFrame(columns=['user_id','n_events','buy_logp','avg_logp']),
        user_purchases=pd.DataFrame(columns=['user_id','product_id','n','last_ts']),
        context_popular=pd.DataFrame(columns=['device','region','product_id','n']))


def estado():
    return State(catalog={f'p{i}': {'is_active': True} for i in range(30)}, best=['p0'])


class Fiabilidad(unittest.TestCase):
    def test_catalogo_vacio_no_acepta_desconocidos(self):
        self.assertFalse(State().is_valid('desconocido'))

    def test_finalize_rellena_desde_catalogo(self):
        s = estado()
        s.unavailable = {'p1'}
        s.catalog['p2']['is_active'] = False
        out = finalize(['p0', 'p0', 'p1', 'p2', 'inventado'], s)
        self.assertEqual(len(out), 10)
        self.assertEqual(len(set(out)), 10)
        self.assertTrue(all(s.is_valid(p) for p in out))
        self.assertEqual(len(s.fallback()), 10)

    def test_ids_nulos_y_no_hashables(self):
        s = estado()
        self.assertEqual(len(finalize([None, {}, [], float('nan')], s)), 10)
        self.assertFalse(s.is_valid(None))

    def test_catalogo_insuficiente_falla_explicito(self):
        with self.assertRaises(ValueError):
            finalize([], State(catalog={'p': {'is_active': True}}))

    def test_error_stock_no_se_interpreta_como_stock_disponible(self):
        with patch('reco.catalog.query', side_effect=RuntimeError('timeout')):
            with self.assertRaises(RuntimeError):
                catalog.out_of_stock()

    def test_error_catalogo_no_descarta_actualizaciones(self):
        with patch('reco.catalog.query', side_effect=RuntimeError('timeout')):
            with self.assertRaises(RuntimeError):
                catalog.current_catalog()

    def test_refresh_fallido_conserva_estado(self):
        s = estado()
        s.unavailable = {'p1'}
        with patch('reco.catalog.best_sellers', return_value=['p0']), \
             patch('reco.catalog.current_catalog', return_value={'nuevo': {'is_active': True}}), \
             patch('reco.catalog.out_of_stock', side_effect=RuntimeError('timeout')):
            with self.assertRaises(RuntimeError):
                s.refresh(catalog_every=0, stock_every=0)
        self.assertTrue(s.is_valid('p0'))
        self.assertFalse(s.is_valid('p1'))

    def test_personal_vistas_nulas_desconocidas_usuario_nulo(self):
        Personal._shared['test'] = Personal._index(datos())
        for cls in (PersonalV1, Personal):
            p = cls(until='test')
            for views in (None, [], [None, {}, {'product_id': None}, {'product_id': 'nuevo'}]):
                out = finalize(p.recommend(dict(recent_views=views, user_id=None, context=None), estado()), estado())
                self.assertEqual(len(set(out)), 10)

    def test_categoria_v2_y_v1(self):
        Personal._shared['test'] = Personal._index(datos())
        req = {'recent_views': [{'product_id': 'p0'}], 'is_new_user': True}
        v2 = Personal(until='test').recommend(req, estado())
        v1 = PersonalV1(until='test').recommend(req, estado())
        self.assertTrue(all(int(p[1:]) < 15 for p in v2))
        self.assertTrue(any(int(p[1:]) >= 15 for p in v1))

    def test_fecha_con_zona_y_milisegundos(self):
        self.assertAlmostEqual(_epoch('2026-11-15T10:00:00.123+01:00'),
                               _epoch('2026-11-15T09:00:00.123Z'))
        self.assertAlmostEqual(_epoch('2026-11-15T09:00:00.123Z') % 1, .123, places=5)

    def test_ranker_none_nan_y_catalogo_totalmente_nuevo(self):
        f = datos()
        f['item_features']['price_eur'] = None
        f['item_features']['is_new'] = 1
        fz = Featurizer(f)
        req = {'recent_views': [None, {}, {'product_id': None}, {'product_id': 'desconocido'}], 'context': None}
        cand = fz.candidates(req, None)
        self.assertTrue(np.isfinite(fz.matrix(req, None, cand)).all())

    def test_producto_nuevo_inactivo_y_sin_stock(self):
        s = estado()
        s.catalog.update(nuevo={'is_active': True}, inactivo={'is_active': False}, nulo={'is_active': None})
        s.unavailable.add('p0')
        self.assertTrue(s.is_valid('nuevo'))
        for p in ('inactivo', 'nulo', 'p0', 'ausente'):
            self.assertFalse(s.is_valid(p))
        self.assertEqual(finalize(['nuevo'], s)[0], 'nuevo')

    def test_snapshot_aislado_del_refresh(self):
        s = estado()
        old = s.snapshot()
        with patch('reco.catalog.current_catalog', return_value={'nuevo': {'is_active': True}}), \
             patch('reco.catalog.out_of_stock', return_value={'p0'}):
            s.refresh(catalog_every=0, stock_every=0)
        self.assertTrue(old.is_valid('p0'))
        self.assertFalse(s.is_valid('p0'))
        self.assertTrue(s.is_valid('nuevo'))

    def test_refresh_usa_reloj_monotono(self):
        s = estado()
        with patch('reco.state.time.monotonic', return_value=100), \
             patch('reco.state.time.time', side_effect=AssertionError('reloj local')), \
             patch('reco.catalog.current_catalog', return_value=s.catalog), \
             patch('reco.catalog.out_of_stock', return_value=set()):
            s.refresh()
        self.assertEqual(s._stock_at, 100)

    def test_ranker_refresh_no_bloquea_ni_modifica_featurizer(self):
        p = Ranker()
        p.fz = Featurizer(datos())
        p._live_bought = frozenset()
        p._thread = None
        p._rebuilt_at = time.monotonic()
        entered, release = threading.Event(), threading.Event()
        def consulta(*args, **kwargs):
            entered.set()
            release.wait(3)
            return [{'u': 'u', 'p': 'p0'}]
        with patch('reco.policies.ranker.query', side_effect=consulta):
            try:
                p.refresh(estado())
                self.assertTrue(entered.wait(1))
                self.assertTrue(p._thread.is_alive())
                same = p._thread
                p.refresh(estado())
                self.assertIs(p._thread, same)
                self.assertFalse(p.fz.bought.get('u'))
            finally:
                release.set()
                if p._thread:
                    p._thread.join(3)
        self.assertIn(('u', 'p0'), p._live_bought)
        self.assertFalse(p.fz.bought.get('u'))

    def test_ranker_estadisticas_usuario_nulas(self):
        f = datos()
        f['user_stats'] = pd.DataFrame([dict(user_id='u', n_events=None, buy_logp=None, avg_logp=None)])
        fz = Featurizer(f)
        cand = fz.candidates({}, 'u')
        self.assertTrue(np.isfinite(fz.matrix({}, 'u', cand)).all())

    def test_parquet_publicacion_atomica_y_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            pd.DataFrame({'x': [1]}).to_parquet(dest / 'tabla.parquet')
            original = pd.DataFrame.to_parquet
            def escribir(df, ruta, **kwargs):
                # Durante la escritura solo es visible la versión anterior.
                self.assertEqual(features.load()['tabla'].x.tolist(), [1])
                original(df, ruta, **kwargs)
            with patch('reco.features.folder', return_value=dest), \
                 patch('reco.features.queries', return_value={'tabla': 'SELECT 1'}), \
                 patch('reco.features.query', return_value=[{'x': 2}]):
                with patch.object(pd.DataFrame, 'to_parquet', escribir):
                    features.build(verbose=False)
                self.assertEqual(features.load()['tabla'].x.tolist(), [2])
                with patch.object(pd.DataFrame, 'to_parquet', side_effect=OSError('disco lleno')):
                    with self.assertRaises(OSError):
                        features.build(verbose=False)
                self.assertEqual(features.load()['tabla'].x.tolist(), [2])
                self.assertEqual(len(list(dest.iterdir())), 1)

    def test_ab_crc32_usa_expresion_real_de_serve(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'reco/serve.py').read_text())
        expr = next(n for n in ast.walk(tree) if isinstance(n, ast.BinOp)
                    and isinstance(n.op, ast.Mod) and 'crc32' in ast.unparse(n))
        counts = [0]*3
        for i in range(9000):
            req = {'request_id': f'r_{i}'}
            arm = eval(compile(ast.Expression(expr), '<reparto>', 'eval'),
                       {'zlib': zlib, 'req': req, 'arms': [0,1,2]})
            self.assertEqual(arm, zlib.crc32(req['request_id'].encode()) % 3)
            counts[arm] += 1
        self.assertTrue(all(2700 < n < 3300 for n in counts), counts)


if __name__ == '__main__':
    unittest.main()
