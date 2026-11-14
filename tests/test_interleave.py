"""Mezcla, degradación segura y atribución verificable sin red."""
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from reco.policies.interleave import Interleave, mezclar
from scripts.interleave_report import resumir, estadistica


class Estado:
    def is_valid(self, p):
        return isinstance(p, str) and p != 'invalido'

    def fallback(self):
        return [f'f{i}' for i in range(10)]


class Politica:
    name = 'prueba'
    refresh_every = 60

    def __init__(self, lista=None):
        self.lista = lista

    def recommend(self, req, state):
        if self.lista is None:
            raise RuntimeError('fallo simulado')
        return self.lista


class TestInterleave(unittest.TestCase):
    def setUp(self):
        self.a = [f'a{i}' for i in range(10)]
        self.b = [f'b{i}' for i in range(10)]

    def test_distintos_equilibrio_determinismo(self):
        for i in range(100):
            salida, autores = mezclar(self.a, self.b, str(i))
            self.assertEqual(len(set(salida)), 10)
            self.assertEqual(autores.count('A'), 5)
            self.assertEqual(autores.count('B'), 5)
            self.assertEqual((salida, autores), mezclar(self.a, self.b, str(i)))
        self.assertGreater(len({tuple(mezclar(self.a, self.b, str(i))[0]) for i in range(100)}), 1)

    def test_prefijo_y_solapamiento(self):
        for n in range(11):
            b = self.a[:n] + self.b[n:]
            out, owners = mezclar(self.a, b, 'r')
            self.assertEqual(out[:n], self.a[:n])
            self.assertEqual(owners[:n], ['ambas']*n)
            self.assertLessEqual(abs(owners.count('A')-owners.count('B')), 1)
        out, _ = mezclar(self.a, self.a[::-1], 'r')
        self.assertEqual(len(set(out)), 10)

    def test_fallos_sin_registro(self):
        class Registro:
            def put(self, r):
                raise AssertionError('No registrar fallback')
        for mala in (None, self.a[:9], ['invalido']+self.a[:9], self.a[:9]+[self.a[0]]):
            for a, b in ((mala, self.b), (self.b, mala)):
                p = Interleave(Politica(a), Politica(b))
                p.registro = Registro()
                self.assertEqual(p.recommend({'request_id': 'r'}, Estado()), self.b)
        p = Interleave(Politica(), Politica())
        self.assertEqual(p.recommend({}, Estado()), Estado().fallback())

    def test_refresh_independiente(self):
        a, b = Politica(self.a), Politica(self.b)
        a.refresh_every, b.refresh_every = 600, 120
        llamadas = []
        a.refresh = lambda state: llamadas.append('A')
        b.refresh = lambda state: llamadas.append('B')
        p = Interleave(a, b)
        with patch('reco.policies.interleave.time.monotonic', return_value=1000):
            p.refresh(Estado())
        with patch('reco.policies.interleave.time.monotonic', return_value=1120):
            p.refresh(Estado())
        self.assertEqual(llamadas, ['A', 'B', 'B'])
        p._refreshed = [float('-inf'), float('-inf')]
        def fallo(state):
            raise RuntimeError('refresh A')
        a.refresh = fallo
        with self.assertRaises(RuntimeError):
            p.refresh(Estado())
        self.assertEqual(llamadas[-1], 'B')

    def test_registro_local(self):
        from reco.policies.interleave import _Registro
        with tempfile.TemporaryDirectory() as carpeta:
            with patch('reco.policies.interleave.LOGS_DIR', Path(carpeta)), patch.dict(
                    'os.environ', {'RECO_INTERLEAVE_REMOTE': '0'}):
                registro = _Registro()
                p = Interleave(Politica(self.a), Politica(self.b))
                p.registro = registro
                out = p.recommend({'request_id': 'r'}, Estado())
                registro.close()
                row = json.loads((Path(carpeta)/'interleave.jsonl').read_text())
                self.assertEqual(row['shown'], out)
                self.assertEqual(len(row['owners']), 10)
                self.assertFalse(registro.remoto)

    def test_atribucion(self):
        r = dict(request_id='r', shown=self.a, owners=['ambas', 'A', 'B']+['A']*7,
                 policy_a='a', policy_b='b', is_new_user=True, home=True)
        f = dict(shown=self.a, served_by='team', clicked='a1', purchased='a2')
        n, out = resumir([r], {'r': [f]})
        self.assertEqual(n, 1)
        self.assertEqual(out[0]['clic']['A'], 1)
        self.assertEqual(out[0]['compra']['B'], 1)
        self.assertEqual(resumir([r], {'r': [{**f, 'shown': self.b}]})[0], 0)
        self.assertEqual(resumir([r, r], {'r': [f]})[0], 0)
        self.assertEqual(resumir([{**r, 'dry_run': True}], {'r': [f]})[0], 0)
        f['clicked'] = 'a0'
        self.assertEqual(resumir([r], {'r': [f]})[1][0]['clic']['A'], 0)
        self.assertAlmostEqual(estadistica(5, 5)['p'], 1)


if __name__ == '__main__':
    unittest.main()
