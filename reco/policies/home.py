"""Home: popularidad global de los últimos siete días; el resto se delega en personal_v3.

Las variantes son para el análisis, no requieren SQL durante recommend().
El refresco reutiliza el índice publicado por Personal en su subproceso habitual.
"""
from __future__ import annotations
from collections import defaultdict
from .base import Policy
from .personal import PersonalV3
from ..ranker import _views


class HomeBest(Policy):
    name = 'home_best'
    refresh_every = PersonalV3.refresh_every

    def __init__(self, until=None, interna=None, estrategia='global'):
        super().__init__(until)
        self.interna = interna if interna is not None else PersonalV3(until)
        self.estrategia = estrategia

    def fit(self, state):
        self.interna.fit(state)
        # Personal comparte índices; no descargamos ni duplicamos las features en refresh.
        self.perfil = PersonalV3(self.until)
        self.perfil.fit(state)
        f = self.load_features()
        uc = f['user_cat'].sort_values('w', ascending=False)
        self.favoritas = uc.drop_duplicates('user_id').set_index('user_id').category.to_dict()

    def refresh(self, state):
        self.interna.refresh(state)
        if self.interna is not self.perfil and not isinstance(self.interna, PersonalV3):
            self.perfil.refresh(state)

    def recommend(self, req, state):
        if _views(req):
            return self.interna.recommend(req, state)
        ix = self.perfil.ix
        ctx = req.get('context') or {}
        pop = dict(ix.top_pop)
        contextual = dict(ix.ctx_pop.get((ctx.get('device'), ctx.get('region')), ()))
        modo = self.estrategia
        if modo == 'global' or not contextual:
            score = dict(pop)
        elif modo == 'mixta':
            score = {p: .4*pop.get(p,0)+.3*contextual.get(p,0) for p in pop.keys() | contextual.keys()}
        else:
            score = {p: n+.05*pop.get(p,0) for p,n in contextual.items()}
            for p,n in pop.items():
                score.setdefault(p,.05*n)
        if modo in ('favorita','una_categoria'):
            user = None if req.get('is_new_user') else req.get('user_id')
            cat = self.favoritas.get(user) if modo == 'favorita' else None
            if cat is None:
                pesos = defaultdict(float)
                for p,n in contextual.items():
                    pesos[ix.cat_of.get(p)] += n
                cat = max(pesos,key=pesos.get) if pesos else ix.cat_of.get(next(iter(pop),None))
            score = {p:n for p,n in ix.cat_top.get(cat,())}
        orden = sorted(score,key=lambda p:(-score[p],p))
        if modo == 'diversa':
            elegidos, resto, cuenta = [],[],defaultdict(int)
            for p in orden:
                c = ix.cat_of.get(p)
                if cuenta[c] < 2:
                    elegidos.append(p)
                    cuenta[c] += 1
                else:
                    resto.append(p)
            orden = elegidos+resto
        # Reserva suficiente para que cambios de stock no vacíen los primeros diez.
        return [p for p in orden if state.is_valid(p)][:10]
