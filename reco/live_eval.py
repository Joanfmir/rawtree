"""Modelo directo de recompensa ajustado exclusivamente a impresiones del feedback vivo.

No es una estimación causal ni IPS: faltan propensiones y soporte para muchos productos.
La extrapolación a productos/slots nunca mostrados y a listas distintas puede fallar;
no modela competencia entre productos ni cambios futuros inducidos por las compras.
La suma de probabilidades marginales estima compras por petición, no una simulación.
La validación temporal usa un modelo congelado y SOLO feedback posterior al corte.
Las comparaciones de versiones son observacionales (horas y públicos distintos), con
ruido binomial; un orden diferente no demuestra que una política sea peor.

Las covariables parten de features_until_2026-11-07, sin reconstruir la semana
intermedia, y reproducen eventos, catálogo y compras de feedback vivos en orden.
La base histórica puede incluir compras de history_sessions como covariable, nunca
como etiqueta de recompensa. La afinidad base se mantiene fija; las señales vivas
se acumulan. No se usan IDs ni puntuaciones del ranker en el modelo de recompensa.
La evaluación de políticas usa features y catálogo ACTUALES: es una reproducción
sobre contextos recientes, no un backtest del estado exacto de producción entonces.
Ejecutar en proceso independiente del servidor; nunca importa ni arranca serve.
Solo SELECT remotos; los informes se guardan opcionalmente en disco local.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from . import catalog, features
from .client import query
from .config import TEAM, check
from .policies import POLICIES
from .safety import finalize
from .state import State

CATEGORICAS = ['relacion', 'pagina', 'dispositivo']
COLUMNAS = ['slot', 'relacion', 'n_vistas', 'vistas_producto', 'pagina', 'dispositivo',
            'nuevo', 'popularidad', 'buy_rate', 'precio', 'compras_previas',
            'compras_carrusel', 'afinidad_categoria']


def fecha(value):
    """Timestamp UTC homogéneo, con precisión de nanosegundos."""
    return pd.Timestamp(value).tz_localize('UTC') if pd.Timestamp(value).tzinfo is None else pd.Timestamp(value).tz_convert('UTC')


def literal(value):
    """Literal SQL para timestamps generados internamente."""
    return fecha(value).strftime('%Y-%m-%d %H:%M:%S.%f')


def paginas(sql, tam=100000):
    """Lecturas acotadas; el SQL debe tener un ORDER BY determinista."""
    rows, offset = [], 0
    while True:
        batch = query(f'{sql} LIMIT {tam} OFFSET {offset}', timeout=180)
        rows.extend(batch)
        if len(batch) < tam:
            return rows
        offset += tam


def cargar(last, train_last):
    """Congela una ventana de feedback maduro y carga todas sus peticiones."""
    feedback = query(f"""SELECT request_id::String AS request_id, ts::String AS feedback_ts,
        shown::Array(String) AS shown, clicked::String AS clicked, purchased::String AS purchased
        FROM {TEAM}.feedback WHERE request_id::String != '__init__' AND served_by::String = 'team'
        ORDER BY ts::DateTime64(3) DESC, request_id::String DESC LIMIT {train_last}""", timeout=180)
    if not feedback:
        raise ValueError('No hay feedback servido por el equipo')
    fb = pd.DataFrame(feedback).sort_values('feedback_ts').drop_duplicates('request_id', keep='last')
    lo, hi = fb.feedback_ts.min(), fb.feedback_ts.max()
    # Margen: feedback llega segundos después; no mezclamos peticiones sin resultado maduro.
    start = fecha(lo) - pd.Timedelta(minutes=5)
    reqs = paginas(f"""SELECT request_id::String AS request_id, ts::String AS ts,
        user_id::String AS user_id, is_new_user::Bool AS is_new_user, context, recent_views
        FROM shop.requests WHERE ts::DateTime64(3) >= '{literal(start)}'
        AND ts::DateTime64(3) <= '{literal(hi)}'
        ORDER BY ts::DateTime64(3), request_id::String""")
    requests = pd.DataFrame(reqs).drop_duplicates('request_id')
    df = requests.merge(fb, on='request_id', validate='one_to_one').sort_values(['ts', 'request_id']).reset_index(drop=True)
    df = df[df.shown.map(lambda x: len(x) == 10 and len(set(x)) == 10)].reset_index(drop=True)
    if len(df) < 100:
        raise ValueError(f'Solo {len(df)} peticiones válidas; insuficientes para validación temporal')
    # La etiqueta de versión solo cuenta si los items coinciden con lo realmente mostrado.
    submissions = paginas(f"""SELECT s.request_id::String AS request_id, s.items::Array(String) AS items,
        s.model_version::String AS model_version FROM {TEAM}.submissions AS s
        INNER JOIN shop.requests AS r ON r.request_id::String = s.request_id::String
        WHERE r.ts::DateTime64(3) >= '{literal(start)}' AND r.ts::DateTime64(3) <= '{literal(hi)}'
        ORDER BY s.request_id::String, s.model_version::String, toString(s.items)""")
    shown = dict(zip(df.request_id, df.shown.map(tuple)))
    versions = defaultdict(set)
    for row in submissions:
        if tuple(row['items']) == shown.get(row['request_id']):
            versions[row['request_id']].add(row['model_version'])
    df['version'] = [next(iter(versions[r])) if len(versions[r]) == 1 else 'ambigua/sin_match' for r in df.request_id]
    target = pd.DataFrame(query(f"""SELECT request_id::String AS request_id, ts::String AS ts,
        user_id::String AS user_id, is_new_user::Bool AS is_new_user, context, recent_views
        FROM shop.requests WHERE ts::DateTime64(3) <= '{literal(df.ts.max())}'
        ORDER BY ts::DateTime64(3) DESC, request_id::String DESC LIMIT {last}""", timeout=180))
    target = target.drop_duplicates('request_id').sort_values(['ts', 'request_id']).reset_index(drop=True)
    if len(target) < last:
        print(f'Aviso: solo {len(target)} peticiones disponibles en la ventana para --last {last}', flush=True)
    return df, target


class Covariables:
    """Índices propios de recompensa, independientes del Featurizer del ranker."""

    def __init__(self):
        base = features.load(until='2026-11-07')
        self.cat = {r['product_id']: r for r in query(catalog.STATIC_CATALOG, timeout=60)}
        self.items = {r['product_id']: r for r in base['item_features'].fillna(0).to_dict('records')}
        self.bought = {}
        for r in base['user_purchases'].to_dict('records'):
            self.bought[r['user_id'], r['product_id']] = [float(r['n']), float(r.get('n_carousel', 0))]
        self.affinity = {}
        uc = base['user_cat'].copy()
        uc['a'] = uc.w / uc.groupby('user_id').w.transform('sum').clip(lower=1)
        self.affinity = {(r.user_id, r.category): float(r.a) for r in uc.itertuples()}
        self.live_aff = Counter()
        self.live_total = Counter()

    def actualizar(self, kind, r):
        if kind == 'catalogo':
            self.cat[r['product_id']] = r
            return
        pid, uid = r['product_id'], r['user_id']
        categoria = self.cat.get(pid, {}).get('category', '')
        purchase = kind == 'carrusel' or r.get('event_type') == 'purchase'
        if purchase:
            counts = self.bought.setdefault((uid, pid), [0., 0.])
            counts[0] += 1
            counts[1] += kind == 'carrusel'
        if kind == 'organico':
            item = self.items.setdefault(pid, {})
            key = 'purchases' if purchase else 'views' if r['event_type'] == 'view' else 'carts'
            item[key] = item.get(key, 0) + 1
        weight = 5 if purchase else 3 if r.get('event_type') == 'add_to_cart' else 1
        self.live_aff[uid, categoria] += weight
        self.live_total[uid] += weight

    def matriz(self, req, products):
        views = req.get('recent_views') or []
        counts = Counter(v['product_id'] for v in views)
        cats = {self.cat.get(p, {}).get('category') for p in counts}
        uid = req.get('user_id')
        ctx = req.get('context') or {}
        out = []
        for slot, pid in enumerate(products, 1):
            cat = self.cat.get(pid, {})
            category = cat.get('category', '')
            item = self.items.get(pid, {})
            bought = self.bought.get((uid, pid), [0, 0])
            # Mismo cálculo en entrenamiento y en políticas; no hay etiquetas futuras.
            pop = float(item.get('purchases', 0))
            rate = (pop + 1) / (float(item.get('views', 0)) + 50)
            affinity = (self.affinity.get((uid, category), 0) * 20 + self.live_aff[uid, category]) / (20 + self.live_total[uid])
            out.append([slot, 'visto' if pid in counts else 'categoria' if category in cats else 'otra',
                        len(views), counts[pid], ctx.get('page', 'desconocida'), ctx.get('device', 'desconocido'),
                        int(bool(req.get('is_new_user'))), np.log1p(pop), rate,
                        np.log1p(max(float(cat.get('price_eur', 0)), 0)), *bought, affinity])
        return out


def cronologia(df):
    """Eventos conocidos antes de cada petición; feedback disponible según su ts de llegada."""
    hi = literal(df.ts.max())
    ev = paginas(f"""SELECT ts::String AS ts, product_id::String AS product_id,
        user_id::String AS user_id, event_type::String AS event_type FROM shop.events
        WHERE ts::DateTime64(3) <= '{hi}' ORDER BY ts::DateTime64(3), user_id::String, product_id::String, event_type::String""")
    updates = paginas(f"""SELECT ts::String AS ts, product_id::String AS product_id,
        category::String AS category, price_eur::Float64 AS price_eur FROM shop.catalog_updates
        WHERE ts::DateTime64(3) <= '{hi}' ORDER BY ts::DateTime64(3), product_id::String""")
    # Incluye compras anteriores a la ventana seleccionada, también de fallback.
    buys = paginas(f"""SELECT f.ts::String AS ts, r.user_id::String AS user_id,
        f.purchased::String AS product_id FROM {TEAM}.feedback AS f INNER JOIN shop.requests AS r
        ON r.request_id::String = f.request_id::String WHERE f.request_id::String != '__init__'
        AND f.purchased::String != '' AND f.ts::DateTime64(3) <= '{hi}'
        ORDER BY f.ts::DateTime64(3), f.request_id::String""")
    timeline = [(fecha(r['ts']), kind, r) for kind, rows in [('organico', ev), ('catalogo', updates), ('carrusel', buys)] for r in rows]
    return sorted(timeline, key=lambda x: x[0])


def marco(rows):
    frame = pd.DataFrame(rows, columns=COLUMNAS)
    for c in CATEGORICAS:
        vocab = {'relacion': ['visto', 'categoria', 'otra'], 'pagina': ['home', 'product', 'category', 'desconocida'],
                 'dispositivo': ['web', 'ios', 'android', 'desconocido']}[c]
        frame[c] = pd.Categorical(frame[c], categories=vocab)
    return frame


def preparar(df, target, policies, state):
    """Una pasada temporal; genera candidatos por recommend + finalize sin refrescos en hilos."""
    cov = Covariables()
    timeline = cronologia(df)
    print(f'Cronología cargada: {len(timeline)} eventos; generando impresiones y recomendaciones...', flush=True)
    ix, rows, proposed = 0, [], {name: [] for name in policies}
    wanted = set(target.request_id)
    supports = {name: [0, 0] for name in policies}
    exposed = set()
    logged = set(df.request_id)
    combined = pd.concat([df, target[~target.request_id.isin(logged)]], ignore_index=True)
    combined = combined.sort_values(['ts', 'request_id'])
    for number, req in enumerate(combined.to_dict('records')):
        if number and number % 5000 == 0:
            print(f'Peticiones procesadas: {number}/{len(combined)}', flush=True)
        ts = fecha(req['ts'])
        while ix < len(timeline) and timeline[ix][0] < ts:
            _, kind, row = timeline[ix]
            cov.actualizar(kind, row)
            ix += 1
        if req['request_id'] in logged:
            rows.extend(cov.matriz(req, req['shown']))
        if req['request_id'] in wanted:
            for name, policy in policies.items():
                recs = finalize(policy.recommend(req, state), state)
                if len(recs) != 10:
                    raise ValueError(f'{name}: finalize devolvió {len(recs)} productos')
                proposed[name].extend(cov.matriz(req, recs))
                supports[name][0] += sum(p not in exposed for p in recs)
                supports[name][1] += 10
        if req['request_id'] in logged:
            exposed.update(req['shown'])
    return marco(rows), {k: marco(v) for k, v in proposed.items()}, supports


def ajustar(X, y):
    if len(np.unique(y)) < 2:
        raise ValueError('El tramo de entrenamiento no contiene positivos y negativos')
    model = lgb.LGBMClassifier(n_estimators=180, learning_rate=.04, num_leaves=12,
                              min_child_samples=400, reg_lambda=10, verbosity=-1,
                              n_jobs=2, random_state=17, deterministic=True, force_col_wise=True)
    model.fit(X, y)
    return model


def diagnostico(y, p):
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float('nan')
    metrics = {'n': len(y), 'positivos': int(y.sum()), 'auc': auc,
               'brier': brier_score_loss(y, p), 'logloss': log_loss(y, p, labels=[0, 1]),
               'observado': float(y.mean()), 'predicho': float(p.mean())}
    d = pd.DataFrame({'real': y, 'pred': p})
    d['decil'] = pd.qcut(d.pred.rank(method='first'), 10, labels=False) + 1
    return metrics, d.groupby('decil').agg(n=('real', 'size'), positivos=('real', 'sum'), observado=('real', 'mean'), predicho=('pred', 'mean'))


def comparar(df, pred):
    d = df.copy()
    d['pred'] = pred.reshape(-1, 10).sum(axis=1)
    d['real'] = d.purchased.fillna('').ne('').astype(int)
    result = d.groupby('version').agg(n=('real', 'size'), compras=('real', 'sum'), real_100=('real', 'mean'), pred_100=('pred', 'mean'))
    result[['real_100', 'pred_100']] *= 100
    result['error_pp'] = result.pred_100 - result.real_100
    result['se_real_pp'] = 100 * np.sqrt((result.real_100 / 100) * (1 - result.real_100 / 100) / result.n)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--policies', nargs='+', choices=sorted(POLICIES), default=['personal_v1', 'personal', 'gulias'])
    ap.add_argument('--last', type=int, default=20000)
    ap.add_argument('--train-last', type=int, default=100000, help='Máximo de peticiones con feedback para ajustar/validar; <= 500000')
    ap.add_argument('--cut', help='Corte UTC; por defecto percentil temporal 70%%')
    ap.add_argument('--output', help='Guardar informe JSON local')
    args = ap.parse_args()
    if not 100 <= args.train_last <= 500000 or not 1 <= args.last <= 500000:
        ap.error('--train-last debe estar entre 100 y 500000; --last entre 1 y 500000')
    check()
    t = time.monotonic()
    df, target = cargar(args.last, args.train_last)
    cut = fecha(args.cut) if args.cut else fecha(df.iloc[int(len(df) * .7)].ts)
    train = np.array([fecha(ts) < cut for ts in df.ts])
    if not train.any() or train.all():
        ap.error('El corte debe dejar peticiones a ambos lados')
    print(f'Feedback válido: {len(df)}; entrenamiento {train.sum()}, validación {(~train).sum()}; corte UTC {cut}', flush=True)
    print('Cargando estado actual y políticas (sin serve ni escrituras remotas)...', flush=True)
    # Exigir features existentes evita regenerarlas o alterar los artefactos compartidos.
    if not features.folder(None).exists():
        raise FileNotFoundError('Faltan data/features_live; ejecutar build_features.py por separado')
    state = State()
    state.refresh()
    print(f'Estado: {len(state.catalog)} productos, {len(state.unavailable)} sin stock', flush=True)
    policies = {name: POLICIES[name]() for name in args.policies}
    for policy in policies.values():
        policy.fit(state)
        # El refresh de gulias también actualiza compras; ejecutar solo la parte síncrona.
        if hasattr(policy, '_add_live_purchases'):
            policy._add_live_purchases()
        print(f'Política cargada: {policy.name}', flush=True)
    X, candidates, support = preparar(df, target, policies, state)
    print(f'Impresiones preparadas: {len(X)}; peticiones para políticas: {len(target)}', flush=True)
    mask = np.repeat(train, 10)
    labels = {kind: np.array([int(p == (r[kind] or '')) for r in df.to_dict('records') for p in r['shown']]) for kind in ['clicked', 'purchased']}
    report = {'corte_utc': str(cut), 'n': len(df), 'n_peticiones_politicas': len(target),
              'validacion': {}, 'politicas': [], 'soporte': support,
              'ventana_feedback': [df.ts.min(), df.ts.max()],
              'ventana_politicas': [target.ts.min(), target.ts.max()]}
    models = {}
    for kind, y in labels.items():
        model = ajustar(X[mask], y[mask])
        pred = model.predict_proba(X[~mask])[:, 1]
        metrics, deciles = diagnostico(y[~mask], pred)
        report['validacion'][kind] = metrics
        print(f'\nValidación temporal {kind}: {json.dumps(metrics)}')
        print(deciles.to_string(float_format=lambda x: f'{x:.6f}'))
        report['validacion'][kind]['deciles'] = deciles.reset_index().to_dict('records')
        if kind == 'purchased':
            comparison = comparar(df[~train], pred)
            print('\nVersiones en validación (compras/100; error en puntos porcentuales):')
            print(comparison.to_string(float_format=lambda x: f'{x:.4f}'))
            usable = comparison.drop(index='ambigua/sin_match', errors='ignore')
            real_order = usable.sort_values('real_100', ascending=False).index.tolist()
            pred_order = usable.sort_values('pred_100', ascending=False).index.tolist()
            print(f'Orden real: {real_order}\nOrden predicho: {pred_order}\nCoinciden: {real_order == pred_order if len(usable) > 1 else "no comprobable (<2 versiones)"}')
            report['versiones'] = comparison.reset_index().to_dict('records')
            report['orden_real'], report['orden_predicho'] = real_order, pred_order
            # Predicciones sucesivas fuera de muestra: permite cubrir versiones antiguas
            # que ya no aparecen después del corte principal. El primer 10% es arranque.
            rolling = np.full(len(y), np.nan)
            boundaries = np.linspace(int(len(df) * .1), len(df), 5, dtype=int)
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                # Ningún timestamp compartido puede caer a ambos lados del corte.
                before = np.array([fecha(ts) < fecha(df.iloc[left].ts) for ts in df.ts])
                past = np.repeat(before, 10)
                if len(np.unique(y[past])) < 2:
                    continue
                block = ajustar(X[past], y[past])
                rolling[left * 10:right * 10] = block.predict_proba(X.iloc[left * 10:right * 10])[:, 1]
            valid_requests = np.isfinite(rolling.reshape(-1, 10)).all(axis=1)
            progressive = comparar(df[valid_requests], rolling[np.repeat(valid_requests, 10)])
            print('\nVersiones: validación temporal sucesiva (primer 10% excluido):')
            print(progressive.to_string(float_format=lambda x: f'{x:.4f}'))
            usable = progressive.drop(index='ambigua/sin_match', errors='ignore')
            real_order = usable.sort_values('real_100', ascending=False).index.tolist()
            pred_order = usable.sort_values('pred_100', ascending=False).index.tolist()
            print(f'Orden sucesivo real: {real_order}\nOrden sucesivo predicho: {pred_order}\nCoinciden: {real_order == pred_order if len(usable) > 1 else "no comprobable"}')
            report['versiones_sucesivas'] = progressive.reset_index().to_dict('records')
            report['orden_sucesivo_real'], report['orden_sucesivo_predicho'] = real_order, pred_order
        # Reajuste final con todos los resultados maduros, separado del diagnóstico fuera de muestra.
        models[kind] = ajustar(X, y)
    segments = {'todos': np.ones(len(target), dtype=bool), 'conocido': ~target.is_new_user.astype(bool).to_numpy(),
                'nuevo': target.is_new_user.astype(bool).to_numpy(),
                'home': target.context.map(lambda c: (c or {}).get('page') == 'home').to_numpy()}
    for name, candidate in candidates.items():
        predictions = {k: model.predict_proba(candidate)[:, 1].reshape(-1, 10).sum(axis=1) for k, model in models.items()}
        for segment, m in segments.items():
            report['politicas'].append({'politica': name, 'segmento': segment, 'n': int(m.sum()),
                'compras_100': float(predictions['purchased'][m].mean() * 100) if m.any() else None,
                'clics_100': float(predictions['clicked'][m].mean() * 100) if m.any() else None})
        print(f'{name}: productos sin exposición anterior en la ventana = {support[name][0]}/{support[name][1]}')
    print('\nEstimación directa con modelo reajustado (no causal, sin garantía de orden):')
    print(pd.DataFrame(report['politicas']).to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'\nTiempo total: {time.monotonic() - t:.1f}s. No se ha escrito en RawTree.')
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + '\n')


if __name__ == '__main__':
    main()
