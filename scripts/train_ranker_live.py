"""Reentrena con feedback real, sin escribir en RawTree.

Ejecutar cada hora (en proceso aparte del servidor):
 /Users/joanfernandezmir/Desktop/DATA/rawtree/.venv/bin/python scripts/train_ranker_live.py
Resultados y análisis: data/ranker_live_fb_report.json; modelo: data/ranker_live_fb.txt.
Duración medida: 216 s con 17.687 peticiones vivas y 180 árboles (crece con los datos).
El servidor carga el modelo al arrancar; reentrenar no recarga un servidor existente.
Las features_live son una foto actual: la separación de etiquetas es temporal, pero las
features agregadas no son point-in-time. Las métricas solo comparan los diez expuestos;
no estiman el efecto causal de cambiar candidatos ni corrigen la propensión desconocida.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from reco import features
from reco.client import query
from reco.config import DATA_DIR, TEAM
from reco.ranker import FEATURES, Featurizer
from scripts.train_ranker import build_rows


def pages(sql):
    """Paginación con orden estable; nunca más de 100.000 filas por consulta."""
    rows = []
    offset = 0
    while True:
        chunk = query(f'{sql} LIMIT 100000 OFFSET {offset}', timeout=900)
        rows.extend(chunk)
        if len(chunk) < 100000:
            return pd.DataFrame(rows)
        offset += len(chunk)


def live_sessions():
    # El máximo fija la foto mientras llegan nuevas peticiones.
    end = query(f"SELECT max(ts::DateTime64(3)) AS ts FROM {TEAM}.feedback")[0]['ts']
    df = pages(f"""SELECT f.request_id::String AS request_id, r.ts::DateTime64(3) AS ts,
        r.user_id::String AS user_id, r.is_new_user::Bool AS is_new_user,
        r.context.device::String AS device, r.context.region::String AS region,
        r.context.page::String AS page, r.recent_views::Array(JSON) AS recent_views,
        f.shown::Array(String) AS shown,
        f.clicked::String AS clicked, f.purchased::String AS purchased
        FROM {TEAM}.feedback f INNER JOIN shop.requests r
        ON f.request_id::String = r.request_id::String
        WHERE f.served_by::String = 'team' AND f.request_id::String != '__init__'
        AND f.ts::DateTime64(3) <= '{end}' ORDER BY ts, request_id""")
    df = df.drop_duplicates('request_id').copy()
    df['context'] = df[['device', 'region', 'page']].to_dict('records')
    df['ts'] = pd.to_datetime(df.ts, utc=True)
    return df.sort_values(['ts', 'request_id']).reset_index(drop=True)


def matrix(fz, df):
    xs, buy, click, groups, meta = [], [], [], [], []
    skipped = 0
    for r in df.to_dict('records'):
        shown = r['shown']
        if len(shown) != 10 or any(p not in fz.idx for p in shown):
            skipped += 1
            continue
        cand = np.array([fz.idx[p] for p in shown])
        user = None if r['is_new_user'] else r['user_id']
        x = fz.matrix(r, user, cand, np.arange(10))
        xs.append(x)
        buy.extend(p == r['purchased'] for p in shown)
        click.extend(p == r['clicked'] for p in shown)
        groups.append(10)
        cats = {fz.cat_name.get(v['product_id']) for v in (r['recent_views'] or [])}
        for slot, p in enumerate(shown):
            meta.append((r['user_id'], p, r['ts'], slot, fz.cat_name[p] in cats))
    print(f'Impresiones: {len(buy):,}; peticiones omitidas por catálogo/formato: {skipped}', flush=True)
    return np.vstack(xs), np.array(buy, int), np.array(click, int), groups, meta


def metrics(score, buy, click, groups):
    out = {}
    for name, y in [('compra', buy), ('clic', click)]:
        out['auc_' + name] = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None
        rr, nd = [], []
        start = 0
        for n in groups:
            lab = y[start:start+n]
            if lab.sum():
                order = np.argsort(-score[start:start+n], kind='stable')
                rank = int(np.flatnonzero(lab[order])[0]) + 1
                rr.append(1 / rank)
                nd.append(1 / np.log2(rank+1))
            start += n
        out['mrr_' + name] = float(np.mean(rr)) if rr else None
        out['ndcg_' + name] = float(np.mean(nd)) if nd else None
        out['n_' + name] = len(rr)
    return out


def rebuy_analysis(fz, meta, buy):
    """Cuenta SOLO compras previas a cada impresión, separando origen."""
    start = min(m[2] for m in meta).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    hist = pages(f"""SELECT user_id, product_id, source, count() AS n FROM (
        SELECT user_id::String AS user_id, product_id::String AS product_id, 'organico' AS source
        FROM shop.history_events WHERE event_type::String = 'purchase' AND ts::DateTime64(3) < '{start}'
        UNION ALL SELECT user_id::String, purchased::String, 'carrusel' FROM shop.history_sessions
        WHERE purchased::String != '' AND ts::DateTime64(3) < '{start}')
        GROUP BY user_id, product_id, source ORDER BY user_id, product_id, source""")
    counts = defaultdict(lambda: [0, 0])
    for r in hist.to_dict('records'):
        counts[r['user_id'], r['product_id']][r['source'] == 'carrusel'] += int(r['n'])
    events = pages("""SELECT user_id::String AS user_id, product_id::String AS product_id,
        ts::DateTime64(3) AS ts FROM shop.events WHERE event_type::String = 'purchase'
        ORDER BY ts, user_id, product_id""")
    # Incluye fallback: también son compras reales del usuario.
    carousel = pages(f"""SELECT r.user_id::String AS user_id, f.purchased::String AS product_id,
        f.ts::DateTime64(3) AS ts FROM {TEAM}.feedback f INNER JOIN shop.requests r
        ON f.request_id::String = r.request_id::String WHERE f.purchased::String != ''
        AND f.request_id::String != '__init__' ORDER BY ts, user_id, product_id""")
    ev = []
    for source, df in enumerate((events, carousel)):
        for r in df.to_dict('records'):
            ev.append((pd.to_datetime(r['ts'], utc=True), r['user_id'], r['product_id'], source))
    ev.sort()
    j = 0
    rows = []
    for (u, p, ts, slot, cat), y in zip(meta, buy):
        while j < len(ev) and ev[j][0] < ts:
            _, eu, ep, source = ev[j]
            counts[eu, ep][source] += 1
            j += 1
        a, b = counts[u, p]
        rows.append((str(min(a+b, 2)) + ('+' if a+b >= 2 else ''), b > 0, slot+1, cat, int(y)))
    d = pd.DataFrame(rows, columns=['previas', 'carrusel_previo', 'slot', 'categoria_vista', 'compra'])
    detail = d.groupby(['previas', 'carrusel_previo', 'slot', 'categoria_vista']).compra.agg(['count', 'sum', 'mean']).reset_index()
    # Estandarización por slot y categoría: misma distribución de estratos para cada grupo.
    weights = d.groupby(['slot', 'categoria_vista']).size() / len(d)
    summary = []
    for key, g in d.groupby(['previas', 'carrusel_previo']):
        rates = g.groupby(['slot', 'categoria_vista']).compra.mean()
        common = weights.index.intersection(rates.index)
        summary.append(dict(previas=key[0], carrusel_previo=bool(key[1]), impresiones=len(g),
            compras=int(g.compra.sum()), conversion=float(g.compra.mean()),
            conversion_estandarizada=float((rates.loc[common]*weights.loc[common]).sum()/weights.loc[common].sum()),
            cobertura_estratos=float(weights.loc[common].sum())))
    detail.to_csv(DATA_DIR / 'ranker_live_fb_rebuy.csv', index=False)
    return summary


def train(x, y, groups, weights, objective, rounds):
    params = dict(objective=objective, learning_rate=.04, num_leaves=15, min_data_in_leaf=150,
                  lambda_l2=5, verbosity=-1, num_threads=2, seed=17, feature_fraction=.9)
    if objective == 'lambdarank':
        params.update(label_gain=[0, 1, 3, 7], lambdarank_truncation_level=10)
    ds = lgb.Dataset(x, label=y, group=groups, weight=weights, feature_name=FEATURES,
                     categorical_feature=['page', 'device'])
    return lgb.train(params, ds, num_boost_round=rounds)


def main():
    t = time.monotonic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rounds', type=int, default=180)
    ap.add_argument('--live-weight', type=float, default=10)
    ap.add_argument('--valid-share', type=float, default=.25)
    args = ap.parse_args()
    if not 0 < args.valid_share < 1 or args.live_weight <= 0 or args.rounds < 1:
        ap.error('Parámetros fuera de rango')
    df = live_sessions()
    if len(df) < 100:
        raise RuntimeError('Feedback insuficiente')
    cut = df.ts.iloc[int(len(df)*(1-args.valid_share))]
    fz = Featurizer(features.load())
    xt, bt, ct, gt, mt = matrix(fz, df[df.ts < cut])
    xv, bv, cv, gv, mv = matrix(fz, df[df.ts >= cut])
    if bt.sum() < 10 or bv.sum() < 10:
        raise RuntimeError('Menos de 10 compras en entrenamiento o validación')
    report = dict(corte=str(cut), inicio=str(df.ts.min()), fin=str(df.ts.max()),
                  train_impresiones=len(bt), valid_impresiones=len(bv), train_compras=int(bt.sum()),
                  valid_compras=int(bv.sum()), valid_clics=int(cv.sum()), resultados={})
    results = report['resultados']
    xzero = xv.copy()
    xzero[:, FEATURES.index('pos')] = 0
    def evaluate(name, model=None, score=None):
        if score is None:
            results[name] = dict(observada=metrics(model.predict(xv, num_threads=1), bv, cv, gv),
                                 pos0=metrics(model.predict(xzero, num_threads=1), bv, cv, gv))
        else:
            results[name] = dict(observada=metrics(score, bv, cv, gv), pos0=metrics(score, bv, cv, gv))
        print(name, json.dumps(results[name]), flush=True)
    baseline = lgb.Booster(model_file=str(DATA_DIR / 'ranker_live.txt'))
    evaluate('actual', baseline)
    evaluate('actual_politica', score=baseline.predict(xzero, num_threads=1)+(xzero[:, FEATURES.index('same_cat_last')]==1))
    evaluate('popularidad', score=xv[:, FEATURES.index('pop_7d')])
    evaluate('orden_mostrado', score=-xv[:, FEATURES.index('pos')])
    # Histórico con features previas a la semana, sin usar etiquetas vivas de validación.
    hfz = Featurizer(features.load(until='2026-11-07'))
    cache = DATA_DIR / 'clicked_sessions_all.pkl'
    if cache.exists():
        hist = pd.read_pickle(cache).copy()
    else:
        hist = pages("""SELECT request_id::String AS request_id, ts::DateTime64(3) AS ts,
            user_id::String AS user_id, context.device::String AS device,
            context.region::String AS region, context.page::String AS page,
            recent_views::Array(JSON) AS recent_views, shown::Array(String) AS shown,
            clicked::String AS clicked, purchased::String AS purchased
            FROM shop.history_sessions WHERE clicked::String != ''
            AND ts::DateTime64(3) >= '2026-11-07' AND ts::DateTime64(3) < '2026-11-14'
            ORDER BY ts, request_id""")
        hist['context'] = hist[['device', 'region', 'page']].to_dict('records')
    hist['ts'] = pd.to_datetime(hist.ts, utc=True)
    hist = hist[(hist.ts >= '2026-11-07') & (hist.ts < '2026-11-14')]
    hx, hy, hg = build_rows(hfz, hist, np.random.default_rng(17), .3)
    hy = np.where(hy == 2, 3, hy)
    specs = {}
    for mix in (False, True):
        x = np.vstack([hx, xt]) if mix else xt
        g = hg+gt if mix else gt
        w = np.r_[np.ones(len(hy)), np.full(len(bt), args.live_weight)] if mix else np.ones(len(bt))
        for objective in ('binary', 'regression', 'lambdarank'):
            ly = bt if objective == 'binary' else np.where(bt, 3, ct)
            hist_y = (hy == 3).astype(int) if objective == 'binary' else hy
            y = np.r_[hist_y, ly] if mix else ly
            name = ('mixto_' if mix else 'vivo_')+objective
            model = train(x, y, g, w, objective, args.rounds)
            evaluate(name, model)
            specs[name] = (mix, objective)
    best = max(specs, key=lambda k: results[k]['pos0']['ndcg_compra'])
    report['mejor'] = best
    report['mejora_ndcg_vs_actual'] = results[best]['pos0']['ndcg_compra']-results['actual']['pos0']['ndcg_compra']
    print('Analizando compras previas por instante, origen, slot y categoría…', flush=True)
    report['recompra'] = rebuy_analysis(fz, mt+mv, np.r_[bt, bv])
    mix, objective = specs[best]
    xall = np.vstack([xt, xv])
    ball, call = np.r_[bt, bv], np.r_[ct, cv]
    yall = ball if objective == 'binary' else np.where(ball, 3, call)
    if mix:
        xall = np.vstack([hx, xall])
        yall = np.r_[(hy == 3).astype(int) if objective == 'binary' else hy, yall]
    weights = np.r_[np.ones(len(hy)), np.full(len(ball), args.live_weight)] if mix else np.ones(len(ball))
    model = train(xall, yall, hg+gt+gv if mix else gt+gv, weights, objective, args.rounds)
    out = DATA_DIR / 'ranker_live_fb.txt'
    temp = out.with_suffix('.tmp')
    model.save_model(str(temp))
    temp.replace(out)
    report['duracion_s'] = time.monotonic()-t
    report['parametros'] = vars(args)
    (DATA_DIR / 'ranker_live_fb_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
