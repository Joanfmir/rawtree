"""Evaluación reproducible de transferencia por texto y diagnóstico de usuarios nuevos.

Solo SELECT. Los resultados y los datos de análisis se guardan en data/coldstart_eval/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from reco.coldstart import catalog_text, text_neighbors, inherit, build
from reco.client import query_df
from reco.config import DATA_DIR, TEAM


def evaluate(catalog, items, edges, out):
    rng = np.random.default_rng(17)
    eligible = items.loc[items.views.fillna(0) >= 100, 'product_id']
    targets = set(rng.choice(sorted(set(eligible) & set(catalog.product_id)),
                             min(500, len(eligible)), replace=False))
    donors = set(items.product_id) - targets
    text = text_neighbors(catalog, donors)
    c = catalog.merge(items[['product_id', 'purchases']], on='product_id', how='left').fillna({'purchases': 0})
    alternatives = {}
    for name, key in [('subcat_pop', 'subcategory'), ('brand_pop', 'brand')]:
        rows = []
        for _, g in c.groupby(['category', key]):
            ds = g[g.product_id.isin(donors)].sort_values(['purchases', 'product_id'], ascending=[False, True]).head(10)
            rows.extend((p, q, 1.0) for p in g.product_id if p in targets for q in ds.product_id)
        alternatives[name] = pd.DataFrame(rows, columns=['p', 'q', 'sim'])
    alternatives['text'] = text
    result = {}
    truth = edges[(edges.kind == 'buy') & edges.p.isin(targets) & ~edges.q.isin(targets)]
    truth = truth[truth.p != truth.q].sort_values('n', ascending=False).groupby('p').head(20)
    real = {p: set(g.q) for p, g in truth.groupby('p')}
    for name, nb in alternatives.items():
        pred = inherit(nb, items, edges, targets)
        pp = pred['cold_priors'].merge(items, on='product_id', suffixes=('_pred', '_real'))
        pe = pred['cold_neighbors']
        pe = pe[pe.kind == 'buy'].groupby('p').head(20)
        got = {p: set(g.q) for p, g in pe.groupby('p')}
        recalls = [len(q & got.get(p, set())) / len(q) for p, q in real.items()]
        result[name] = {'recall20': float(np.mean(recalls)), 'targets': len(targets),
                        'truth_targets': len(real), 'prior_coverage': len(pp),
                        **{f'spearman_{col}': float(pp[f'{col}_pred'].corr(pp[f'{col}_real'], method='spearman'))
                           for col in ['purchases', 'buy_rate']}}
    (out / 'mask_metrics.json').write_text(json.dumps(result, indent=2))
    return result


def live(out, catalog):
    # Instantánea temporal común; jamás se descarga el stream completo sin límite.
    end = query_df('SELECT max(ts::DateTime64(3)) AS ts FROM shop.requests').iloc[0].ts
    end = str(end)
    start = str(pd.Timestamp(end) - pd.Timedelta(hours=4))
    r = query_df(f"""SELECT request_id::String AS request_id, user_id::String AS user_id,
        ts::DateTime64(3) AS ts, is_new_user::Bool AS is_new_user,
        context.page::String AS page, context.device::String AS device,
        context.region::String AS region, recent_views::Array(JSON) AS recent_views
        FROM shop.requests WHERE ts::DateTime64(3) >= '{start}'
        AND ts::DateTime64(3) <= '{end}' LIMIT 100000""")
    f = query_df(f"""SELECT request_id::String AS request_id, shown::Array(String) AS shown,
        purchased::String AS purchased, clicked::String AS clicked, served_by::String AS served_by
        FROM {TEAM}.feedback WHERE request_id::String != '__init__'
        AND ts::DateTime64(3) >= '{start}' AND ts::DateTime64(3) <= '{end}' LIMIT 100000""")
    if len(r) == 100000 or len(f) == 100000:
        raise RuntimeError('Ventana truncada: reducir las cuatro horas antes de analizar')
    d = r.merge(f.drop_duplicates('request_id'), on='request_id')
    cat = dict(zip(catalog.product_id, catalog.category))
    def views(v):
        if isinstance(v, str):
            v = json.loads(v)
        return [x.get('product_id') for x in (v or [])]
    d['view_categories'] = d.recent_views.map(lambda v: set(cat.get(p) for p in views(v)) - {None})
    vg = d.assign(view_category=d.view_categories.map(sorted)).explode('view_category')
    vg['purchase_in_category'] = vg.purchased.map(cat).eq(vg.view_category).astype(int)
    vg.groupby(['is_new_user', 'view_category']).agg(requests=('request_id', 'size'),
        purchases_in_category=('purchase_in_category', 'sum')).to_csv(out / 'live_view_category.csv')
    d['buy'] = d.purchased.fillna('').ne('').astype(int)
    d['buy_cat'] = d.purchased.map(cat)
    d['same_view_category'] = [b in cs if b else False for b, cs in zip(d.buy_cat, d.view_categories)]
    tables = {}
    for key in ['page', 'device', 'region', 'buy_cat']:
        g = d.groupby(['is_new_user', key], dropna=False).agg(requests=('buy', 'size'), purchases=('buy', 'sum'))
        g['conversion'] = g.purchases / g.requests
        tables[key] = g.reset_index()
        tables[key].to_csv(out / f'live_{key}.csv', index=False)
    d['has_views'] = d.view_categories.map(bool)
    g = d.groupby(['is_new_user', 'has_views']).agg(requests=('buy', 'size'), purchases=('buy', 'sum'),
                                                               same_category_buys=('same_view_category', 'sum'))
    g.to_csv(out / 'live_summary.csv')
    # Ranking de mañana contra compras orgánicas de la segunda mitad, sin fuga temporal.
    mid = str(pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2)
    e = query_df(f"""SELECT product_id::String AS product_id,
        countIf(event_type::String = 'view' AND ts::DateTime64(3) < '{mid}') AS views,
        countIf(event_type::String = 'purchase' AND ts::DateTime64(3) < '{mid}') AS train_buys,
        countIf(event_type::String = 'purchase' AND ts::DateTime64(3) >= '{mid}') AS test_buys
        FROM (SELECT product_id::String AS product_id, event_type::String AS event_type,
        ts::DateTime64(3) AS ts FROM shop.events) WHERE ts::DateTime64(3) >= '{start}'
        AND ts::DateTime64(3) <= '{end}' GROUP BY product_id""")
    e.to_parquet(out / 'organic_split.parquet', index=False)
    e = catalog[['product_id', 'category']].merge(e, how='left').fillna(0)
    hist = pd.read_parquet(DATA_DIR / 'features_until_2026-11-07/item_features.parquet')
    e = e.merge(hist[['product_id', 'purchases_7d']], how='left').fillna(0)
    e['buy_rate'] = (e.train_buys + 1) / (e.views + 50)
    ranks = {}
    # Recuperación de compras observadas por categoría, no estimación causal del carrusel.
    for name, score in [('popular_pre_07nov', 'purchases_7d'), ('buy_rate', 'buy_rate'), ('trend_live', 'train_buys')]:
        top = e.sort_values(score, ascending=False).groupby('category').head(10)
        ranks[name] = {'organic_purchase_recall10': float(top.test_buys.sum() / max(e.test_buys.sum(), 1))}
    # Comparación por usuarios identificados en requests; no confundir desconocido con conocido.
    segmented = query_df(f"""SELECT e.product_id AS product_id, r.is_new_user AS is_new_user,
        count() AS purchases FROM
        (SELECT product_id::String AS product_id, user_id::String AS user_id,
        ts::DateTime64(3) AS ts FROM shop.events WHERE event_type::String = 'purchase'
        AND ts::DateTime64(3) >= '{mid}' AND ts::DateTime64(3) <= '{end}') AS e
        INNER JOIN (SELECT user_id::String AS user_id, max(is_new_user::Bool) AS is_new_user
        FROM shop.requests WHERE ts::DateTime64(3) >= '{start}'
        AND ts::DateTime64(3) <= '{end}' GROUP BY user_id) AS r ON e.user_id = r.user_id
        GROUP BY e.product_id, r.is_new_user""")
    segmented.to_csv(out / 'organic_by_new_user.csv', index=False)
    for name, score in [('popular_pre_07nov', 'purchases_7d'), ('buy_rate', 'buy_rate'), ('trend_live', 'train_buys')]:
        top_ids = set(e.sort_values(score, ascending=False).groupby('category').head(10).product_id)
        for flag, seg in segmented.groupby('is_new_user'):
            ranks[name][f'organic_recall10_new_{flag}'] = float(
                seg.loc[seg.product_id.isin(top_ids), 'purchases'].sum() / max(seg.purchases.sum(), 1))
    # Impresiones reales (feedback.shown), para comparar afinidad de categoría sin duplicados.
    imp = d[['request_id', 'is_new_user', 'view_categories', 'purchased', 'shown']].explode('shown')
    imp['category'] = imp.shown.map(cat)
    imp['same_category'] = [c in cs for c, cs in zip(imp.category, imp.view_categories)]
    imp['buy'] = imp.shown.eq(imp.purchased).astype(int)
    ig = imp.groupby(['is_new_user', 'same_category']).agg(impressions=('buy', 'size'), purchases=('buy', 'sum'))
    ig['buy_rate'] = ig.purchases / ig.impressions
    ig.to_csv(out / 'live_category_exposures.csv')
    # Productos comprados por contexto: distribución, con denominadores de peticiones en los otros CSV.
    dg = d[d.buy == 1].groupby(['is_new_user', 'page', 'device', 'region', 'buy_cat']).size()
    dg.rename('purchases').to_csv(out / 'live_purchase_context.csv')
    summary = {'start_utc': start, 'end_utc': end, 'requests': len(d),
               'new_users': int(d.is_new_user.sum()), 'ranking_proxy': ranks}
    (out / 'live_metrics.json').write_text(json.dumps(summary, indent=2))
    print(g.to_string())
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--live', action='store_true')
    ap.add_argument('--refresh-catalog', action='store_true')
    args = ap.parse_args()
    out = DATA_DIR / 'coldstart_eval'
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'catalog.parquet'
    if args.refresh_catalog or not path.exists():
        catalog_text().to_parquet(path, index=False)
    catalog = pd.read_parquet(path)
    items = pd.read_parquet(DATA_DIR / 'features_live/item_features.parquet')
    edges = pd.read_parquet(DATA_DIR / 'features_live/item_neighbors.parquet')
    t = time.perf_counter()
    cold = build(items, edges, catalog, cache_dir=None)
    for name, table in cold.items():
        table.to_parquet(out / f'{name}.parquet', index=False)
    print('Construcción segundos:', round(time.perf_counter() - t, 3), {n: len(v) for n, v in cold.items()})
    print(json.dumps(evaluate(catalog, items, edges, out), indent=2))
    if args.live:
        live(out, catalog)


if __name__ == '__main__':
    main()
