"""Transferencia de señales por texto; construir fuera del proceso de servicio."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .client import query_df
from .config import DATA_DIR


def catalog_text() -> pd.DataFrame:
    """Catálogo completo con última versión, únicamente SELECT (~5.100 filas)."""
    cols = ('product_id::String AS product_id, category::String AS category, '
            'subcategory::String AS subcategory, brand::String AS brand, '
            'price_eur::Float64 AS price_eur, description::String AS description')
    return query_df(f"""SELECT product_id, argMax(category,v) AS category,
        argMax(subcategory,v) AS subcategory, argMax(brand,v) AS brand,
        argMax(price_eur,v) AS price_eur, argMax(description,v) AS description
        FROM (SELECT {cols}, toDateTime64('1970-01-01',3) AS v FROM shop.catalog
        UNION ALL SELECT {cols}, ts::DateTime64(3) AS v FROM shop.catalog_updates)
        GROUP BY product_id ORDER BY product_id""")


def text_neighbors(catalog: pd.DataFrame, donors: set[str], k: int = 10,
                   dimensions: int = 64) -> pd.DataFrame:
    """TF-IDF + SVD; vecinos donantes de la misma categoría, nunca el propio ID."""
    if k < 1:
        raise ValueError('k debe ser positivo')
    c = catalog.fillna('').sort_values('product_id').reset_index(drop=True)
    if len(c) < 2:
        return pd.DataFrame(columns=['p', 'q', 'sim'])
    text = c.description.astype(str)
    if not text.str.strip().any():
        return pd.DataFrame(columns=['p', 'q', 'sim'])
    x = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=20000,
                        token_pattern=r'(?u)\b\w+\b').fit_transform(text)
    d = min(dimensions, x.shape[1] - 1, x.shape[0] - 1)
    z = normalize(TruncatedSVD(d, random_state=17, n_iter=5).fit_transform(x)
                  if d > 0 else x.toarray())
    rows = []
    for _, g in c.groupby('category', sort=True):
        ids = g.index.to_numpy()
        di = np.array([i for i in ids if c.product_id[i] in donors], dtype=int)
        if not len(di):
            continue
        sims = z[ids] @ z[di].T
        for i, ss in zip(ids, sims):
            ss[di == i] = -1
            order = np.argsort(-ss, kind='stable')[:k]
            rows.extend((c.product_id[i], c.product_id[di[j]], float(ss[j]))
                        for j in order if ss[j] > 0)
    return pd.DataFrame(rows, columns=['p', 'q', 'sim'])


PRIOR_COLS = ['views', 'carts', 'purchases', 'views_7d', 'purchases_7d', 'purchases_1d', 'buy_rate']


def inherit(neighbors: pd.DataFrame, items: pd.DataFrame, item_neighbors: pd.DataFrame,
            targets: set[str]) -> dict[str, pd.DataFrame]:
    """Media ponderada de priors y mezcla de distribuciones de co-compra/co-vista.

    No utiliza ninguna fila saliente ni señal del producto ocultado como donante.
    Los n heredados son pesos relativos, no recuentos de eventos observados.
    """
    nb = neighbors[neighbors.p.isin(targets)].copy()
    nb['w'] = nb.sim / nb.groupby('p').sim.transform('sum')
    values = items[['product_id'] + PRIOR_COLS].copy().fillna(0)
    values['buy_rate'] = (values.purchases + 1) / (values.views + 50)
    m = nb.merge(values, left_on='q', right_on='product_id')
    for col in PRIOR_COLS:
        m[col] *= m.w
    priors = m.groupby('p')[PRIOR_COLS].sum().reset_index().rename(columns={'p': 'product_id'})
    edges = item_neighbors.copy()
    edges = edges[~edges.p.isin(targets) & ~edges.q.isin(targets)]
    edges['n'] = edges.n / edges.groupby(['p', 'kind']).n.transform('sum').clip(lower=1)
    e = nb.rename(columns={'p': 'target', 'q': 'donor'}).merge(edges, left_on='donor', right_on='p')
    e['n'] *= e.w
    e = e[e.target != e.q].groupby(['target', 'q', 'kind']).n.sum().reset_index()
    e = e.rename(columns={'target': 'p'}).sort_values(['p', 'kind', 'n', 'q'], ascending=[True, True, False, True])
    e = e.groupby(['p', 'kind']).head(30).reset_index(drop=True)
    return {'text_neighbors': neighbors[['p', 'q', 'sim']], 'cold_priors': priors, 'cold_neighbors': e}


def build(items: pd.DataFrame, item_neighbors: pd.DataFrame, catalog: pd.DataFrame | None = None,
          cache_dir: Path | None = DATA_DIR / 'coldstart', k: int = 10) -> dict[str, pd.DataFrame]:
    """Cache por contenido y parámetros; solo hereda para productos sin vistas.

    Ejecutar en subproceso durante refresh. No llamar desde recommend().
    """
    catalog = catalog_text() if catalog is None else catalog
    targets = set(catalog.product_id) - set(items.loc[items.views.fillna(0) > 0, 'product_id'])
    donors = set(items.product_id) - targets
    digest = hashlib.sha256(f'v1:{k}'.encode())
    for frame in (catalog, items, item_neighbors):
        digest.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    folder = Path(cache_dir) / digest.hexdigest()[:20] if cache_dir else None
    names = ['text_neighbors', 'cold_priors', 'cold_neighbors']
    if folder and all((folder / f'{n}.parquet').exists() for n in names):
        return {n: pd.read_parquet(folder / f'{n}.parquet') for n in names}
    out = inherit(text_neighbors(catalog, donors, k), items, item_neighbors, targets)
    if folder:
        folder.mkdir(parents=True, exist_ok=True)
        for n, df in out.items():
            df.to_parquet(folder / f'{n}.parquet', index=False)
    return out


def augment(feats: dict[str, pd.DataFrame], cold: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Copia integrable antes de Featurizer/_index; conserva señales ya observadas."""
    out = dict(feats)
    it = feats['item_features'].set_index('product_id').copy()
    prior = cold['cold_priors'].set_index('product_id')
    ids = it.index.intersection(prior.index)
    ids = ids[it.loc[ids, 'views'].fillna(0).to_numpy() == 0]
    it = it.astype({col: float for col in PRIOR_COLS})
    it.loc[ids, PRIOR_COLS] = prior.loc[ids, PRIOR_COLS]
    out['item_features'] = it.reset_index()
    out['item_neighbors'] = pd.concat([feats['item_neighbors'], cold['cold_neighbors']], ignore_index=True)
    return out
