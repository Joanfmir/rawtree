"""Segunda etapa (ordenador): features de un producto EN RELACIÓN con los candidatos que elige el selector.

Arquitectura de dos modelos:
    1. selector   clasificador P(compra) sobre todo el catálogo → se queda con los SET_SIZE mejores
    2. ordenador  ranking sobre esos candidatos, con las features de este módulo

Este código se usa igual al construir el dataset (para los 10 productos que enseñó el sistema viejo) y al
servir (para los 50 candidatos), así que el significado de cada feature es el mismo en los dos sitios.

Grupos (ver FeatureSchema V3_RANK):
    A  lo que opina el selector        s1_prob, s1_rank, s1_rel, s1_z
    B  posición relativa en la lista   rel_price, rel_pop, rel_affinity, rel_cvr
    C  redundancia                     set_same_sub, set_same_brand, sub_rank, set_same_cat
    D  seguridad del selector          set_top_prob, set_spread
    E  señales directas (de v2)        DIRECT
"""
from __future__ import annotations

import numpy as np

SET_SIZE = 50
LIST = ("s1_prob", "s1_rank", "s1_rel", "s1_z", "rel_price", "rel_pop", "rel_affinity", "rel_cvr",
        "set_same_sub", "set_same_brand", "sub_rank", "set_same_cat", "set_top_prob", "set_spread")
DIRECT = ("u_attr_share", "sv_attr_share", "recently_viewed", "sv_recency", "view_to_purchase_score",
          "it_c_cvr", "it_trend", "u_logp_diff", "is_new_user", "page")


def _percentile(in_set: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Parte del conjunto que queda por debajo de cada valor (los empates cuentan la mitad). 0 = el más bajo."""
    s = np.sort(in_set)
    return (np.searchsorted(s, values, "left") + np.searchsorted(s, values, "right")) / 2 / len(s)


def select(live, selector, req, allowed: np.ndarray, size: int = SET_SIZE):
    """Primera etapa: puntúa todo el catálogo y devuelve (columnas, probabilidades, índices de los candidatos).
    `allowed` marca lo que se puede recomendar a este usuario (activo, en stock, no comprado)."""
    schema = selector.schema
    cols = live._columns(req, schema)
    p = selector.predict_matrices(*live.matrices(req, schema, np.arange(live.n), out=cols))
    ranked = np.argsort(-np.where(allowed, p, -np.inf), kind="stable")
    return cols, p, ranked[:min(size, int(allowed.sum()))]


def list_features(live, cols: dict, p: np.ndarray, chosen: np.ndarray, rows: np.ndarray) -> dict[str, np.ndarray]:
    """Features de lista (grupos A–D) de los productos `rows` frente a los candidatos `chosen`.
    `rows` no tiene por qué estar dentro de `chosen` (en el histórico se enseñaban otros productos):
    su puesto se topa en len(chosen) + 1."""
    ps, pr = p[chosen], p[rows]
    top, mean, std = ps[0], ps.mean(), ps.std()
    num = lambda name: np.asarray(cols[name], dtype=float)   # noqa: E731
    same = lambda code: (code[rows][:, None] == code[chosen][None, :])   # noqa: E731
    same_sub = same(live.sub)
    return {
        "s1_prob": pr,
        "s1_rank": np.minimum(1 + (ps[None, :] > pr[:, None]).sum(axis=1), len(chosen) + 1).astype(float),
        "s1_rel": pr / max(top, 1e-12),
        "s1_z": (pr - mean) / std if std > 0 else np.zeros(len(rows)),
        "rel_price": _percentile(live.logp[chosen], live.logp[rows]),
        "rel_pop": _percentile(num("it_buys_7d")[chosen], num("it_buys_7d")[rows]),
        "rel_affinity": _percentile(num("u_attr_share")[chosen], num("u_attr_share")[rows]),
        "rel_cvr": _percentile(num("it_c_cvr")[chosen], num("it_c_cvr")[rows]),
        "set_same_sub": same_sub.mean(axis=1),
        "set_same_brand": same(live.brand).mean(axis=1),
        "sub_rank": 1.0 + (same_sub & (ps[None, :] > pr[:, None])).sum(axis=1),
        "set_same_cat": same(live.cat).mean(axis=1),
        "set_top_prob": np.full(len(rows), top),
        "set_spread": np.full(len(rows), top - ps[min(9, len(ps) - 1)]),
    }


def direct_features(cols: dict, rows: np.ndarray) -> dict[str, np.ndarray]:
    """Grupo E al servir: las mismas columnas de v2, tal como las calcula LiveState."""
    return {name: np.asarray(cols[name])[rows] for name in DIRECT}
