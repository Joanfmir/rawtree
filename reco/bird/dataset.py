"""Dataset de Bird: una fila por (petición, producto enseñado). Carga, validación del contrato y preparación.

prepare() es la ÚNICA función que convierte un DataFrame en la matriz del modelo; la usan el entrenamiento
y la inferencia, así los tipos y los valores ausentes se tratan igual en los dos sitios.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from .schema import V1, FeatureSchema

UNKNOWN = "UNKNOWN"
KEY = ["request_id", "product_id"]


def load(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def prepare(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    """Columnas del esquema, en su orden: categóricas como texto (ausente → "UNKNOWN"), numéricas como float (NaN)."""
    missing = [c for c in schema.features if c not in df.columns]
    if missing:
        raise KeyError(f"faltan features del esquema {schema.version}: {missing}")
    out = {}
    for col in schema.features:
        s = df[col]
        if col in schema.categorical:
            s = s.astype(object)
            out[col] = s.where(s.notna() & (s != ""), UNKNOWN).astype(str).astype(object)
        else:
            out[col] = pd.to_numeric(s, errors="coerce").astype("float64")
    return pd.DataFrame(out, index=df.index)


def check(df: pd.DataFrame, schema: FeatureSchema, name: str = "dataset") -> list[str]:
    """Incumplimientos del contrato en un split. Lista vacía = correcto."""
    need = list(dict.fromkeys(["request_id", "user_id", "product_id", "timestamp", *schema.features, schema.target]))
    missing = [c for c in need if c not in df.columns]
    if missing:
        return [f"{name}: faltan columnas {missing}"]
    bad: list[str] = []
    if len(df) == 0:
        return [f"{name}: vacío"]
    dup = int(df.duplicated(KEY).sum())
    if dup:
        bad.append(f"{name}: {dup} pares (request_id, product_id) duplicados")
    ts = df["timestamp"]
    if not pdt.is_datetime64_any_dtype(ts) or str(getattr(ts.dt, "tz", None)) != "UTC":
        bad.append(f"{name}: timestamp debe ser datetime con zona UTC (es {ts.dtype})")
    for col in [schema.target, "target_click"]:
        if col in df.columns and not df[col].isin([0, 1]).all():
            bad.append(f"{name}: {col} tiene valores distintos de 0/1")
    if "position" in df.columns:
        pos = df["position"]
        if not pdt.is_integer_dtype(pos) or not pos.between(1, 10).all():
            bad.append(f"{name}: position debe ser entera entre 1 y 10")
        elif df.duplicated(["request_id", "position"]).any():
            bad.append(f"{name}: posiciones repetidas dentro de una petición")
    per_req = df.groupby("request_id", sort=False)
    size = per_req.size()
    if (size > 10).any():
        bad.append(f"{name}: {int((size > 10).sum())} peticiones con más de 10 productos")
    if (size < 10).any():
        bad.append(f"{name}: {int((size < 10).sum())} peticiones incompletas (menos de 10 productos)")
    if (per_req[schema.target].sum() > 1).any():
        bad.append(f"{name}: peticiones con más de una compra")
    if (per_req["user_id"].nunique() > 1).any() or (per_req["timestamp"].nunique() > 1).any():
        bad.append(f"{name}: user_id o timestamp no son constantes dentro de una petición")
    for col in schema.categorical:
        if not pdt.is_string_dtype(df[col]) and not pdt.is_object_dtype(df[col]):
            bad.append(f"{name}: {col} debe ser texto (es {df[col].dtype})")
        elif df[col].isna().any():
            bad.append(f'{name}: {col} tiene nulos (deben ser "{UNKNOWN}")')
    for col in schema.numeric:
        if not pdt.is_numeric_dtype(df[col]):
            bad.append(f"{name}: {col} debe ser numérica (es {df[col].dtype})")
    if "propensity" in df.columns:
        p = df["propensity"]
        if not pdt.is_numeric_dtype(p) or p.isna().any() or not ((p >= 0) & (p <= 1)).all():
            bad.append(f"{name}: propensity debe estar en [0, 1] y sin nulos")
    if df[schema.target].sum() == 0:
        bad.append(f"{name}: ninguna compra (target siempre 0)")
    return bad


def validate(train: pd.DataFrame, valid: pd.DataFrame, schema: FeatureSchema, strict: bool = True) -> list[str]:
    """Contrato §6 completo: cada split por separado + reparto cronológico sin peticiones compartidas."""
    bad = check(train, schema, "train") + check(valid, schema, "validation")
    if not bad:
        shared = set(train.request_id) & set(valid.request_id)
        if shared:
            bad.append(f"{len(shared)} peticiones aparecen en train y en validation")
        if valid.timestamp.min() <= train.timestamp.max():
            bad.append(f"validation empieza ({valid.timestamp.min()}) antes de que acabe train ({train.timestamp.max()})")
        if list(train.columns) != list(valid.columns) or (train.dtypes.astype(str) != valid.dtypes.astype(str)).any():
            bad.append("train y validation no tienen las mismas columnas y tipos")
    if bad and strict:
        raise ValueError("El dataset no cumple el contrato:\n  - " + "\n  - ".join(bad))
    return bad


def describe(df: pd.DataFrame, schema: FeatureSchema) -> str:
    n_req = df.request_id.nunique()
    buys = int(df[schema.target].sum())
    return (f"{len(df):,} filas · {n_req:,} peticiones · {buys:,} compras "
            f"({100 * buys / max(n_req, 1):.2f} % de las peticiones) · {df.timestamp.min():%Y-%m-%d %H:%M} → "
            f"{df.timestamp.max():%Y-%m-%d %H:%M}")


def split_by_time(df: pd.DataFrame, valid_share: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Corte cronológico por petición: las últimas `valid_share` peticiones van a validación."""
    first = df.groupby("request_id").timestamp.min().sort_values()
    cut = first.iloc[int(len(first) * (1 - valid_share))]
    return df[df.timestamp < cut].reset_index(drop=True), df[df.timestamp >= cut].reset_index(drop=True)


# ── Datos sintéticos: SOLO para probar el pipeline, no dicen nada del rendimiento real ─────────────────

def fake_catalog(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cats = [f"cat_{i}" for i in range(8)]
    cat = rng.integers(0, len(cats), n)
    sub = cat * 5 + rng.integers(0, 5, n)
    return pd.DataFrame({
        "product_id": [f"p_{i:06x}" for i in range(n)],
        "category": [cats[c] for c in cat],
        "subcategory": [f"sub_{s}" for s in sub],
        "brand": [f"brand_{b}" for b in rng.integers(0, 40, n)],
        "price_eur": np.round(np.exp(rng.normal(3.5, 1.0, n)), 2),
    })


def synthetic(n_requests: int = 5000, seed: int = 0, items: pd.DataFrame | None = None,
              start: str = "2026-10-01", end: str = "2026-11-14") -> pd.DataFrame:
    """Dataset con el esquema V1 (12 columnas del contrato). La compra depende de categoría, marca, precio y posición,
    así que un modelo que funcione tiene algo que aprender."""
    rng = np.random.default_rng(seed)
    items = (fake_catalog(seed=seed) if items is None else items).reset_index(drop=True)
    n_items = len(items)
    shown = rng.integers(0, n_items, (n_requests, 10))
    while True:   # 10 productos distintos por petición
        srt = np.sort(shown, axis=1)
        rows = np.flatnonzero((srt[:, 1:] == srt[:, :-1]).any(axis=1))
        if len(rows) == 0:
            break
        shown[rows] = rng.integers(0, n_items, (len(rows), 10))

    def effect(col: str, scale: float) -> np.ndarray:
        codes, uniques = pd.factorize(items[col].fillna(UNKNOWN))
        return rng.normal(0, scale, len(uniques))[codes]

    logp = np.log(items.price_eur.fillna(items.price_eur.median()).to_numpy())
    utility = effect("category", 0.6) + effect("subcategory", 0.3) + effect("brand", 0.4) - 0.35 * (logp - logp.mean())
    position = np.tile(np.arange(1, 11), (n_requests, 1))
    prob = 1 / (1 + np.exp(-(-4.6 + utility[shown] - 0.18 * (position - 1))))
    buy = rng.random(prob.shape) < prob
    # como mucho una compra y un clic por petición
    first = buy.argmax(axis=1)
    bought = buy.any(axis=1)
    buy[:] = False
    buy[np.flatnonzero(bought), first[bought]] = True
    click = buy.copy()
    extra = ~bought & (rng.random(n_requests) < 0.25)
    click[np.flatnonzero(extra), rng.integers(0, 10, int(extra.sum()))] = True

    t0, t1 = pd.Timestamp(start, tz="UTC").value, pd.Timestamp(end, tz="UTC").value
    ts = pd.to_datetime(np.sort(rng.integers(t0, t1, n_requests)), utc=True)
    flat = shown.ravel()
    df = pd.DataFrame({
        "request_id": np.repeat([f"r_{i:07d}" for i in range(n_requests)], 10),
        "user_id": np.repeat([f"u_{u:05d}" for u in rng.integers(0, max(n_requests // 4, 1), n_requests)], 10),
        "product_id": items.product_id.to_numpy()[flat],
        "timestamp": np.repeat(ts, 10),
        "category": items.category.fillna(UNKNOWN).to_numpy()[flat],
        "subcategory": items.subcategory.fillna(UNKNOWN).to_numpy()[flat],
        "brand": items.brand.fillna(UNKNOWN).to_numpy()[flat],
        "price_eur": items.price_eur.to_numpy()[flat].astype("float32"),
        "position": position.ravel().astype("int8"),
        "target_purchase": buy.ravel().astype("int8"),
        "target_click": click.ravel().astype("int8"),
        "propensity": rng.uniform(0.02, 0.5, flat.size),
    })
    for col in ("request_id", "user_id", "product_id", "category", "subcategory", "brand"):
        df[col] = df[col].astype("string")
    return df


__all__ = ["load", "prepare", "check", "validate", "describe", "split_by_time", "synthetic", "fake_catalog", "V1"]
