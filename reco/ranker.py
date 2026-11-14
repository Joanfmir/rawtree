"""Re-ranker: un gradient boosting (HistGradientBoosting de scikit-learn) que reordena los candidatos
de una política base. La base propone candidatos con sus señales; el modelo aprende, de los carruseles
reales del log, qué combinación de señales acaba en clic o compra.

    python scripts/train_hgb.py                      # modelo de producción  → data/models/hgb_live.pkl
    python scripts/train_hgb.py --until 2026-11-07   # modelo para evaluar   → data/models/hgb_2026-11-07.pkl
    python -m reco.evaluate --policies personal hgb  # ¿mejora a la base?

Una fila por (carrusel, producto mostrado). Columnas, en tres grupos:
  sig_*    las señales de la política base (base.signals()). No hay lista fija: se usan las que
           devuelva la base, así que una señal nueva entra sola al reentrenar.
  item_*   datos del producto (precio, popularidad, tasa de compra, categoría).
  pair_*   relación del producto con lo último que miró el comprador (subcategoría, marca, precio).
  req_*    datos de la petición (nº de vistas, página, dispositivo, usuario conocido) y `slot`.
`slot` es la posición en la que el sistema viejo enseñó el producto: al entrenar absorbe el sesgo de
posición (arriba se compra más) y al servir se fija a 0 para todos los candidatos.

Sin fugas: las señales de cada semana de entrenamiento se calculan solo con historia anterior a ella
(features --until <inicio de la semana>), igual que hace reco.evaluate con el test.

Si la base cambia (señales nuevas, quitadas o recalculadas) hay que REENTRENAR. Mientras tanto no se
rompe nada: las columnas que falten se tratan como dato ausente y las que sobren se ignoran.
"""
from __future__ import annotations

import pickle
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import features
from .client import query
from .config import DATA_DIR

HISTORY_END = "2026-11-14"   # último día de shop.history_sessions
MODELS_DIR = DATA_DIR / "models"

ITEM_COLS = ["price_eur", "views", "carts", "purchases", "views_7d", "purchases_7d", "purchases_1d", "buy_rate", "is_new"]
COUNT_COLS = {"views", "carts", "purchases", "views_7d", "purchases_7d", "purchases_1d"}   # van en log1p
ITEM_NAMES = [f"item_{c}" for c in ITEM_COLS] + ["item_category"]
PAIR_NAMES = ["pair_same_subcat_last", "pair_same_brand_last", "pair_same_subcat_any", "pair_price_ratio"]
REQ_NAMES = ["req_n_views", "req_page", "req_device", "req_known_user", "slot"]
CATEGORICAL = ["item_category", "req_page", "req_device"]
PAGES = {"home": 0, "product": 1, "category": 2}
DEVICES = {"web": 0, "ios": 1, "android": 2}


def model_path(until: str | None) -> Path:
    return MODELS_DIR / f"hgb_{until or 'live'}.pkl"


class Featurizer:
    """Convierte (petición, candidatos con señales) en la matriz que entra al modelo."""

    def __init__(self, items: pd.DataFrame, categories: list[str] | None = None) -> None:
        items = items.reset_index(drop=True)
        self.row = {pid: i for i, pid in enumerate(items.product_id)}
        num = items[ITEM_COLS].apply(pd.to_numeric, errors="coerce").astype(float)
        for c in COUNT_COLS:
            num[c] = np.log1p(num[c].fillna(0))
        self.categories = categories or sorted(items.category.dropna().astype(str).unique())
        code = {c: i for i, c in enumerate(self.categories)}
        cat = items.category.map(code).astype(float)   # categoría desconocida → NaN
        table = np.column_stack([num.to_numpy(), cat.to_numpy()])
        # última fila = producto que no está en la tabla (todo ausente); el índice -1 cae en ella
        self.items = np.vstack([table, np.full((1, table.shape[1]), np.nan)])
        self.subcat = np.append(pd.factorize(items.subcategory)[0].astype(float), np.nan)
        self.brand = np.append(pd.factorize(items.brand)[0].astype(float), np.nan)
        self.price = np.append(num.price_eur.to_numpy(), np.nan)

    @staticmethod
    def names(signal_names) -> list[str]:
        return [f"sig_{n}" for n in signal_names] + ITEM_NAMES + PAIR_NAMES + REQ_NAMES

    def matrix(self, req: dict, signal_names, cands: dict, pids: list[str], slots=None) -> np.ndarray:
        """Una fila por producto de `pids`, columnas en el orden de names(signal_names)."""
        n, k = len(pids), len(signal_names)
        zero = [0.0] * k
        sig = np.array([cands.get(p, zero) for p in pids], dtype=float).reshape(n, k)
        idx = np.array([self.row.get(p, -1) for p in pids])

        views = [self.row.get(v.get("product_id"), -1) for v in (req.get("recent_views") or [])]
        pair = np.full((n, len(PAIR_NAMES)), np.nan)
        if views:
            last = views[-1]
            pair[:, 0] = self.subcat[idx] == self.subcat[last]
            pair[:, 1] = self.brand[idx] == self.brand[last]
            pair[:, 2] = np.isin(self.subcat[idx], self.subcat[views])
            mean_price = np.nanmean(self.price[views]) if not np.all(np.isnan(self.price[views])) else np.nan
            pair[:, 3] = self.price[idx] / mean_price if mean_price and mean_price > 0 else np.nan

        ctx = req.get("context") or {}
        r = np.empty((n, len(REQ_NAMES)))
        r[:, 0] = len(views)
        r[:, 1] = PAGES.get(ctx.get("page"), np.nan)
        r[:, 2] = DEVICES.get(ctx.get("device"), np.nan)
        r[:, 3] = 0.0 if req.get("is_new_user") or not req.get("user_id") else 1.0
        r[:, 4] = 0.0 if slots is None else slots
        return np.hstack([sig, self.items[idx], pair, r])


class Ranker:
    """Modelo entrenado + lo necesario para usarlo: nombres de columnas, categorías y política base."""

    def __init__(self, model, feature_names: list[str], categories: list[str], base: str, meta: dict) -> None:
        self.model, self.feature_names, self.categories, self.base, self.meta = model, feature_names, categories, base, meta
        self._cols: dict[tuple, tuple] = {}

    def align(self, X: np.ndarray, signal_names) -> np.ndarray:
        """Reordena las columnas de X a las que vio el modelo al entrenar. Si la base ha cambiado sus
        señales desde entonces, las que falten quedan como ausentes (NaN) y las nuevas se ignoran."""
        key = tuple(signal_names)
        if key not in self._cols:
            have = {n: i for i, n in enumerate(Featurizer.names(signal_names))}
            missing = [n for n in self.feature_names if n not in have]
            extra = [n for n in have if n not in self.feature_names]
            if missing or extra:
                print(f"[hgb] las señales de la base han cambiado desde el entrenamiento (faltan {missing}, "
                      f"nuevas {extra}): reentrena con scripts/train_hgb.py")
            self._cols[key] = (np.array([have.get(n, -1) for n in self.feature_names]), bool(missing))
        cols, has_missing = self._cols[key]
        out = X[:, cols]
        if has_missing:
            out[:, cols < 0] = np.nan
        return out

    def score(self, X: np.ndarray, signal_names) -> np.ndarray:
        return self.model.predict_proba(self.align(X, signal_names))[:, 1]

    def save(self, until: str | None) -> Path:
        path = model_path(until)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(self))
        tmp.replace(path)   # cambio atómico: serve nunca lee un fichero a medias
        return path


def load(until: str | None) -> Ranker | None:
    path = model_path(until)
    return pickle.loads(path.read_bytes()) if path.exists() else None


# ── Entrenamiento ─────────────────────────────────────────────────────────────────────────────

def load_carousels(start: str, end: str) -> pd.DataFrame:
    """Carruseles del log con clic entre dos fechas (los que no tienen clic no enseñan a ordenar)."""
    cache = DATA_DIR / f"train_carousels_{start}_{end}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    df = pd.DataFrame(query(f"""
        SELECT request_id, ts, user_id, context, recent_views, shown, propensity, clicked, purchased
        FROM shop.history_sessions
        WHERE ts >= '{start}' AND ts < '{end}' AND clicked::String != ''""", timeout=600))
    for col in ("clicked", "purchased"):
        df[col] = df[col].fillna("").astype(str)
    df["recent_views"] = df["recent_views"].apply(lambda v: v or [])
    DATA_DIR.mkdir(exist_ok=True)
    cache.write_bytes(pickle.dumps(df))
    return df


def training_table(base, fz: Featurizer, df: pd.DataFrame, known_users: set[str]) -> dict:
    """Filas de entrenamiento de un bloque de carruseles: X, clic y compra (0/1), propensity y señales."""
    slots = np.arange(10, dtype=float)
    X, click, buy, prop, names = [], [], [], [], None
    for row in df.itertuples(index=False):
        shown = list(row.shown)
        req = {"request_id": row.request_id, "user_id": row.user_id, "is_new_user": row.user_id not in known_users,
               "context": row.context, "recent_views": row.recent_views, "ts": row.ts}
        names, cands = base.signals(req, None)
        X.append(fz.matrix(req, names, cands, shown, slots[:len(shown)]))
        click.append([p == row.clicked for p in shown])
        buy.append([p == row.purchased for p in shown])
        prop.append(list(row.propensity)[:len(shown)])
    return {"X": np.vstack(X), "click": np.concatenate(click).astype(int), "buy": np.concatenate(buy).astype(int),
            "prop": np.concatenate(prop).astype(float), "signals": names}


def week_starts(until: str | None, weeks: int) -> list[str]:
    """Inicios de las `weeks` semanas de entrenamiento, justo antes de `until` (o del final del log)."""
    end = date.fromisoformat(until or HISTORY_END)
    return [(end - timedelta(days=7 * k)).isoformat() for k in range(weeks, 0, -1)]


def build_table(until: str | None = None, weeks: int = 3, base_name: str = "personal", verbose: bool = True) -> dict:
    """Tabla de entrenamiento con las `weeks` semanas anteriores a `until` (None = final del log).
    Cada semana usa las señales de la base calculadas solo con historia anterior a esa semana."""
    from .policies import POLICIES

    parts, fz = [], None
    for start in week_starts(until, weeks):
        end = (date.fromisoformat(start) + timedelta(days=7)).isoformat()
        t = time.time()
        if not features.folder(start).exists():
            features.build(until=start, verbose=False)
        base = POLICIES[base_name](until=start)
        base.fit(None)
        items = pd.read_parquet(features.folder(start) / "item_features.parquet")
        fz = Featurizer(items, categories=fz.categories if fz else None)
        known = set(pd.read_parquet(features.folder(start) / "user_subcat.parquet", columns=["user_id"]).user_id)
        df = load_carousels(start, end)
        parts.append(training_table(base, fz, df, known))
        if verbose:
            print(f"  semana {start}: {len(df):,} carruseles · {parts[-1]['click'].sum():,} clics · "
                  f"{parts[-1]['buy'].sum():,} compras · {time.time() - t:.0f}s")
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("X", "click", "buy", "prop")}
    out.update(signals=parts[-1]["signals"], categories=fz.categories, base=base_name, until=until, weeks=weeks)
    return out


def fit(table: dict, target: str = "mixed", ips_clip: float = 0.05, verbose: bool = True, **params) -> Ranker:
    """Ajusta el gradient boosting sobre una tabla de build_table().

    target    'mixed' (el clic es acierto y la compra pesa el triple), 'click' o 'purchase'.
    ips_clip  cada fila pesa 1 / max(propensity, ips_clip): compensa que el sistema viejo enseñaba sobre
              todo best sellers, para que el modelo no aprenda "popular = bueno". 0 lo desactiva.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, click, buy = table["X"], table["click"], table["buy"]
    feature_names = Featurizer.names(table["signals"])
    y = buy if target == "purchase" else click
    w = 1.0 + 2.0 * buy if target == "mixed" else np.ones(len(y))
    if ips_clip:
        w = w / np.clip(table["prop"], ips_clip, None)
    w = w / w.mean()
    cfg = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
               early_stopping=True, validation_fraction=0.1, n_iter_no_change=25, random_state=0)
    cfg.update(params)
    model = HistGradientBoostingClassifier(
        categorical_features=[feature_names.index(c) for c in CATEGORICAL], **cfg)
    t = time.time()
    model.fit(X, y, sample_weight=w)
    meta = {"until": table["until"], "weeks": table["weeks"], "target": target, "ips_clip": ips_clip,
            "rows": int(len(y)), "positives": int(y.sum()), "trees": int(model.n_iter_),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if verbose:
        print(f"  {len(y):,} filas · {len(feature_names)} columnas · {model.n_iter_} árboles · {time.time() - t:.0f}s")
    return Ranker(model, feature_names, table["categories"], table["base"], meta)


def train(until: str | None = None, weeks: int = 3, base_name: str = "personal", verbose: bool = True, **fit_args) -> Ranker:
    return fit(build_table(until, weeks, base_name, verbose), verbose=verbose, **fit_args)
