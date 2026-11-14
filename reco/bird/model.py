"""Entrenamiento, guardado y carga de Bird (CatBoostClassifier → P(compra) por impresión).

El artefacto es una carpeta con tres ficheros que viajan siempre juntos:
    model.cbm     el modelo
    schema.json   lista ordenada de features, categóricas, target y posición de referencia
    metrics.json  métricas de validación, parámetros y con qué datos se entrenó
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, FeaturesData, Pool

from ..config import DATA_DIR
from . import metrics
from .dataset import prepare
from .schema import FeatureSchema

# Valores de partida del contrato; no se ajustan hasta tener features personalizadas.
# Único cambio: la parada temprana mira Logloss y PRAUC solo se informa. Con tan pocas compras PRAUC salta
# mucho entre iteraciones y la parada se quedaba con 1 árbol (medido con datos sintéticos).
DEFAULT_PARAMS = dict(iterations=500, depth=6, learning_rate=0.05, loss_function="Logloss",
                      eval_metric="Logloss", custom_metric=["PRAUC"], random_seed=42, thread_count=4,
                      allow_writing_files=False)   # sin carpeta catboost_info/ en el repo
# Compras ≪ no compras. Por defecto NO se repondera: Logloss ya ordena bien y deja las probabilidades
# calibradas. "sqrt"/"balanced" dan el mismo ranking pero inflan P(compra) (×10 con "sqrt" en la prueba),
# así que solo tienen sentido si después se recalibra.
IMBALANCE = {"none": None, "sqrt": "SqrtBalanced", "balanced": "Balanced"}


def model_dir(version: str, until: str | None = None) -> Path:
    return DATA_DIR / "bird" / (f"{version}_until_{until}" if until else version)


@dataclass
class BirdModel:
    model: CatBoostClassifier
    schema: FeatureSchema
    info: dict = field(default_factory=dict)

    def pool(self, df: pd.DataFrame, label=None) -> Pool:
        return Pool(prepare(df, self.schema), label=label, cat_features=list(self.schema.categorical))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """P(compra) de cada fila. `df` solo necesita las columnas del esquema (sobran las demás)."""
        if len(df) == 0:
            return np.zeros(0)
        return self.model.predict_proba(self.pool(df), thread_count=1)[:, 1]

    def predict_matrices(self, num: np.ndarray, cat: np.ndarray) -> np.ndarray:
        """P(compra) desde matrices ya preparadas (ver LiveState.matrices). CatBoost empareja las columnas por
        nombre, así que da exactamente lo mismo que predict() con un DataFrame, unas 10 veces más rápido."""
        if len(num) == 0:
            return np.zeros(0)
        data = FeaturesData(num_feature_data=num, cat_feature_data=cat,
                            num_feature_names=list(self.schema.numeric),
                            cat_feature_names=list(self.schema.categorical))
        return self.model.predict_proba(Pool(data), thread_count=1)[:, 1]

    def importance(self) -> pd.Series:
        imp = pd.Series(self.model.get_feature_importance(), index=list(self.schema.features))
        return imp.sort_values(ascending=False)

    def save(self, folder: Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(folder / "model.cbm"))
        self.schema.save(folder / "schema.json")
        (folder / "metrics.json").write_text(json.dumps(self.info, indent=2, default=str), encoding="utf-8")
        return folder

    @classmethod
    def load(cls, folder: Path) -> BirdModel:
        folder = Path(folder)
        if not (folder / "model.cbm").exists():
            raise FileNotFoundError(f"No hay modelo en {folder}. Entrénalo con: python scripts/bird.py train")
        model = CatBoostClassifier()
        model.load_model(str(folder / "model.cbm"))
        schema = FeatureSchema.load(folder / "schema.json")
        if list(model.feature_names_) != list(schema.features):
            raise ValueError(f"{folder}: el modelo y schema.json no tienen las mismas features")
        info_path = folder / "metrics.json"
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        return cls(model, schema, info)


def train(train_df: pd.DataFrame, valid_df: pd.DataFrame, schema: FeatureSchema, params: dict | None = None,
          imbalance: str = "none", early_stopping: int | None = 50, verbose: int = 0) -> BirdModel:
    """Entrena con parada temprana sobre `valid_df` (posterior en el tiempo a `train_df`).
    early_stopping=None: nº fijo de iteraciones (modelo final con todos los datos; `valid_df` solo informa)."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    if IMBALANCE[imbalance]:
        params["auto_class_weights"] = IMBALANCE[imbalance]
    cats = list(schema.categorical)
    weight = lambda df: df["sample_weight"].to_numpy() if "sample_weight" in df.columns else None   # noqa: E731
    train_pool = Pool(prepare(train_df, schema), label=train_df[schema.target].to_numpy(), cat_features=cats,
                      weight=weight(train_df))
    valid_pool = Pool(prepare(valid_df, schema), label=valid_df[schema.target].to_numpy(), cat_features=cats,
                      weight=weight(valid_df))
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=early_stopping,
              use_best_model=early_stopping is not None, verbose=verbose)
    bird = BirdModel(model, schema)
    bird.info = {
        "version": schema.version,
        "params": params,
        "imbalance": imbalance,
        "best_iteration": int(model.get_best_iteration() or 0),
        "trees": int(model.tree_count_),
        "train": {"filas": len(train_df), "compras": int(train_df[schema.target].sum()),
                  "desde": train_df.timestamp.min(), "hasta": train_df.timestamp.max()},
        "validation": {"filas": len(valid_df), "compras": int(valid_df[schema.target].sum()),
                       "desde": valid_df.timestamp.min(), "hasta": valid_df.timestamp.max()},
        "metrics": metrics.report(valid_df, bird.predict(valid_df), schema),
        "importance": bird.importance().round(2).to_dict(),
    }
    return bird
