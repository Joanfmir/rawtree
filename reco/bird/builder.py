"""Feature Builder PROVISIONAL: build_features(request, candidatos, estado) → DataFrame con las features del esquema.

El contrato dice que el equipo de datos entregará esta función para que entrenamiento e inferencia calculen
las features igual. Hasta entonces, esta versión cubre V1 (metadatos de catálogo + posición de referencia).
Cuando llegue la suya, se sustituye este módulo y no hay que tocar ni el modelo ni la política.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .schema import V1, FeatureSchema

ITEM_COLUMNS = ("category", "subcategory", "brand", "price_eur")


@dataclass
class FeatureState:
    """Lo que hace falta saber para describir un candidato. Hoy: el catálogo. Mañana: historial del usuario, etc."""
    items: pd.DataFrame   # índice product_id; columnas ITEM_COLUMNS

    @classmethod
    def from_items(cls, df: pd.DataFrame) -> FeatureState:
        """Desde una tabla con product_id (p. ej. data/features_*/item_features.parquet)."""
        return cls(df.drop_duplicates("product_id").set_index("product_id")[list(ITEM_COLUMNS)])

    @classmethod
    def from_catalog(cls, catalog: dict[str, dict]) -> FeatureState:
        """Desde State.catalog ({product_id: fila}), el catálogo en vivo."""
        return cls.from_items(pd.DataFrame(list(catalog.values())))


def build_features(request: dict, candidate_ids, feature_state: FeatureState,
                   schema: FeatureSchema = V1) -> pd.DataFrame:
    """Una fila por candidato, en el mismo orden que `candidate_ids`. Producto desconocido → valores ausentes."""
    df = feature_state.items.reindex(list(candidate_ids))
    df["position"] = schema.reference_position
    missing = [f for f in schema.features if f not in df.columns]
    if missing:
        raise NotImplementedError(f"el builder provisional no sabe calcular {missing} (esquema {schema.version})")
    return df[list(schema.features)]
