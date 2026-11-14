"""Esquema de features: la única fuente de verdad que comparten entrenamiento, artefacto e inferencia.

Una versión nueva del dataset = una entrada nueva en SCHEMAS. El modelo guardado lleva su esquema al lado
(schema.json), así que en inferencia nunca se adivina el orden ni el tipo de las columnas.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

METADATA = ("request_id", "user_id", "product_id", "timestamp", "target_click", "propensity")


@dataclass(frozen=True)
class FeatureSchema:
    version: str
    features: tuple[str, ...]              # en el orden en que las recibe el modelo
    categorical: tuple[str, ...]           # subconjunto de features
    target: str = "target_purchase"
    metadata: tuple[str, ...] = METADATA   # nunca se usan como features
    # Features que cambian con la petición o el usuario. Si no hay ninguna, la puntuación de un producto
    # es la misma para todas las peticiones y se calcula una sola vez (ver policies/bird.py).
    per_request: tuple[str, ...] = ()
    reference_position: int = 1            # posición con la que se puntúan los candidatos en inferencia

    def __post_init__(self) -> None:
        for name in ("features", "categorical", "metadata", "per_request"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if len(set(self.features)) != len(self.features):
            raise ValueError(f"esquema {self.version}: features repetidas")
        for name in ("categorical", "per_request"):
            extra = set(getattr(self, name)) - set(self.features)
            if extra:
                raise ValueError(f"esquema {self.version}: {name} fuera de features: {sorted(extra)}")
        leaked = set(self.features) & (set(self.metadata) | {self.target})
        if leaked:
            raise ValueError(f"esquema {self.version}: metadatos o target usados como feature: {sorted(leaked)}")

    @property
    def numeric(self) -> tuple[str, ...]:
        return tuple(f for f in self.features if f not in self.categorical)

    @property
    def is_static(self) -> bool:
        return not self.per_request

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> FeatureSchema:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


V1 = FeatureSchema(
    version="v1",
    features=("category", "subcategory", "brand", "price_eur", "position"),
    categorical=("category", "subcategory", "brand"),
)

# ── v2: dataset de reco/bird/build_dataset.py ───────────────────────────────────────────────────────
ITEM_CAT = ("category", "subcategory", "brand", "attr1", "attr2")
ITEM_NUM = ("price_eur", "it_price_vs_sub", "it_age_days", "it_buy_share", "it_view_share", "it_buy_rate",
            "it_cart_rate", "it_buys_7d", "it_views_7d", "it_buys_1d", "it_trend", "it_sub_rank",
            "it_c_ctr", "it_c_cvr")
CONTEXT = ("page", "device", "region")
SESSION = ("n_recent_views", "recently_viewed", "sv_recency", "sv_count", "sv_same_sub_last", "sv_same_cat_last",
           "sv_same_brand_last", "sv_sub_share", "sv_cat_share", "sv_brand_share", "sv_attr_share",
           "sv_logp_diff", "view_to_purchase_score", "coview_score")
USER = ("is_new_user", "u_events", "u_buys", "u_days_since_last", "u_logp_diff", "u_buy_logp_diff")
AFFINITY = ("u_sub_share", "u_brand_share", "u_cat_share", "u_attr_share", "u_sub_w", "u_brand_w", "u_sub_w_7d",
            "u_sub_buys", "u_brand_buys")
PAIR = ("up_views", "up_carts", "up_c_clicks", "up_days_since_seen")
# Compras anteriores de ESE producto. En el histórico predicen muchísimo (recompra); en vivo el simulador
# la penaliza. Por eso van en un grupo aparte: v2_norebuy es el mismo modelo sin ellas.
REBUY = ("up_buys", "up_c_buys", "previous_purchase_count", "already_purchased", "days_since_last_purchase")
GROUPS = {"producto": ITEM_CAT + ITEM_NUM, "contexto": CONTEXT, "sesion": SESSION, "usuario": USER,
          "afinidad": AFFINITY, "usuario_x_producto": PAIR, "recompra": REBUY, "posicion": ("position",)}
_META_V2 = METADATA + ("sample_weight", "revenue_eur")


def _v2(version: str, rebuy: bool) -> FeatureSchema:
    per_request = CONTEXT + SESSION + USER + AFFINITY + PAIR + (REBUY if rebuy else ())
    return FeatureSchema(version=version, features=ITEM_CAT + ITEM_NUM + per_request + ("position",),
                         categorical=ITEM_CAT + CONTEXT, per_request=per_request, metadata=_META_V2)


V2 = _v2("v2", rebuy=True)
V2_NOREBUY = _v2("v2_norebuy", rebuy=False)

SCHEMAS = {s.version: s for s in (V1, V2, V2_NOREBUY)}
