"""Métricas de Bird.

  classification()  ¿acierta la probabilidad de compra de cada impresión? (logloss, PR-AUC, calibración)
  ranking_shown()   ¿pone arriba el producto comprado? SOLO entre los 10 que enseñó el sistema viejo.
                    No es rendimiento sobre el catálogo completo: para eso está `python -m reco.evaluate`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from .schema import FeatureSchema


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10, w: np.ndarray | None = None) -> pd.DataFrame:
    """Deciles de probabilidad predicha: media predicha frente a tasa real de compra (ponderadas por `w`)."""
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, max(len(edges) - 2, 0))
    t = pd.DataFrame({"bin": idx, "w": w, "wy": w * y, "wp": w * p}).groupby("bin").sum()
    return pd.DataFrame({"n": t.w.to_numpy(), "predicha": (t.wp / t.w).to_numpy(), "real": (t.wy / t.w).to_numpy()})


def classification(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> dict[str, float]:
    """`w` = sample_weight: deshace el muestreo de carruseles sin clic, así las tasas son las reales."""
    y, p = np.asarray(y).astype(int), np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    both = 0 < y.sum() < len(y)
    table = calibration_table(y, p, w=w)
    base = float(np.average(y, weights=w))
    return {
        "n": int(len(y)),
        "compras": int(y.sum()),
        "tasa_real": base,
        "tasa_predicha": float(np.average(p, weights=w)),
        "logloss": float(log_loss(y, p, labels=[0, 1], sample_weight=w)),
        # referencia: predecir siempre la tasa media. Un modelo útil queda por debajo.
        "logloss_constante": float(log_loss(y, np.full(len(y), np.clip(base, 1e-9, 1 - 1e-9)), labels=[0, 1],
                                            sample_weight=w)),
        "pr_auc": float(average_precision_score(y, p, sample_weight=w)) if both else float("nan"),  # azar = tasa_real
        "roc_auc": float(roc_auc_score(y, p, sample_weight=w)) if both else float("nan"),
        "ece": float((table.n * (table.predicha - table.real).abs()).sum() / table.n.sum()),
    }


def ranking_shown(df: pd.DataFrame, p: np.ndarray, target: str = "target_purchase") -> dict[str, float]:
    """Reordena los 10 enseñados por puntuación y mira dónde queda el comprado (peticiones con compra).
    `log_*` es lo mismo con el orden original del carrusel, como referencia."""
    d = pd.DataFrame({"request_id": df["request_id"].to_numpy(), "y": df[target].to_numpy(), "p": np.asarray(p)})
    d["rank"] = d.groupby("request_id", sort=False).p.rank(ascending=False, method="first")
    if "position" in df.columns:
        d["log_rank"] = df["position"].to_numpy()
    if "propensity" in df.columns:
        d["w"] = 1 / np.clip(df["propensity"].to_numpy(dtype=float), 1e-3, None)
    buys = d[d.y == 1]
    out: dict[str, float] = {"peticiones_con_compra": int(len(buys))}
    if len(buys) == 0:
        return out

    def block(rank: pd.Series, prefix: str, w: np.ndarray | None = None) -> None:
        r = rank.to_numpy(dtype=float)
        mean = (lambda v: float(v.mean())) if w is None else (lambda v: float((w * v).sum() / w.sum()))
        for k in (1, 3, 5):
            out[f"{prefix}hit@{k}"] = mean((r <= k).astype(float))
        out[f"{prefix}mrr@10"] = mean(1 / r)
        out[f"{prefix}ndcg@10"] = mean(1 / np.log2(r + 1))

    block(buys["rank"], "")
    if "log_rank" in buys:
        block(buys["log_rank"], "log_")
    if "w" in buys:   # SNIPS: corrige que el sistema viejo enseñaba unos productos mucho más que otros
        block(buys["rank"], "snips_", buys["w"].to_numpy())
    return out


def report(df: pd.DataFrame, p: np.ndarray, schema: FeatureSchema, probabilities: bool = True) -> dict[str, dict]:
    """probabilities=False (modelo de ranking): `p` son puntuaciones, así que solo se informan las métricas
    que dependen del orden (PR-AUC, ROC-AUC y el ranking por petición), no logloss ni calibración."""
    w = df["sample_weight"].to_numpy() if "sample_weight" in df.columns else None
    y = df[schema.target].to_numpy()
    if not probabilities:
        both = 0 < y.sum() < len(y)
        auc = lambda fn: float(fn(y, p, sample_weight=w)) if both else float("nan")   # noqa: E731
        return {"orden_global": {"n": int(len(y)), "compras": int(y.sum()),
                                 "pr_auc": auc(average_precision_score), "roc_auc": auc(roc_auc_score)},
                "ranking_en_los_10_mostrados": ranking_shown(df, p, schema.target)}
    return {"clasificacion": classification(y, p, w),
            "ranking_en_los_10_mostrados": ranking_shown(df, p, schema.target)}


def show(rep: dict[str, dict]) -> str:
    lines = []
    for title, block in rep.items():
        lines.append(f"── {title.replace('_', ' ')}")
        for k, v in block.items():
            lines.append(f"   {k:<22} {v:,.4f}" if isinstance(v, float) else f"   {k:<22} {v:,}")
    return "\n".join(lines)
