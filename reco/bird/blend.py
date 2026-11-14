"""Mezcla de dos listas ordenadas (Bird y personal) en una sola. Sin modelo: solo combina puestos.

  rerank  personal propone sus `n` mejores y Bird decide el orden (lo que Bird no puntuó queda al final,
          en el orden de personal)
  rrf     fusión por puestos recíprocos: cada lista aporta 1 / (k + puesto) y `alpha` es el peso de Bird.
          Sube lo que convence a los dos; un producto que solo está en una lista también puede entrar.
"""
from __future__ import annotations


def rerank(bird: list[str], personal: list[str], n: int = 30) -> list[str]:
    pos = {p: i for i, p in enumerate(bird)}
    head = personal[:n]
    order = sorted(range(len(head)), key=lambda i: (pos.get(head[i], len(bird)), i))
    return [head[i] for i in order]


def rrf(bird: list[str], personal: list[str], alpha: float = 0.5, k: float = 10.0) -> list[str]:
    score: dict[str, float] = {}
    for weight, items in ((alpha, bird), (1 - alpha, personal)):
        for i, p in enumerate(items):
            score[p] = score.get(p, 0.0) + weight / (k + i + 1)
    return sorted(score, key=score.get, reverse=True)


def fuse(bird: list[str], personal: list[str], mode: str = "rrf", **params) -> list[str]:
    if not personal:
        return bird
    if not bird:
        return personal
    return rerank(bird, personal, **params) if mode == "rerank" else rrf(bird, personal, **params)
