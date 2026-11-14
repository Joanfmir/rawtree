"""Estado compartido que leen las políticas: catálogo, stock, best sellers."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import catalog


@dataclass
class State:
    catalog: dict[str, dict] = field(default_factory=dict)
    unavailable: set[str] = field(default_factory=set)
    best: list[str] = field(default_factory=list)
    _catalog_at: float = 0.0
    _stock_at: float = 0.0

    def refresh(self, catalog_every: float = 30, stock_every: float = 1) -> None:
        now = time.time()
        if not self.best:
            self.best = catalog.best_sellers()
        if now - self._catalog_at >= catalog_every:
            self.catalog = catalog.current_catalog()
            self._catalog_at = now
        if now - self._stock_at >= stock_every:
            self.unavailable = catalog.out_of_stock()
            self._stock_at = now

    def is_valid(self, pid: str) -> bool:
        row = self.catalog.get(pid)
        active = bool(row and row.get("is_active", True)) if self.catalog else True
        return active and pid not in self.unavailable

    def fallback(self) -> list[str]:
        return [p for p in self.best if self.is_valid(p)][:10]
