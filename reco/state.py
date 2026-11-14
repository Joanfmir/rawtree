"""Estado compartido: publicación coherente sin bloquear durante las consultas SQL."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import RLock
from itertools import chain

from . import catalog


@dataclass
class State:
    catalog: dict[str, dict] = field(default_factory=dict)
    unavailable: set[str] = field(default_factory=set)
    best: list[str] = field(default_factory=list)
    _catalog_at: float = 0.0
    _stock_at: float = 0.0
    _lock: object = field(default_factory=RLock, repr=False, compare=False)

    def snapshot(self) -> State:
        """Referencias estables: refresh sustituye colecciones, nunca las modifica."""
        with self._lock:
            return State(self.catalog, self.unavailable, self.best)

    def refresh(self, catalog_every: float = 30, stock_every: float = 1) -> None:
        now = time.monotonic()
        # No publicar nada si una consulta falla; conservar el último estado completo.
        best = self.best or catalog.best_sellers()
        update_catalog = not self.catalog or now - self._catalog_at >= catalog_every
        update_stock = not self._stock_at or now - self._stock_at >= stock_every
        rows = catalog.current_catalog() if update_catalog else self.catalog
        stock = catalog.out_of_stock() if update_stock else self.unavailable
        with self._lock:
            self.best, self.catalog, self.unavailable = best, rows, stock
            if update_catalog:
                self._catalog_at = now
            if update_stock:
                self._stock_at = now

    def is_valid(self, pid: str) -> bool:
        if not isinstance(pid, str) or not pid:
            return False
        with self._lock:
            row = self.catalog.get(pid)
            active = row.get("is_active") if row else False
            return (active is True or type(active) is int and active == 1) and pid not in self.unavailable

    def fallback(self) -> list[str]:
        snap = self.snapshot()
        out = []
        for pid in chain(snap.best, snap.catalog):
            if pid not in out and snap.is_valid(pid):
                out.append(pid)
                if len(out) == 10:
                    break
        return out
