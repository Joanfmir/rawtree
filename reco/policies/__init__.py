"""Registro de políticas: añade aquí las nuevas para poder usarlas con --policy."""
from .baseline import Baseline
from .coview import Coview

POLICIES = {p.name: p for p in (Baseline, Coview)}
