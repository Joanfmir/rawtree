"""Registro de políticas: añade aquí las nuevas para poder usarlas con --policy."""
from .baseline import Baseline
from .coview import Coview
from .hgb import Hgb
from .personal import Personal
from .personal2 import Personal2

POLICIES = {p.name: p for p in (Baseline, Coview, Personal, Personal2, Hgb)}
