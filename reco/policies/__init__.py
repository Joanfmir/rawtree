"""Registro de políticas: añade aquí las nuevas para poder usarlas con --policy."""
from .baseline import Baseline
from .coview import Coview
from .personal import Personal, PersonalV1

POLICIES = {p.name: p for p in (Baseline, Coview, Personal, PersonalV1)}
