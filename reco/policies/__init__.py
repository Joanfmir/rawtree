"""Registro de políticas: añade aquí las nuevas para poder usarlas con --policy."""
from .baseline import Baseline
from .coview import Coview
from .explore import GuliasExplore, PersonalExplore
from .home import HomeBest
from .personal import Personal, PersonalNoRebuy, PersonalV1, PersonalV3, PersonalV4
from .ranker import Ranker, RankerFeedback

POLICIES = {p.name: p for p in (Baseline, Coview, Personal, PersonalV1, PersonalV3, PersonalV4, PersonalNoRebuy,
                                Ranker, RankerFeedback, PersonalExplore, GuliasExplore, HomeBest)}

from .interleave import PARES  # noqa: E402  (necesita POLICIES ya definido)

POLICIES.update({p.name: p for p in PARES})
