"""Registro de políticas: añade aquí las nuevas para poder usarlas con --policy."""
from .baseline import Baseline
from .bird import Bird, BirdBlend, BirdBlendHoy, BirdBlendRank, BirdRank, BirdV2, BirdV2Hoy
from .coview import Coview
from .explore import GuliasExplore, PersonalExplore
from .home import HomeBest
from .personal import Personal, PersonalNoRebuy, PersonalV1, PersonalV3, PersonalV4, PersonalV5
from .ranker import Ranker, RankerFeedback

POLICIES = {p.name: p for p in (Baseline, Coview, Personal, PersonalV1, PersonalV3, PersonalV4, PersonalV5, PersonalNoRebuy,
                                Ranker, RankerFeedback, PersonalExplore, GuliasExplore, HomeBest, Bird, BirdV2, BirdRank, BirdBlend, BirdBlendRank,
                                BirdV2Hoy, BirdBlendHoy)}

from .interleave import PARES  # noqa: E402  (necesita POLICIES ya definido)

POLICIES.update({p.name: p for p in PARES})

# Versiones de Bird con nombre estandarizado vN-bird (ver CLAUDE.md): el mismo nombre sirve de política
# y de model_version en el leaderboard. Cada versión nueva se añade aquí.
BIRD_VERSIONS = {"v1-bird": Bird, "v2-bird": BirdV2, "v3-bird": BirdBlendRank}
POLICIES.update(BIRD_VERSIONS)

# Modelos de prueba entrenados solo con el feedback en vivo (no son una versión vN-bird): para el A/B,
# `make up AB=v3-bird=v3-bird,v2-hoy=v2-hoy`.
POLICIES.update({"v2-hoy": BirdV2Hoy, "v2-hoy-blend": BirdBlendHoy})
