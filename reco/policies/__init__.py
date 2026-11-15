"""Registro de políticas: añade aquí las nuevas para poder usarlas con --policy."""
from .baseline import Baseline
from .bird import Bird, BirdBlend, BirdBlendRank, BirdRank, BirdTwoStage, BirdV2, BirdV5, BirdV5Delta, BirdV5DeltaRetrained, BirdV5Organic
from .coview import Coview
from .explore import GuliasExplore, PersonalExplore
from .home import HomeBest
from .personal import Personal, PersonalNoRebuy, PersonalV1, PersonalV3, PersonalV4, PersonalV5
from .ranker import Ranker, RankerFeedback

POLICIES = {p.name: p for p in (Baseline, Coview, Personal, PersonalV1, PersonalV3, PersonalV4, PersonalV5, PersonalNoRebuy,
                                Ranker, RankerFeedback, PersonalExplore, GuliasExplore, HomeBest, Bird, BirdV2, BirdRank, BirdBlend, BirdBlendRank, BirdTwoStage,
                                BirdV5, BirdV5Delta, BirdV5DeltaRetrained, BirdV5Organic)}

from .interleave import PARES  # noqa: E402  (necesita POLICIES ya definido)

POLICIES.update({p.name: p for p in PARES})

# Versiones de Bird con nombre estandarizado vN-bird (ver CLAUDE.md): el mismo nombre sirve de política
# y de model_version en el leaderboard. Cada versión nueva se añade aquí.
BIRD_VERSIONS = {"v1-bird": Bird, "v2-bird": BirdV2, "v3-bird": BirdBlendRank, "v4-bird": BirdTwoStage,
                 "v5-bird": BirdV5, "v6-bird": BirdV5Delta, "v7-bird": BirdV5DeltaRetrained,
                 "v8-bird": BirdV5Organic}
POLICIES.update(BIRD_VERSIONS)
