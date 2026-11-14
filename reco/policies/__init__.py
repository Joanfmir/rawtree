"""Registro de políticas: añade aquí las nuevas para poder usarlas con --policy."""
from .baseline import Baseline
from .bird import Bird, BirdBlend, BirdBlendRank, BirdRank, BirdV2
from .coview import Coview
from .explore import GuliasExplore, PersonalExplore
from .home import HomeBest
from .joan import Joan, JoanRerank
from .personal import Personal, PersonalNoRebuy, PersonalV1, PersonalV3, PersonalV4, PersonalV5
from .ranker import Ranker, RankerFeedback

POLICIES = {p.name: p for p in (Baseline, Coview, Personal, PersonalV1, PersonalV3, PersonalV4, PersonalV5, PersonalNoRebuy,
                                Ranker, RankerFeedback, PersonalExplore, GuliasExplore, HomeBest, Bird, BirdV2, BirdRank, BirdBlend, BirdBlendRank, Joan, JoanRerank)}

from .interleave import PARES  # noqa: E402  (necesita POLICIES ya definido)

POLICIES.update({p.name: p for p in PARES})

# Versiones de Bird con nombre estandarizado vN-bird (ver CLAUDE.md): el mismo nombre sirve de política
# y de model_version en el leaderboard. Cada versión nueva se añade aquí.
BIRD_VERSIONS = {"v1-bird": Bird, "v2-bird": BirdV2, "v3-bird": BirdBlendRank}
POLICIES.update(BIRD_VERSIONS)

# v3-joan: política joan (bird_v2 + reordenador orgánico para conocidos, RRF(gulias, v3-bird) para nuevos).
# Mismo nombre como política y como model_version: `make up AB=bird_v2=v2-bird-norebuy,v3-joan=v3-joan`.
POLICIES["v3-joan"] = Joan
