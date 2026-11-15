# DreamTeam (team_17) · Real-time recommender for the RawTree Challenge

**Datathon FME 2026 · 14–15 November 2026 · UPC Barcelona**

A shopper arrives every ~0.3 s. For each one we have 3 seconds to pick the 10 products they will see. This repo
is the system we ran for about 24 hours against the live stream: a CatBoost purchase model with 81 features computed
in RawTree SQL, served from memory, retrained with our own live feedback, and chosen through simultaneous A/B
tests.

- Challenge statement (organisers' original README): [`ENUNCIADO.md`](ENUNCIADO.md)
- Demo script: [`docs/PRESENTACION.md`](docs/PRESENTACION.md)
- The 81 features, one by one, with the measurement that justifies each: [`docs/DATASET_V5.md`](docs/DATASET_V5.md)

## Result

**4th of 42 teams in the final round (Sunday 10:00–12:00 CET), with a score of 54.30 out of 100.**

| Final round, 21,600 requests | DreamTeam | Baseline | Organisers' reference |
|---|---|---|---|
| Expected purchases per 100 requests | 2.14 | 0.75 | 2.14 |
| The same on cold requests (new shopper or new product viewed) | 2.47 | 1.18 | 2.20 |
| Answered validly and on time | 97.3 % | 100 % | 100 % |

We sold almost three times as much as the best-seller baseline and matched the organisers' reference recommender,
beating it on cold start. Reliability was our weakest component: the three teams above us answered more than
99.9 % of requests on time.

---

## Why it looks like this: five decisions

Each of these changed the system. They are in the order we made them.

### 1. Stop recommending what the shopper already bought

**What we saw.** In the 60-day history, 62 % of purchases repeat a product the user had already bought, and an
already-bought product is 6.5× more likely to be bought again. Our first personalised policy exploited that
and won every offline metric.

**What happened live.** The opposite. In our own feedback, an already-bought product converted about 9× *less*
(the challenge statement hints at it: "a shopper who already bought a product tends not to want it again").

**What we did.** Every policy from `v2-bird` on never proposes a product the user has bought, and the offline
tests exclude repurchases.

### 2. Do not trust an offline metric until it has been checked against the leaderboard

**What we saw.** After decision 1 we no longer knew which offline number to believe. So we took the 8 versions
we had already served, with their real conversion on the leaderboard, and asked each offline test to rank them.

| Offline test · metric | Spearman vs leaderboard | Pairs ordered correctly |
|---|---|---|
| **organic · `ndcg@10`** | **0.98** | **27 / 28** |
| history · `ips_ndcg` (no repurchases) | 0.95 | 26 / 28 |
| history · `ips_ndcg` with repurchases | −0.14 | 12 / 28 |
| history · `hit@10` with repurchases | −0.60 | 8 / 28 |

**What we did.** The reference test became **organic**: real requests from the live stream, and the target is
what that user went on to buy *on their own* (`shop.events`) in the next 30 minutes. `make offline C="<policy>"`
runs it, recomputes this reliability table, and returns a verdict for the candidate (worse → drop · tie → only
live can tell · better → A/B). The metric we had started with would have ranked our versions backwards.

### 3. Features must mean the same thing in the history and in the stream

**What we saw.** The live stream has ~29× more events per hour than the history. A feature like "purchases in
the last 7 days" means something different when training than when serving. We measured the drift of each
feature (how well that feature alone tells a history row from a live row): the 12 worst averaged 0.44.

**What we did.** Dataset v5: no absolute counts, only shares, ratios and smoothed lifts (average drift 0.12), with a
unit test that enforces it. The same audit showed things the history cannot teach at all (it has no price changes;
live has 2-hour promotions with +42 % conversion in our carousel; 100 new products arrive every 4 hours and sell
from the first minute), so those features are learned from live rows only. Training and serving share **one**
implementation of every formula (`reco/bird/feat5.py`), and a parity check compares both paths value by value.

### 4. Our own feedback is the only direct measurement of the simulator

**What we did.** Live rows (`team_17.feedback` × `shop.requests`) go into the training set with 3× weight, and
two features are computed from them: clicks and purchases of each product in *our* carousel, normalised by the
position it was shown in. We retrained as feedback accumulated: `v5-bird` (Saturday night) → `v7-bird` (same
model, more live data) → the final-round model, retrained with feedback up to Sunday 10:00 CET.

### 5. A version goes to production only if it wins a simultaneous A/B

**Why.** Trends and stock change during the day, so comparing a version from 15:00 with one from 18:00 says
nothing. Requests are split across arms by `crc32(request_id)`, each arm writes its own `model_version`, and we
compare them over the same window with standard errors.

**The idea it killed.** `v8-bird` added organic purchases to the training set. Offline it tied; its own note said
"do not take it to the final round without winning an A/B". Over three hours with four arms on equal traffic:

| Version | Requests | Purchases per 100 requests |
|---|---|---|
| v7-bird | 7,764 | 2.29 ± 0.17 |
| v5-bird | 7,883 | 2.21 ± 0.17 |
| v6-bird | 7,724 | 2.20 ± 0.17 |
| v8-bird | 7,866 | 1.92 ± 0.15 |

v5, v6 and v7 are indistinguishable; v8 was last in every hour. Twenty-four minutes into the final round we
stopped the A/B and served a single version, the retrained v7.

---

## How the system works

1. **Read.** The serve loop polls `shop.requests` in batches.
2. **Score.** For each request the model scores every in-stock product the user has not bought:
   P(purchase | user, session, context, product). There is no candidate generator in the final version: the
   whole catalogue is scored from memory. End to end (polling included) the median answer takes ~1.6 s and the
   95th percentile ~2.4 s of the 3 s allowed.
3. **Guard.** `finalize()` guarantees 10 distinct, active, in-stock IDs. No answer leaves without passing through it.
4. **Write.** The list goes to `team_17.submissions` with the arm's `model_version`.
5. **Learn.** State is refreshed from the stream in the background (popularity and trends in 10-minute steps, a
   per-user delta of what happened since the last full download, catalogue and stock). The model itself is
   retrained by hand with the new feedback; switching arms is a team decision, not an automatic bandit.

Heavy work never happens inside `recommend()`: it goes into SQL, `fit()` or `refresh()`.

## How the model is trained

| | |
|---|---|
| Model | CatBoost classifier, P(purchase) per impression |
| History rows | `shop.history_sessions`: every carousel with a click + 1 in 25 without (weighted ×25). Train 26/09–30/10, validation 31/10–06/11, test 07/11–14/11 |
| Live rows | Our own carousels from `team_17.feedback`: every one with a click + 1 in 10 without. Weight ×3 |
| Repurchases | Impressions of products the user had already bought are removed |
| Cold start | In 30 % of history requests the user's history is hidden, so the model learns to decide from session and context alone |
| Features | 81: product (catalogue, price, promotion, organic popularity, trend), response in the carousel, context (device, region, page), session (`recent_views` vs candidate), user, user × candidate affinity |
| Validation | Temporal split for model selection; the organic test and a live A/B before production |

Commands: `python scripts/bird_v5.py dataset --refresh-live` → `train --name v5_live2 --final` → `parity`.
Trained models are in `data/bird/<name>/` (`model.cbm`, `schema.json`, `metrics.json`).

### Version history

| Version | What it is | Live result |
|---|---|---|
| `baseline` | All-time best sellers (starter kit) | 0.74 |
| `Leattner.v1–v4` (`personal`) | Weighted signals: session, co-view neighbours, user affinities, context popularity | 1.04 → 1.45. First versions above the baseline |
| `v1-bird` | CatBoost, 5 features | Not served: reference for the dataset contract |
| `v2-bird` | CatBoost, 56 features, no repurchase, scores the whole catalogue | 1.62 (as `v2-bird-norebuy`) |
| `v3-bird` | Heuristic proposes 50 candidates, a YetiRank model orders them | 1.53. Looked better than v2 on the history test and lost to it live: the case that made us switch to the organic test |
| `v4-bird` | Two models: classifier selects 50, ranker with list features orders them | Not served live |
| `v5-bird` | v2's model on dataset v5 (81 volume-invariant features, promotions, trends, own feedback) | ≈ 2.2. New production base |
| `v6-bird` | v5 + per-user state up to the minute (delta) | Tie with v5 |
| `v7-bird` | v6 retrained with more live feedback | Tie with v5 and v6. **Final-round model**: retrained with feedback up to 10:00 CET on Sunday and served alone from 10:24 CET under the label `mj` (2.23 over 16,957 final-round requests) |
| `v8-bird` | v7 + organic purchases in training | 1.92. Lost the A/B, discarded |

Last column: realised purchases per 100 requests in our own feedback while that version was live. Versions that
ran at different hours are not directly comparable; the decisions were taken on simultaneous A/B tests.

## How we used RawTree

- **Features are SQL.** Popularity shares, shrunk rates, co-view neighbours, per-user affinities and
  position-normalised feedback lifts are aggregated in RawTree; we download the result, not the events.
- **Columns are `Dynamic`**, so joins and `argMax` need explicit casts (`product_id::String`,
  `ts::DateTime64(3)`).
- **Our key can only `SELECT` and insert JSON**, so nothing derived lives in RawTree: aggregates are cached
  locally as Parquet and loaded into memory.
- **The A/B is a query.** `team_17.submissions` (which arm answered) joined with `team_17.feedback` (what
  happened) gives conversion per version; `scripts/monitor.py` and `scripts/ab.py` are that join.
- **The public leaderboard API** (`/api/state?by=model`) gives the luck-free conversion of each `model_version`;
  we use it as ground truth for the offline tests.

## How we worked: coding agents

Most of the code was written with coding agents, with us deciding what to try and what to believe.

- **Claude Code** was the main tool (most commits carry its co-author line), and **Codex** agents were used on
  Saturday for parallel side tasks (live evaluator, cold-start and trend analyses, the report script).
- **[`CLAUDE.md`](CLAUDE.md) is the team's shared memory.** Every agent session starts by reading it: repo
  structure, rules (one production instance, casts on `Dynamic` columns, evaluate before deploying), the version
  table and every finding with its number. When someone learned something (repurchase is penalised live; which
  offline test to trust), it went there, and the next person's agent started from it.
- **Rules that kept agents safe around production:** one production instance for the whole team and `make dry`
  for everything else, `SELECT`-only analysis, and each model in its own branch until it wins an A/B.
- **What agents were best at:** exploring the data with SQL, building the feature pipeline twice-consistent
  (train/serve parity), reliability review (`tests/RELIABILITY.md`), and analysis on demand during the event.
- **What stayed with us:** which hypothesis to test next, reading A/B results with their error bars, and every
  decision to change production.

## Team

Two sub-teams. The split is approximate: we helped each other throughout.

| Sub-team | Members | Focus |
|---|---|---|
| Algorithms | Lucas, Joan | Recommendation policies and models, serving and A/B testing |
| Dataset | Santi, Jordi | Data exploration, features and training datasets |

## Run it

Requires Python 3.12 and a `.env` with `TEAM_ID` and `RAWTREE_API_KEY` (copy `.env.example`; never commit it).

```bash
make install                 # virtualenv + dependencies
make check                   # connection and table sizes
python scripts/bird_v5.py state        # download the serving state (~2 min)
make dry POLICY=v7-bird      # answer live requests without writing them
make up POLICY=v7-bird       # production, supervised (only ONE instance per team)
make up AB=v6-bird=v6-bird,v7-bird=v7-bird   # simultaneous A/B
make monitor                 # conversion, on-time rate and latency per version, last 30 min
make offline C="bird_v5"     # offline tests checked against the leaderboard
```

## Repository map

| Path | What |
|---|---|
| `reco/serve.py` | Live loop: pull → recommend → finalize → push; A/B assignment |
| `reco/policies/` | One class per strategy (`fit`, `refresh`, `recommend`); registered in `__init__.py` |
| `reco/bird/` | The Bird models: dataset builders, feature formulas (`feat5.py`), live state (`live5.py`), training |
| `reco/safety.py` | `finalize()`: the last guard before every answer |
| `reco/offline_live.py`, `scripts/offline_vs_live.py` | Offline tests and their reliability against the leaderboard |
| `scripts/bird_v5.py` | Dataset, training, parity and comparison commands for the final model |
| `scripts/supervise.py` | Supervisor: single instance, automatic restart |
| `tests/` | Unit tests (feature invariance, A/B hashing, reliability regressions) |
| `docs/` | Feature documentation, architecture, experiments, demo script |
| `starter/` | The organisers' starter kit, untouched |
