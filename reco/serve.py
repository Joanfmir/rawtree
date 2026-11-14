"""Bucle en vivo: pull (shop.requests) → predict (política) → push (team_NN.submissions).

    python -m reco.serve --policy coview            # en producción
    python -m reco.serve --policy coview --dry-run  # prueba: calcula pero NO envía

Nunca debe caerse: cualquier error en una petición se responde con el fallback.
"""
from __future__ import annotations

import argparse
import time

from . import config
from .client import insert, query
from .policies import POLICIES
from .safety import finalize
from .state import State

NEW_REQUESTS = """
SELECT request_id, user_id, is_new_user, context, recent_views, ts
FROM shop.requests
ORDER BY ts DESC
LIMIT 100"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="baseline", choices=sorted(POLICIES))
    ap.add_argument("--dry-run", action="store_true", help="no inserta en submissions")
    ap.add_argument("--poll", type=float, default=0.3, help="segundos entre lecturas")
    args = ap.parse_args()
    config.check()

    state = State()
    while True:  # sin catálogo no podemos validar: reintenta hasta tenerlo
        try:
            state.refresh()
            break
        except Exception as e:
            print(f"[init] {e}; reintento en 5 s")
            time.sleep(5)

    policy = POLICIES[args.policy]()
    policy.fit(state)
    version = f"{policy.name}{'-dry' if args.dry_run else ''}"
    print(f"[serve] {config.TEAM} · política={policy.name} · dry_run={args.dry_run}")

    seen: set[str] = set()
    refreshed_at, stats_at, answered, errors = 0.0, time.time(), 0, 0
    while True:
        try:
            try:
                state.refresh()
            except Exception as e:  # seguimos con el último estado conocido
                print(f"[state] {e}")
            if time.time() - refreshed_at >= policy.refresh_every:
                refreshed_at = time.time()
                try:
                    policy.refresh(state)
                except Exception as e:
                    print(f"[policy.refresh] {e}")

            fresh = [r for r in query(NEW_REQUESTS) if r["request_id"] not in seen]
            answers = []
            for req in fresh:
                seen.add(req["request_id"])
                try:
                    items = policy.recommend(req, state)
                except Exception as e:
                    errors += 1
                    print(f"[recommend] {req['request_id']}: {e}")
                    items = []
                answers.append({"request_id": req["request_id"], "items": finalize(items, state),
                                "model_version": version})
            if answers:
                if args.dry_run:
                    print(f"[dry-run] {answers[0]}")
                else:
                    insert("submissions", answers)
                answered += len(answers)

            if time.time() - stats_at >= 60:
                print(f"[stats] {answered} respuestas/min · {errors} errores · {len(state.unavailable)} sin stock")
                stats_at, answered, errors = time.time(), 0, 0
            if len(seen) > 200_000:
                seen = set(list(seen)[-50_000:])
        except Exception as e:  # los compradores siguen llegando: no nos caemos nunca
            print(f"[error] {e}")
            time.sleep(1)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
