"""Bucle en vivo: pull (shop.requests) → predict (política) → push (team_NN.submissions).

    python -m reco.serve --policy coview            # en producción
    python -m reco.serve --policy coview --dry-run  # prueba: calcula pero NO envía
    python -m reco.serve --ab personal_v1=Leattner.v1,personal=Leattner.v2   # A/B en vivo

Nunca debe caerse: cualquier error en una petición se responde con el fallback.
"""
from __future__ import annotations

import argparse
import threading
import time
import zlib
from datetime import datetime, timezone

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


def _epoch(ts) -> float:
    """ts de RawTree ('YYYY-MM-DD HH:MM:SS.fff…', UTC) a segundos."""
    try:
        return datetime.fromisoformat(str(ts)[:23]).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return time.time()


def _keep_state_fresh(state: State) -> None:
    """Stock (cada 1 s) y catálogo (cada 30 s) en un hilo aparte: no añaden latencia al bucle."""
    while True:
        try:
            state.refresh()
        except Exception as e:  # seguimos con el último estado conocido
            print(f"[state] {e}")
        time.sleep(0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="baseline", choices=sorted(POLICIES))
    ap.add_argument("--dry-run", action="store_true", help="no inserta en submissions")
    ap.add_argument("--name", help="model_version que verá el leaderboard (por defecto, el nombre de la política)")
    ap.add_argument("--ab", help="A/B en vivo: 'politica=Nombre,politica=Nombre'. Cada petición va a un brazo fijo "
                                 "según su request_id (reparto a partes iguales). Sustituye a --policy/--name")
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

    threading.Thread(target=_keep_state_fresh, args=(state,), daemon=True).start()
    spec = args.ab or f"{args.policy}={args.name or args.policy}"
    arms = []   # [(política, model_version)]
    for part in spec.split(","):
        pol_name, _, label = part.strip().partition("=")
        policy = POLICIES[pol_name]()
        policy.fit(state)
        arms.append((policy, f"{label or pol_name}{'-dry' if args.dry_run else ''}"))
    print(f"[serve] {config.TEAM} · " + " | ".join(f"{p.name}→{v}" for p, v in arms) + f" · dry_run={args.dry_run}")

    seen: set[str] = set()
    refreshed_at, stats_at, answered, errors = 0.0, time.time(), 0, 0
    lags: list[float] = []
    while True:
        try:
            if time.time() - refreshed_at >= min(p.refresh_every for p, _ in arms):
                refreshed_at = time.time()
                for policy, _ in arms:
                    try:
                        policy.refresh(state)
                    except Exception as e:
                        print(f"[policy.refresh] {e}")

            now = time.time()
            fresh = []
            for r in query(NEW_REQUESTS):
                if r["request_id"] in seen:
                    continue
                seen.add(r["request_id"])
                if now - _epoch(r.get("ts")) < 4:   # más viejas ya cayeron al fallback (límite: 3 s)
                    fresh.append(r)
            answers = []
            for req in fresh:
                policy, version = arms[zlib.crc32(req["request_id"].encode()) % len(arms)]
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
                now = time.time()
                lags += [now - _epoch(r.get("ts")) for r in fresh]

            if time.time() - stats_at >= 60:
                ok = sorted(lags)
                lat = f"latencia p50 {ok[len(ok) // 2]:.2f}s p95 {ok[int(len(ok) * .95)]:.2f}s" if ok else "sin datos"
                print(f"[stats] {answered} respuestas/min · {lat} · {errors} errores · {len(state.unavailable)} sin stock")
                stats_at, answered, errors, lags = time.time(), 0, 0, []
            if len(seen) > 200_000:
                seen = set(list(seen)[-50_000:])
        except Exception as e:  # los compradores siguen llegando: no nos caemos nunca
            print(f"[error] {e}")
            time.sleep(1)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
