"""Bucle en vivo: pull (shop.requests) → predict (política) → push (team_NN.submissions).

    python -m reco.serve --policy coview            # en producción
    python -m reco.serve --policy coview --dry-run  # prueba: calcula pero NO envía
    python -m reco.serve --policy personal --ab hgb --ab-share 0.2   # test A/B en una sola instancia

Test A/B: cada petición va a la política A (--policy) o a la B (--ab) según un hash estable de su
request_id; --ab-share es la fracción que va a B. La versión que contestó queda en model_version,
y scripts/monitor.py compara las compras de cada una.

Nunca debe caerse: cualquier error en una petición se responde con el fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import threading
import time
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


def in_b(request_id: str, share: float) -> bool:
    """True si la petición va a la política B. Estable: la misma petición cae siempre en el mismo lado."""
    h = int(hashlib.md5(str(request_id).encode()).hexdigest()[:8], 16)
    return h / 2**32 < share


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="baseline", choices=sorted(POLICIES))
    ap.add_argument("--dry-run", action="store_true", help="no inserta en submissions")
    ap.add_argument("--name", help="model_version que verá el leaderboard (por defecto, el nombre de la política)")
    ap.add_argument("--ab", default=None, choices=sorted(POLICIES), help="política B para el test A/B")
    ap.add_argument("--ab-share", type=float, default=0.5, help="fracción de peticiones que va a B")
    ap.add_argument("--ab-name", help="model_version de B (por defecto, el nombre de su política)")
    ap.add_argument("--poll", type=float, default=0.3, help="segundos entre lecturas")
    args = ap.parse_args()
    if args.ab == args.policy:
        ap.error("--ab debe ser una política distinta de --policy")
    if not 0 < args.ab_share < 1:
        ap.error("--ab-share debe estar entre 0 y 1")
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
    # policies[0] es A (la principal); policies[1], si existe, es B
    policies = [POLICIES[name]() for name in (args.policy, args.ab) if name]
    for p in policies:
        p.share(policies)   # p. ej. hgb reutiliza la instancia de personal en vez de cargar otra
    for p in policies:
        p.fit(state)
    suffix = "-dry" if args.dry_run else ""
    names = [args.name or policies[0].name] + ([args.ab_name or policies[1].name] if args.ab else [])
    versions = [f"{n}{suffix}" for n in names]
    ab = f" · B={policies[1].name} como {versions[1]} ({args.ab_share:.0%})" if args.ab else ""
    print(f"[serve] {config.TEAM} · política={policies[0].name} · versión={versions[0]}{ab} · dry_run={args.dry_run}")

    seen: set[str] = set()
    stats_at, errors = time.time(), 0
    refreshed_at = [0.0] * len(policies)
    answered = [0] * len(policies)
    lags: list[float] = []
    while True:
        try:
            for i, p in enumerate(policies):
                if time.time() - refreshed_at[i] >= p.refresh_every:
                    refreshed_at[i] = time.time()
                    try:
                        p.refresh(state)
                    except Exception as e:
                        print(f"[{p.name}.refresh] {e}")

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
                i = 1 if args.ab and in_b(req["request_id"], args.ab_share) else 0
                try:
                    items = policies[i].recommend(req, state)
                except Exception as e:
                    errors += 1
                    print(f"[recommend:{policies[i].name}] {req['request_id']}: {e}")
                    items = []
                    if i == 1:  # si falla B, contesta A (y se apunta como A)
                        i = 0
                        try:
                            items = policies[0].recommend(req, state)
                        except Exception as e2:
                            print(f"[recommend:{policies[0].name}] {req['request_id']}: {e2}")
                answered[i] += 1
                answers.append({"request_id": req["request_id"], "items": finalize(items, state),
                                "model_version": versions[i]})
            if answers:
                if args.dry_run:
                    print(f"[dry-run] {answers[0]}")
                else:
                    insert("submissions", answers)
                now = time.time()
                lags += [now - _epoch(r.get("ts")) for r in fresh]

            if time.time() - stats_at >= 60:
                ok = sorted(lags)
                lat = f"latencia p50 {ok[len(ok) // 2]:.2f}s p95 {ok[int(len(ok) * .95)]:.2f}s" if ok else "sin datos"
                split = " + ".join(f"{n} {v}" for n, v in zip(answered, versions))
                print(f"[stats] {split} respuestas/min · {lat} · {errors} errores · {len(state.unavailable)} sin stock")
                stats_at, errors, lags, answered = time.time(), 0, [], [0] * len(policies)
            if len(seen) > 200_000:
                seen = set(list(seen)[-50_000:])
        except Exception as e:  # los compradores siguen llegando: no nos caemos nunca
            print(f"[error] {e}")
            time.sleep(1)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
