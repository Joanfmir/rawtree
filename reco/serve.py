"""Bucle en vivo: pull (shop.requests) → predict (política) → push (team_NN.submissions).

    python -m reco.serve --policy coview            # en producción
    python -m reco.serve --policy coview --dry-run  # prueba: calcula pero NO envía
    python -m reco.serve --ab personal_v1=Leattner.v1,personal=Leattner.v2   # A/B en vivo

Nunca debe caerse: cualquier error en una petición se responde con el fallback.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path

import requests
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


def acquire_lock(path: Path):
    """Conserva abierto el descriptor: flock se libera incluso tras SIGKILL."""
    handle = path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip() or "propietario inicializando"
        handle.close()
        raise SystemExit(f"Instancia ya activa en {path}: {owner}")
    handle.seek(0)
    handle.truncate()
    json.dump({"pid": os.getpid(), "cwd": str(Path.cwd()), "started": time.time()}, handle)
    handle.flush()
    return handle


def send_answers(answers):
    """Solo ConnectTimeout garantiza que no se enviaron bytes; no repetir envíos ambiguos."""
    try:
        insert("submissions", answers, timeout=(0.5, 2.0))   # si tarda más de 2 s, ya llegaría tarde
    except requests.exceptions.ConnectTimeout:
        insert("submissions", answers, timeout=(0.5, 1.5))


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
    # El override permite probar sin tocar el bloqueo de producción.
    lock = None
    if not args.dry_run or os.environ.get("RECO_LOCK_IN_DRY_RUN") == "1":
        path = Path.home() / ".rawtree_team17_serve.lock"
        if args.dry_run:
            path = Path(os.environ.get("RECO_TEST_LOCK", str(path)))
        lock = acquire_lock(path)
    config.check()

    started_at = time.time()
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

    seen: dict[str, float] = {}
    refreshed_at = [0.0] * len(arms)
    stats_at, answered, errors = time.time(), 0, 0
    lags: list[float] = []
    while True:
        try:
            for i, (policy, _) in enumerate(arms):   # cada política refresca a su ritmo
                if time.time() - refreshed_at[i] >= policy.refresh_every:
                    refreshed_at[i] = time.time()
                    try:
                        policy.refresh(state)
                    except Exception as e:
                        print(f"[policy.refresh] {policy.name}: {e}")

            now = time.time()
            seen = {rid: expiry for rid, expiry in seen.items() if expiry > now}
            fresh = []
            for r in query(NEW_REQUESTS, timeout=(0.5, 1.5)):
                if r["request_id"] in seen:
                    continue
                seen[r["request_id"]] = now + 120
                if started_at <= _epoch(r.get("ts")) and time.time() - _epoch(r.get("ts")) < 3.5:   # más viejas ya cayeron al fallback (límite: 3 s; margen por reloj)
                    fresh.append(r)
            answers = []
            for req in fresh:
                policy, version = arms[zlib.crc32(req["request_id"].encode()) % len(arms)]
                try:
                    items = finalize(policy.recommend(req, state), state)
                except Exception as e:
                    errors += 1
                    print(f"[recommend] {req['request_id']}: {e}")
                    try:
                        items = finalize([], state)
                    except Exception as fallback_error:
                        print(f"[fallback] {req['request_id']}: {fallback_error}")
                        continue
                answers.append({"request_id": req["request_id"], "items": items,
                                "model_version": version})
            if answers:
                if args.dry_run:
                    print(f"[dry-run] {answers[0]}")
                else:
                    send_answers(answers)
                answered += len(answers)
                now = time.time()
                sent_ids = {answer["request_id"] for answer in answers}
                lags += [now - _epoch(r.get("ts")) for r in fresh if r["request_id"] in sent_ids]

            if time.time() - stats_at >= 60:
                ok = sorted(lags)
                lat = f"latencia p50 {ok[len(ok) // 2]:.2f}s p95 {ok[int(len(ok) * .95)]:.2f}s" if ok else "sin datos"
                print(f"[stats] {answered} respuestas/min · {lat} · {errors} errores · {len(state.unavailable)} sin stock")
                stats_at, answered, errors, lags = time.time(), 0, 0, []
        except Exception as e:  # los compradores siguen llegando: no nos caemos nunca
            print(f"[error] {e}")
            time.sleep(0.1)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
