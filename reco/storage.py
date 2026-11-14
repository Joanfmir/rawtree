"""Guarda cada decisión (petición + respuesta + latencia) para depurar y entrenar.

Dos destinos:
  1. team_17.scratch en RawTree (kind = 'decision'): compartido por el equipo, se cruza con feedback por SQL.
  2. logs/decisions.jsonl en local: copia de seguridad, siempre se escribe.

Se escribe en un hilo de fondo: guardar NUNCA retrasa ni rompe el envío de respuestas.
"""
from __future__ import annotations

import json
import queue
import threading

from .client import Record, insert
from .config import LOGS_DIR

_q: queue.Queue[list[Record]] = queue.Queue(maxsize=1000)
_remote = True


def start(remote: bool = True) -> None:
    """remote=False (dry-run): solo fichero local, no ensucia scratch."""
    global _remote
    _remote = remote
    LOGS_DIR.mkdir(exist_ok=True)
    threading.Thread(target=_worker, daemon=True).start()


def save(records: list[Record]) -> None:
    try:
        _q.put_nowait(records)
    except queue.Full:  # si RawTree va muy lento, preferimos perder logs a bloquear
        print("[storage] cola llena, descarto logs")


def _worker() -> None:
    path = LOGS_DIR / "decisions.jsonl"
    while True:
        batch = _q.get()
        try:
            with path.open("a") as f:
                for r in batch:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[storage] local: {e}")
        if _remote:
            try:
                insert("scratch", batch)
            except Exception as e:
                print(f"[storage] scratch: {e}")
