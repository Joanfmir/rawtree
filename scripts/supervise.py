"""Supervisor único de serve; control por PID verificado mediante su bloqueo."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reco.serve import acquire_lock

WATCHDOG_SECONDS = 150


def lock_path(dry=False):
    if dry:
        return Path(os.environ.get("RECO_TEST_SUPERVISOR_LOCK", "/tmp/rawtree_team17_supervisor_dry.lock"))
    return Path.home() / ".rawtree_team17_supervisor.lock"


def owner(path):
    try:
        with path.open("r+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return None
            except BlockingIOError:
                handle.seek(0)
                return json.load(handle)
    except (OSError, ValueError):
        return None


def last_stats(path):
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 262144))
            lines = handle.read().decode(errors="replace").splitlines()
        return next((line for line in reversed(lines) if line.startswith("[stats]")), "sin [stats]")
    except OSError:
        return "sin [stats]"


def control(command, dry):
    info = owner(lock_path(dry))
    if not info:
        print("Supervisor inactivo")
    else:
        print(f"Supervisor PID {info['pid']} · serve PID {info.get('child_pid', '?')} · {time.time() - info['started']:.0f} s · {info['cwd']}")
        print(last_stats(Path(info['cwd']) / "logs/supervisor.log"))
        if command == "down":
            # Se mantiene el bloqueo del propietario hasta terminar también el hijo.
            os.kill(info['pid'], signal.SIGTERM)
            for _ in range(100):
                if not owner(lock_path(dry)):
                    print("Supervisor detenido")
                    break
                time.sleep(0.1)
            else:
                raise SystemExit("El supervisor no ha terminado en 10 s")
    if command == "status":
        from reco import config
        from reco.client import query
        try:
            rows = query(f"""SELECT count() AS n,
                countIf(served_by::String = 'team') AS on_time
                FROM {config.TEAM}.feedback
                WHERE request_id::String != '__init__'
                AND ts::DateTime64(3) >= now64(3) - INTERVAL 5 MINUTE""", timeout=(0.5, 2))
            n = int(rows[0]['n'])
            print(f"Últimos 5 min: {100 * int(rows[0]['on_time']) / n:.2f}% a tiempo ({n} respuestas)" if n else "Últimos 5 min: sin feedback")
        except Exception as exc:
            print(f"Feedback no disponible: {exc}")


def main():
    args = sys.argv[1:]
    dry = "--dry-run" in args
    if args and args[0] in {"down", "status"}:
        control(args[0], dry)
        return
    os.chdir(ROOT)
    lock = acquire_lock(lock_path(dry))
    (ROOT / "logs").mkdir(exist_ok=True)
    supervisor_started = time.time()
    child = None
    awake = None
    stopping = False

    def stop(_sig, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def terminate(proc):
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    with (ROOT / "logs/supervisor.log").open("a", buffering=1) as log:
        def report(message):
            log.write(f"[supervisor] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
        try:
            if sys.platform == "darwin":
                awake = subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())], stdout=log, stderr=log)
            failures = 0
            while not stopping:
                child = subprocess.Popen([sys.executable, "-u", "-m", "reco.serve", *args], cwd=ROOT, stdout=log, stderr=log)
                lock.seek(0)
                lock.truncate()
                json.dump({"pid": os.getpid(), "cwd": str(ROOT), "started": supervisor_started,
                           "child_pid": child.pid}, lock)
                lock.flush()
                report(f"serve PID {child.pid} iniciado")
                started = heartbeat = time.monotonic()
                # Solo se leen bytes de esta ejecución: un log antiguo no prueba salud.
                with (ROOT / "logs/supervisor.log").open() as reader:
                    reader.seek(0, 2)
                    while not stopping and child.poll() is None:
                        if any(line.startswith("[stats]") for line in reader.readlines()):
                            heartbeat = time.monotonic()
                        if time.monotonic() - heartbeat > WATCHDOG_SECONDS:
                            report(f"serve PID {child.pid} sin [stats] en {WATCHDOG_SECONDS} s; reinicio")
                            terminate(child)
                            break
                        time.sleep(0.2)
                if stopping:
                    break
                # Un conflicto de bloqueo es deliberado: no competir con otra instancia.
                if child.returncode == 1 and owner(Path.home() / ".rawtree_team17_serve.lock") and not dry:
                    report("Otra instancia de serve tiene el bloqueo; supervisor finalizado")
                    break
                if time.monotonic() - started > 150:
                    failures = 0
                delay = (1, 2, 5)[min(failures, 2)]
                failures += 1
                report(f"serve terminado ({child.returncode}); reintento en {delay} s")
                deadline = time.monotonic() + delay
                while not stopping and time.monotonic() < deadline:
                    time.sleep(0.1)
        finally:
            terminate(child)
            terminate(awake)
            report("Supervisor detenido")
            lock.close()


if __name__ == "__main__":
    main()
