"""Entrena el re-ranker hgb (reco/ranker.py) y lo guarda en data/models/.

    python scripts/train_hgb.py                       # producción: últimas 3 semanas del log
    python scripts/train_hgb.py --until 2026-11-07    # para evaluar offline (no ve el test)
    python scripts/train_hgb.py --until 2026-11-07 --eval   # entrena y compara con la base

Hay que relanzarlo cada vez que cambie la política base (señales o pesos). Si serve está corriendo
con --policy hgb o --ab hgb, recoge el modelo nuevo solo, sin reiniciar.
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reco import config, ranker  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--until", default=None, help="YYYY-MM-DD: entrena solo con el log anterior (para evaluar)")
ap.add_argument("--weeks", type=int, default=3, help="semanas de log para entrenar")
ap.add_argument("--base", default="personal", help="política base (debe tener signals())")
ap.add_argument("--target", default=None, choices=["mixed", "click", "purchase"], help="qué cuenta como acierto")
ap.add_argument("--ips-clip", type=float, default=None, help="peso 1/max(propensity, clip); 0 = sin corrección")
ap.add_argument("--eval", action="store_true", help="después, evalúa base y hgb (requiere --until)")
args = ap.parse_args()
config.check()

fit_args = {k: v for k, v in (("target", args.target), ("ips_clip", args.ips_clip)) if v is not None}
print(f"Entrenando hgb sobre {args.base} · semanas {ranker.week_starts(args.until, args.weeks)} · {fit_args or 'opciones por defecto'}")
rk = ranker.train(until=args.until, weeks=args.weeks, base_name=args.base, **fit_args)
print(f"Guardado en {rk.save(args.until)}  ·  {rk.meta}")

if args.eval:
    if not args.until:
        sys.exit("--eval necesita --until (el modelo de producción ya ha visto el test)")
    subprocess.run([sys.executable, "-m", "reco.evaluate", "--policies", args.base, "hgb", "--until", args.until],
                   cwd=config.ROOT, check=True)
