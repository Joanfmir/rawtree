"""Calcula las tablas de features en RawTree y las guarda en data/.

    python scripts/build_features.py                    # live: historia + stream (para servir)
    python scripts/build_features.py --until 2026-11-07 # offline: solo historia antes del corte (para evaluar)
    python scripts/build_features.py --only item_features user_purchases
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reco import config, features  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--until", help="YYYY-MM-DD: usar solo historia anterior (evaluación offline)")
ap.add_argument("--only", nargs="*", help="calcular solo estas tablas")
args = ap.parse_args()
config.check()

t = time.time()
print(f"Features {'hasta ' + args.until if args.until else 'live'} → {features.folder(args.until)}")
features.build(until=args.until, only=args.only)
print(f"Hecho en {time.time() - t:.0f}s")
