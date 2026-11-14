"""CLI del evaluador vivo: python scripts/live_eval.py --policies personal_v1 personal gulias --last 20000."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reco.live_eval import main

if __name__ == '__main__':
    main()
