"""¿Cuánto se parece la evaluación offline al leaderboard? Y, para un modelo nuevo, ¿merece ir al vivo?

    python scripts/offline_vs_live.py                          # fiabilidad con las versiones ya servidas
    python scripts/offline_vs_live.py --candidates personal_v5 # dónde caería una política nueva

Verdad de referencia: la conversión ESPERADA de cada model_version en el leaderboard público
(/api/state?by=model; no tiene suerte: usa la probabilidad del simulador, no la compra real).
Offline: los carruseles de shop.history_sessions desde --until, SIN recompras (con recompras el orden
sale al revés: el simulador penaliza recomprar), con las mismas listas que pasarían por finalize().

Hallazgo del 14/11 (8 versiones): ips_ndcg sin recompras ordena bien 26 de 28 parejas (Spearman 0,95),
pero el único fallo claro fue arriba: v3-bird (mezcla) 0,260 > v2-bird-norebuy 0,207 offline y
1,47 < 1,79 en el leaderboard. Sirve para descartar; entre dos modelos buenos, decide el vivo.
"""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import pickle
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, ROOT

LEADERBOARD = "https://datathon.rawtree.dev/api/state?by=model"
CACHE = DATA_DIR / "offline_live"

# model_version del leaderboard → política que lo reproduce offline (POLICIES). None = no sabemos qué
# código servía exactamente: se enseña pero no cuenta para la fiabilidad. AL SUBIR UNA VERSIÓN NUEVA AL
# VIVO, AÑÁDELA AQUÍ: cada versión más hace la medida de fiabilidad más sólida.
LIVE_VERSIONS: dict[str, str | None] = {
    "baseline": "baseline",
    "Leattner.v1": "personal_v1",
    "Leattner.v2": "personal",            # aprox.: personal ha cambiado algo desde entonces
    "Leattner.v2-gulias": "gulias",
    "Leattner.v2-cat": None,
    "Leattner.v3": "personal_v3",         # aprox.: v3 se retocó a mitad de su tiempo en vivo
    "Leattner.v4": "personal_v4",
    "Bird.v2": None,                      # bird_v2 anterior al modelo sin recompra
    "v2-bird-norebuy": "bird_v2",
    "v3-bird": "bird_blend_rank",
}


# ───────────────────────────── leaderboard ─────────────────────────────

def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def _team_versions(state: dict, team: str) -> tuple[list[dict], bool]:
    for t in state.get("teams", []):
        if t.get("team_id") == team:
            vs = t.get("versions") or [{k: t.get(k) for k in ("model_version", "requests", "conversion",
                                                              "cold_conversion", "revenue_per_request")}]
            return vs, bool(t.get("more_versions"))
    return [], False


def windows(start: float, end: float, max_span: int = 10800, step: int = 300) -> list[tuple[int, int]]:
    """Trocea [start, end] en ventanas contiguas de como mucho `max_span` s, alineadas a `step`
    (el API solo acepta de 15 min a 3 h)."""
    start, end = int(start // step * step), int(end // step * step)
    if end - start < 900:
        return []
    n = math.ceil((end - start) / max_span)
    size = math.ceil((end - start) / n / step) * step
    edges = list(range(start, end, size)) + [end]
    out = list(zip(edges[:-1], edges[1:]))
    if len(out) > 1 and out[-1][1] - out[-1][0] < 900:   # la última no puede quedar por debajo de 15 min
        out[-2:] = [(out[-2][0], out[-1][1])]
    return out


def fetch_leaderboard(team: str, verbose: bool = True) -> pd.DataFrame:
    """Conversión esperada por model_version de `team` desde el inicio de la práctica.
    Combina ventanas de 3 h ponderando por peticiones. Guarda la última copia en data/ (el domingo el
    leaderboard se apaga y se usa la copia)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"leaderboard_{team}.json"
    try:
        head = _get(LEADERBOARD)
        rows = []
        todo = windows(head["practice_start"], head["updated_at"])
        while todo:
            a, b = todo.pop(0)
            vs, more = _team_versions(_get(f"{LEADERBOARD}&from={a}&to={b}"), team)
            if more and b - a >= 1800:      # más de 10 versiones en la ventana: partirla
                mid = (a + b) // 2 // 300 * 300
                todo[:0] = [(a, mid), (mid, b)]
                continue
            rows += [dict(v, window_from=a, window_to=b) for v in vs]
        if not rows:
            raise RuntimeError(f"el leaderboard no tiene versiones de {team}")
        cache.write_text(json.dumps({"fetched_at": time.time(), "rows": rows}))
    except Exception as e:          # sin red o leaderboard oculto: última copia
        if not cache.exists():
            raise
        saved = json.loads(cache.read_text())
        rows = saved["rows"]
        if verbose:
            age = (time.time() - saved["fetched_at"]) / 3600
            print(f"[leaderboard] sin acceso ({e}); uso la copia de hace {age:.1f} h")
    return combine(pd.DataFrame(rows))


def combine(rows: pd.DataFrame) -> pd.DataFrame:
    """Une las ventanas: medias ponderadas por peticiones (y por peticiones de usuarios nuevos)."""
    df = rows.copy()
    df["cold_requests"] = df.get("cold_requests", pd.Series(0, index=df.index)).fillna(0)
    df["_b"] = df.requests * df.conversion
    df["_cb"] = df.cold_requests * df.cold_conversion.fillna(0)
    df["_rev"] = df.requests * df.revenue_per_request.fillna(0)
    g = df.groupby("model_version")[["requests", "_b", "cold_requests", "_cb", "_rev"]].sum()
    out = pd.DataFrame({"requests": g.requests.astype(int), "conv": g._b / g.requests,
                        "cold_requests": g.cold_requests.astype(int),
                        "cold_conv": (g._cb / g.cold_requests.replace(0, np.nan)),
                        "rev": g._rev / g.requests})
    # error estándar binomial: conservador (el leaderboard usa probabilidades, que tienen menos ruido)
    p = out.conv / 100
    out["se"] = 100 * np.sqrt(p * (1 - p) / out.requests)
    return out.sort_values("conv", ascending=False)


# ───────────────────────────── offline ─────────────────────────────

@dataclass
class TestSet:
    """Carruseles de test (sin recompras) y el estado con el que se evalúa, como en reco.evaluate."""
    until: str
    df: pd.DataFrame
    state: object
    known: set

    @classmethod
    def load(cls, until: str) -> TestSet:
        from . import evaluate as ev
        from . import features
        if not features.folder(until).exists():
            features.build(until=until)
        feats = features.load(until=until)
        df = ev.drop_rebuys(ev.load_sessions(until), feats["user_purchases"])
        return cls(until, df, ev.make_state(until, feats), set(feats["user_subcat"].user_id))


def _module_file(name: str) -> Path | None:
    p = ROOT / Path(*name.split("."))
    for cand in (p.with_suffix(".py"), p / "__init__.py"):
        if cand.exists():
            return cand
    return None


def _reco_imports(path: Path) -> set[Path]:
    """Ficheros de reco/ que importa `path`, de forma transitiva (para invalidar la caché)."""
    seen: set[Path] = set()
    todo = [path]
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        pkg = ".".join(f.relative_to(ROOT).with_suffix("").parts)
        if f.name == "__init__.py":
            pkg = pkg.rsplit(".", 1)[0] + ".__init__"
        base = pkg.split(".")[:-1]
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level:
                    root = base[:len(base) - node.level + 1]
                    mod = ".".join(root + ([mod] if mod else []))
                names = [mod] + [f"{mod}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            for n in names:
                if n.startswith("reco") and (m := _module_file(n)):
                    todo.append(m)
    return seen


def _data_fingerprint(until: str) -> list[str]:
    """Modelos y features que pueden cambiar sin tocar el código (fecha de modificación y tamaño)."""
    files = sorted(DATA_DIR.glob(f"*until_{until}*")) + sorted(DATA_DIR.glob(f"features_until_{until}/*")) \
        + sorted(DATA_DIR.glob("bird/*/model.cbm"))
    return [f"{f.relative_to(DATA_DIR)}:{f.stat().st_size}:{int(f.stat().st_mtime)}" for f in files if f.is_file()]


def policy_key(name: str, until: str) -> str:
    """Huella de una política: su código (y lo que importa), evaluate/safety y los modelos en data/."""
    import inspect
    from .policies import POLICIES
    cls = POLICIES[name]
    files = set()
    for c in cls.__mro__:
        if c.__module__.startswith("reco") and (f := inspect.getsourcefile(c)):
            files |= _reco_imports(Path(f).resolve())
    files |= _reco_imports((ROOT / "reco" / "evaluate.py").resolve())
    h = hashlib.sha1(f"{name}|{until}".encode())
    for f in sorted(files):
        h.update(str(f.relative_to(ROOT)).encode())
        h.update(f.read_bytes())
    h.update("|".join(_data_fingerprint(until)).encode())
    return h.hexdigest()[:12]


def stale_lists(name: str, until: str, folder: Path | None = None) -> list[Path]:
    """Cachés de `name` (de cualquier huella). Ojo: "personal" no debe arrastrar a "personal_v3"."""
    pat = re.compile(rf"lists_{re.escape(until)}_{re.escape(name)}_[0-9a-f]{{12}}\.pkl")
    return [p for p in (folder or CACHE).glob(f"lists_{until}_*.pkl") if pat.fullmatch(p.name)]


def policy_lists(name: str, test: TestSet, verbose: bool = True) -> dict[bool, list[list[str]]]:
    """Listas finales (tras finalize) para cada carrusel de test, con el usuario y sin él (cold start).
    Se guardan en caché; se recalculan solas si cambia el código de la política o sus modelos."""
    from .policies import POLICIES
    from .safety import finalize
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"lists_{test.until}_{name}_{policy_key(name, test.until)}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    t = time.time()
    policy = POLICIES[name](until=test.until)
    policy.fit(test.state)
    out = {}
    for hide in (False, True):
        recs = []
        for row in test.df.itertuples(index=False):
            req = {"request_id": row.request_id, "user_id": None if hide else row.user_id,
                   "is_new_user": hide or row.user_id not in test.known,
                   "context": row.context, "recent_views": row.recent_views, "ts": row.ts}
            recs.append(finalize(policy.recommend(req, test.state), test.state))
        out[hide] = recs
    for old in stale_lists(name, test.until):
        old.unlink()
    path.write_bytes(pickle.dumps(out))
    if verbose:
        print(f"  {name}: listas calculadas en {time.time() - t:.0f}s")
    return out


def per_row(recs: list[list[str]], df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Por carrusel con compra: peso IPS (1/propensity del slot donde se enseñó), ganancia NDCG y acierto.
    ips_ndcg = sum(w·g) / sum(w); es la misma cuenta que reco.evaluate._metrics."""
    mask = (df.purchased != "").to_numpy()
    tgt = df.purchased.to_numpy()[mask]
    rr = [r for r, m in zip(recs, mask) if m]
    pos = np.array([r.index(t) if t in r else -1 for r, t in zip(rr, tgt)])
    props = np.array([p[s.index(t)] for p, s, t in zip(df.propensity.to_numpy()[mask], df.shown.to_numpy()[mask], tgt)])
    w = 1 / np.clip(props.astype(float), 1e-3, None)
    hit = pos >= 0
    gain = np.where(hit, 1 / np.log2(np.maximum(pos, 0) + 2), 0.0)
    return w, gain, hit.astype(float)


def scores(lists: dict[bool, list[list[str]]], df: pd.DataFrame) -> dict[str, float]:
    out = {}
    for hide, prefix in ((False, ""), (True, "nuevo_")):
        w, g, h = per_row(lists[hide], df)
        out[f"{prefix}ips_ndcg"] = float((w * g).sum() / w.sum())
        out[f"{prefix}ips_hit"] = float((w * h).sum() / w.sum())
        out[f"{prefix}hit@10"] = float(h.mean())
    return out


def bootstrap_diff(a: dict[bool, list], b: dict[bool, list], df: pd.DataFrame, hide: bool = False,
                   n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """ips_ndcg(a) − ips_ndcg(b) con IC del 95 % por bootstrap emparejado de carruseles."""
    w, ga, _ = per_row(a[hide], df)
    _, gb, _ = per_row(b[hide], df)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(w), size=(n, len(w)))
    ws = w[idx]
    diffs = ((ws * ga[idx]).sum(1) - (ws * gb[idx]).sum(1)) / ws.sum(1)
    point = float(((w * ga).sum() - (w * gb).sum()) / w.sum())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)


# ───────────────────────────── acuerdo offline ↔ vivo ─────────────────────────────

def agreement(names: list[str], offline: np.ndarray, live: np.ndarray, se: np.ndarray) -> dict:
    """Qué tal ordena la métrica offline a las versiones según el leaderboard.
    Parejas "claras": las que el leaderboard separa por más de 2 errores estándar combinados."""
    from scipy.stats import kendalltau, spearmanr
    ok = n = ok_c = n_c = 0
    wrong = []
    for i, j in itertools.combinations(range(len(names)), 2):
        same = (offline[i] - offline[j]) * (live[i] - live[j]) > 0
        clear = abs(live[i] - live[j]) > 2 * math.hypot(se[i], se[j])
        n += 1
        ok += same
        n_c += clear
        ok_c += same and clear
        if not same:
            wrong.append((names[i], names[j], clear))
    k = len(names)
    return {"n": k,
            "spearman": float(spearmanr(offline, live)[0]) if k > 2 else float("nan"),
            "kendall": float(kendalltau(offline, live)[0]) if k > 2 else float("nan"),
            "pairs": (ok, n), "clear_pairs": (ok_c, n_c), "wrong": wrong}
