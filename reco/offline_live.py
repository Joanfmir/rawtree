"""¿Cuánto se parece la evaluación offline al leaderboard? Y, para un modelo nuevo, ¿merece ir al vivo?

    python scripts/offline_vs_live.py                          # fiabilidad con las versiones ya servidas
    python scripts/offline_vs_live.py --candidates personal_v5 # dónde caería una política nueva

Verdad de referencia: la conversión ESPERADA de cada model_version en el leaderboard público
(/api/state?by=model; no tiene suerte: usa la probabilidad del simulador, no la compra real).

Dos tests offline, con las mismas listas que pasarían por finalize():
  TestSet (history)        carruseles de shop.history_sessions desde --until, SIN recompras (con
                           recompras el orden sale al revés: el simulador penaliza recomprar); ips_ndcg.
  OrganicTestSet (organic) peticiones reales de hoy desde un corte; objetivo: compras orgánicas nuevas
                           del usuario en los minutos siguientes; ndcg@10. Features y modelos solo ven lo
                           anterior al corte (features._events incluye shop.events hasta el corte).

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
    "v3-joan": "joan",                    # cuenta en cuanto aparezca en el leaderboard
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

def _features_at(until: str) -> dict[str, pd.DataFrame]:
    from . import features
    if not features.folder(until).exists():
        features.build(until=until)
    return features.load(until=until)


@dataclass
class TestSet:
    """Carruseles de shop.history_sessions desde `until` (sin recompras), como en reco.evaluate.

    Objetivo: el producto comprado en el carrusel del sistema viejo, ponderado por 1/propensity (IPS).
    Usuarios nuevos: no hay en el test, se simulan escondiendo el usuario (segunda pasada)."""
    until: str
    df: pd.DataFrame
    state: object
    known: set
    kind = "history"
    main = "ips_ndcg"          # métrica principal (y "nuevo_" + main para usuarios nuevos)
    passes = (False, True)     # pasada normal y pasada con el usuario escondido

    @classmethod
    def load(cls, until: str) -> TestSet:
        from . import evaluate as ev
        feats = _features_at(until)
        df = ev.drop_rebuys(ev.load_sessions(until), feats["user_purchases"])
        return cls(until, df, ev.make_state(until, feats), set(feats["user_subcat"].user_id))

    @property
    def tag(self) -> str:
        return f"{self.kind}_{self.until}"

    @property
    def n_labels(self) -> int:
        return int((self.df.purchased != "").sum())

    def request(self, row, hide: bool) -> dict:
        return {"request_id": row.request_id, "user_id": None if hide else row.user_id,
                "is_new_user": hide or row.user_id not in self.known,
                "context": row.context, "recent_views": row.recent_views, "ts": row.ts}

    def rows(self, lists: dict[bool, list], new: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(peso, ganancia NDCG, acierto) por fila evaluable; `new` = segmento de usuarios nuevos."""
        return per_row(lists[new], self.df)


@dataclass
class OrganicTestSet(TestSet):
    """Peticiones REALES de hoy (shop.requests) desde `until`, de todos los equipos.

    Objetivo: lo que el usuario compró por su cuenta (shop.events) en los `window` minutos siguientes,
    quitando lo que ya había comprado antes de la petición. No depende de qué carrusel se enseñó, así
    que no hace falta IPS. Los usuarios nuevos son los de verdad (is_new_user de la petición).
    Las features y modelos solo ven lo anterior a `until` (ver features._events)."""
    window: int = 30
    end: str = ""
    kind = "organic"
    main = "ndcg@10"
    passes = (False,)

    @classmethod
    def load(cls, until: str, window: int = 30, end: str | None = None, verbose: bool = True) -> OrganicTestSet:
        from . import evaluate as ev
        from .client import query
        from .config import TEAM
        feats = _features_at(until)
        if end is None:   # la última hora completa con `window` minutos de futuro observados
            last = pd.Timestamp(query("SELECT max(ts::DateTime64(3)) AS t FROM shop.events")[0]["t"])
            end = str((last - pd.Timedelta(minutes=window)).floor("h"))
        CACHE.mkdir(parents=True, exist_ok=True)
        from .features import tag
        cache = CACHE / f"organic_{tag(until)}_{tag(end)}_{window}m.pkl"
        if cache.exists():
            df = pickle.loads(cache.read_bytes())
        else:
            t = time.time()
            buys = pd.DataFrame(query(f"""
                SELECT user_id::String AS user_id, product_id::String AS product_id, ts::DateTime64(3) AS ts,
                       'organic' AS source
                FROM shop.events WHERE event_type::String = 'purchase'
                  AND ts::DateTime64(3) >= '{until}' AND ts::DateTime64(3) < toDateTime64('{end}', 3) + INTERVAL {window} MINUTE
                UNION ALL
                SELECT r.user_id::String, f.purchased::String, f.ts::DateTime64(3), 'carousel'
                FROM {TEAM}.feedback AS f INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
                WHERE f.purchased::String != '' AND f.ts::DateTime64(3) >= '{until}'
                  AND f.ts::DateTime64(3) < '{end}'""", timeout=300))
            reqs = pd.DataFrame(query(f"""
                SELECT request_id::String AS request_id, ts::DateTime64(3) AS ts, user_id::String AS user_id,
                       is_new_user::Bool AS is_new_user, context, recent_views
                FROM shop.requests
                WHERE ts::DateTime64(3) >= '{until}' AND ts::DateTime64(3) < '{end}'
                  AND user_id::String IN (SELECT user_id::String FROM shop.events
                      WHERE event_type::String = 'purchase' AND ts::DateTime64(3) >= '{until}')""", timeout=600))
            df = organic_labels(reqs, buys, feats["user_purchases"], window)
            cache.write_bytes(pickle.dumps(df))
            if verbose:
                print(f"  test orgánico construido en {time.time() - t:.0f}s")
        return cls(until, df, ev.make_state(until, feats), set(feats["user_subcat"].user_id), window, end)

    @property
    def tag(self) -> str:
        from .features import tag
        return f"{self.kind}_{tag(self.until)}_{tag(self.end)}_{self.window}m"

    @property
    def n_labels(self) -> int:
        return len(self.df)

    def request(self, row, hide: bool) -> dict:
        return {"request_id": row.request_id, "user_id": row.user_id, "is_new_user": bool(row.is_new_user),
                "context": row.context, "recent_views": row.recent_views, "ts": row.ts}

    def rows(self, lists: dict[bool, list], new: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        recs = lists[False]
        mask = self.df.is_new_user.to_numpy(bool) if new else np.ones(len(self.df), bool)
        g, h = [], []
        for r, labels, m in zip(recs, self.df.labels, mask):
            if not m:
                continue
            pos = [r.index(p) for p in labels if p in r]
            h.append(1.0 if pos else 0.0)
            g.append(1 / math.log2(min(pos) + 2) if pos else 0.0)
        g, h = np.array(g), np.array(h)
        return np.ones(len(g)), g, h


def organic_labels(reqs: pd.DataFrame, buys: pd.DataFrame, prior: pd.DataFrame, window: int) -> pd.DataFrame:
    """Una fila por petición con al menos una compra orgánica nueva del usuario en (ts, ts + window].

    "Nueva" = el usuario no lo había comprado antes de la petición: ni antes del corte (`prior`, la tabla
    user_purchases de las features) ni entre el corte y la petición (orgánico o en nuestro carrusel)."""
    if reqs.empty:
        return reqs.assign(labels=[])
    reqs = reqs.copy()
    reqs["ts"] = pd.to_datetime(reqs.ts)
    buys = buys.assign(ts=pd.to_datetime(buys.ts)).sort_values("ts")
    owned = prior.groupby(prior.user_id.astype(str)).product_id.apply(lambda s: set(s.astype(str))).to_dict()
    by_user = {u: g for u, g in buys.groupby("user_id")}
    w = pd.Timedelta(minutes=window)
    labels = []
    for u, t in zip(reqs.user_id, reqs.ts):
        g = by_user.get(u)
        if g is None:
            labels.append([])
            continue
        before = set(g.product_id[g.ts <= t]) | owned.get(u, set())
        fut = g[(g.ts > t) & (g.ts <= t + w) & (g.source == "organic")]
        labels.append(list(dict.fromkeys(p for p in fut.product_id if p not in before)))
    reqs["labels"] = labels
    reqs["recent_views"] = reqs.recent_views.apply(lambda v: v or [])
    out = reqs[reqs.labels.str.len() > 0].sort_values("ts").reset_index(drop=True)
    out["ts"] = out.ts.astype(str)
    return out


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
    # modelos de otras carpetas de data/ (p. ej. data/joan/): cuentan si el código de la política las nombra
    code = "".join(f.read_text(encoding="utf-8") for f in files)
    for d in sorted(p for p in DATA_DIR.iterdir() if p.is_dir() and p.name != "bird"):
        if f'"{d.name}"' in code:
            h.update("|".join(f"{m.relative_to(DATA_DIR)}:{m.stat().st_size}:{int(m.stat().st_mtime)}"
                              for m in sorted(d.glob("*/*")) if m.suffix in (".cbm", ".json", ".txt")).encode())
    return h.hexdigest()[:12]


def stale_lists(name: str, tag: str, folder: Path | None = None) -> list[Path]:
    """Cachés de `name` para un test (de cualquier huella). Ojo: "personal" no debe arrastrar a "personal_v3"."""
    pat = re.compile(rf"lists_{re.escape(tag)}_{re.escape(name)}_[0-9a-f]{{12}}\.pkl")
    return [p for p in (folder or CACHE).glob(f"lists_{tag}_*.pkl") if pat.fullmatch(p.name)]


def policy_lists(name: str, test: TestSet, verbose: bool = True) -> dict[bool, list[list[str]]]:
    """Listas finales (tras finalize) para cada fila del test; en el histórico, también con el usuario
    escondido (cold start). Caché por política: se recalcula si cambia su código o sus modelos."""
    from .policies import POLICIES
    from .safety import finalize
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"lists_{test.tag}_{name}_{policy_key(name, test.until)}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    t = time.time()
    policy = POLICIES[name](until=test.until)
    policy.fit(test.state)
    out = {}
    for hide in test.passes:
        out[hide] = [finalize(policy.recommend(test.request(row, hide), test.state), test.state)
                     for row in test.df.itertuples(index=False)]
    for old in stale_lists(name, test.tag):
        old.unlink()
    path.write_bytes(pickle.dumps(out))
    if verbose:
        print(f"  {name} [{test.kind}]: listas calculadas en {time.time() - t:.0f}s")
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


def scores(test: TestSet, lists: dict[bool, list[list[str]]]) -> dict[str, float]:
    """Métrica principal del test (ips_ndcg en el histórico, ndcg@10 en el orgánico), acierto y versión
    para usuarios nuevos (prefijo "nuevo_")."""
    out = {}
    for new, prefix in ((False, ""), (True, "nuevo_")):
        w, g, h = test.rows(lists, new)
        if not len(w):
            continue
        out[f"{prefix}{test.main}"] = float((w * g).sum() / w.sum())
        out[f"{prefix}{'ips_hit' if test.kind == 'history' else 'hit@10'}"] = float((w * h).sum() / w.sum())
        out[f"{prefix}n"] = len(w)
    return out


def bootstrap_diff(test: TestSet, a: dict[bool, list], b: dict[bool, list], new: bool = False,
                   n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Métrica principal de a − la de b, con IC del 95 % por bootstrap emparejado de filas del test."""
    w, ga, _ = test.rows(a, new)
    _, gb, _ = test.rows(b, new)
    if not len(w):
        return float("nan"), float("nan"), float("nan")
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
