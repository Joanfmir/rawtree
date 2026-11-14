"""Política personalizada: suma de señales por producto candidato.

Candidatos y puntuación (los pesos están en DEFAULT_WEIGHTS y se ajustan con reco.evaluate):
  self      el propio producto que acaba de mirar (el 16 % de las compras del log son de algo recién visto)
  buy       vecinos "miró p → compró / añadió al carrito q" de cada producto visto
  view      vecinos "vistos en la misma sesión"
  user      top productos de las subcategorías favoritas del usuario
  ctx       más comprados en su (device, region) en las últimas 2 semanas: clave para usuarios nuevos
  pop       popularidad de los últimos 7 días (prior pequeño para todos)
Multiplicadores:
  brand     × (1 + brand · afinidad del usuario por la marca)
  rebuy     productos que el usuario ya compró: el 62 % de las compras del test son repeticiones y un
            producto ya comprado se compra 6,5× más. Peso (0,5 + 0,5·min(n,4)/4) · exp(-días / rebuy_days)
Categoría de la sesión (v2): en vivo, un producto de una categoría que el cliente acaba de mirar
convierte ~5× más que uno de otra categoría, y el sistema viejo solo enseñaba la categoría de la sesión.
  cat_fill  añade como candidatos los más populares de las categorías vistas (relleno dentro de categoría)
  off       multiplicador para productos de categorías NO vistas (0 = se descartan; 1 = sin efecto, v1)
  cat_decay peso de la categoría de la vista k-ésima más antigua: cat_decay^k (1 = todas igual)
Sin vistas (home) no se aplica.
Las vistas recientes pesan más cuanto más recientes: × decay^k (k = 0 la última).
"""
from __future__ import annotations

import math
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .base import Policy
from ..ranker import _views

# v1: ajustados con scripts/tune.py search (ips_ndcg, corte 2026-11-07). Sin lógica de categoría.
V1 = {
    "self": 0.15, "buy": 1.0, "view": 0.1, "user": 1.0, "ctx": 0.15, "pop": 0.4,
    "brand": 0.5, "rebuy": 1.0, "rebuy_days": 30.0, "decay": 0.6,
    "cat_fill": 0.0, "off": 1.0, "cat_decay": 1.0,
}
# v2: prioriza las categorías de la sesión.
V2 = {**V1, "cat_fill": 0.5, "off": 0.0}
# v3: lo que predice la compra EN VIVO (197.670 impresiones nuestras, 14/11). Dentro de la categoría vista
# manda la calidad general del producto (el cuartil alto de popularidad/tasa de compra convierte ~7-10× el
# bajo), lo recién visto duplica y los vecinos de co-compra ayudan; la afinidad del usuario (subcategoría,
# marca) y la recompra orgánica no aportan. Así que: categoría (ponderada por su peso en las vistas),
# populares de esa categoría, recién visto y co-compra; sin señales de perfil.
V3 = {**V2, "user": 0.0, "brand": 0.0, "rebuy": 0.0, "cat_fill": 2.0, "self": 1.0, "buy": 1.0, "view": 0.1,
      "ctx": 0.3, "pop": 0.4, "cat_share": 1.0,
      # historial por producto (ver recommend): lo comprado en carrusel va arriba del todo
      "hist_carousel": 4.0, "hist_cart": 1.5, "hist_click": 1.2, "organic": 0.1}
# v4: v3 + dos cambios respaldados por el feedback en vivo (14/11, 39.341 peticiones).
#   organic = 0  lo comprado solo de forma orgánica se descarta: en vivo, 2 compras en 24.457 impresiones
#                (0,04-0,18× lo esperado por posición). Offline sin recompras: ndcg 0,416 → 0,422.
#   fb           × lift^fb: lift del producto en nuestro propio carrusel (features.feedback_lift).
#                Validado con partición temporal: lift < 0,6 → convierte 0,6×; lift > 1,1 → 1,2×.
#                Exponente < 1 porque el lift predicho exagera algo el real.
# Probado y DESCARTADO (el código queda, con peso 0):
#   cat_recency  pesar las categorías por recencia (r^k). Offline mejora (ndcg 0,416 → 0,430), pero es un
#                sesgo del log (el sistema viejo solo enseñaba la categoría de la última vista): en vivo,
#                con 2+ categorías vistas, la de una vista ANTERIOR convierte 1,98× y la de la última 1,17×.
#   sub_fill     más vendidos de la subcategoría vista. Peor offline (0,416 → 0,381) y en vivo la
#                subcategoría de la última vista no convierte más que el resto de su categoría.
V4 = {**V3, "organic": 0.0, "fb": 0.7, "cat_recency": 0.0, "sub_fill": 0.0}
# v5: v4 + dos correcciones (feedback en vivo del 14/11, 37.805 peticiones):
#   hist_off     el filtro de categoría (off = 0) ya no borra lo que el usuario tiene en su historial
#                (user_items): fuera de la categoría vista lo multiplica por hist_off en lugar de por 0.
#                En vivo, lo ya comprado en carrusel de OTRA categoría convirtió 0,44 % (9 compras en 2.035
#                impresiones) frente al 0,15 % del candidato normal de la categoría vista; ~2× a igual
#                posición. Evidencia débil (9 compras): por eso 0,5 y no 1, para no quitarle el slot 1 a la sesión.
#   live_rebuy   0 = no se vuelve a ofrecer a un usuario lo que NUESTRO carrusel ya le vendió en vivo
#                (README: quien ya compró algo tiende a no quererlo otra vez). user_items mete esas compras
#                en compras_carrusel, así que en v3/v4 subían a lo más alto (hist_carousel = 4).
#                Se refresca cada 2 min desde team.feedback, sin esperar al rebuild de features (10 min).
#   top          devuelve 30 candidatos: finalize() rellena con los nuestros antes que con best sellers.
V5 = {**V4, "hist_off": 0.5, "live_rebuy": 0.0, "top": 30}
VERSIONS = {"v1": V1, "v2": V2, "v3": V3, "v4": V4, "v5": V5}
DEFAULT_WEIGHTS = V2


def _norm_lists(df, key, item, val, top):
    """{key: [(item, val / max_val_of_key), ...]} ordenado por val desc, máximo `top` por clave."""
    df = df.dropna(subset=[key, item, val])
    df = df[np.isfinite(df[val].astype(float))]
    df = df.sort_values([key, val], ascending=[True, False])
    out: dict = {}
    cur, lst, vmax = object(), None, 1.0
    for k, i, v in zip(df[key].tolist(), df[item].tolist(), df[val].tolist()):
        if k != cur:
            cur, lst, vmax = k, [], max(float(v), 1e-9)
            out[k] = lst
        if len(lst) < top:
            lst.append((i, v / vmax))
    return out


def _group(keys, *cols):
    """{key: [(col1, col2, ...), ...]} conservando el orden de las filas."""
    out: dict = {}
    for k, *vals in zip(keys, *cols):
        out.setdefault(k, []).append(tuple(vals))
    return out


class Personal(Policy):
    name = "personal"
    version = "v2"
    refresh_every = 600.0   # en vivo: recalcula las features cada 10 min en segundo plano

    # El índice es de clase: en un A/B varias versiones comparten las mismas features y un solo refresh.
    _shared: dict = {}
    _building = False
    _live_bought: frozenset = frozenset()   # {(usuario, producto)} vendidos por nuestro carrusel (v5)

    def __init__(self, until: str | None = None, weights: dict | None = None) -> None:
        super().__init__(until)
        self.w = {**VERSIONS[self.version], **(weights or {})}

    @property
    def ix(self) -> SimpleNamespace:
        return Personal._shared[self.until]

    def fit(self, state):
        """Carga las features del disco (si no hay, las construye). En producción, el primer refresh()
        del bucle las recalcula en segundo plano, así que arrancar con unas antiguas es seguro y rápido."""
        if self.until not in Personal._shared:
            Personal._shared[self.until] = self._index(self.load_features())

    def refresh(self, state):
        """En vivo: reconstruye las features en un proceso aparte y cambia el índice de golpe al terminar."""
        if self.until is not None or Personal._building:
            return
        Personal._building = True

        def work():
            # La descarga y el JSON (600k filas) bloquean el GIL varios segundos: en un proceso aparte
            # no frenan el bucle de respuestas. Aquí solo leemos el Parquet y montamos el índice (~1,5 s).
            from .. import features
            from ..config import ROOT
            try:
                subprocess.run([sys.executable, str(ROOT / "scripts" / "build_features.py")], cwd=ROOT,
                               check=True, capture_output=True, timeout=600)
                Personal._shared[None] = self._index(features.load(until=None))
                print(f"[personal] features actualizadas: {len(self.ix.user_bought)} usuarios con compras")
            except Exception as e:
                print(f"[personal] refresh falló, sigo con las anteriores: {e}")
            finally:
                Personal._building = False

        threading.Thread(target=work, daemon=True).start()

    @staticmethod
    def _index(f) -> SimpleNamespace:
        """De las tablas de features a diccionarios rápidos de consultar."""
        ix = SimpleNamespace()
        items = f["item_features"].fillna(0)
        ix.brand_of = dict(zip(items.product_id, items.brand))
        pop = np.log1p(items.purchases_7d.to_numpy(dtype=float))
        items = items.assign(pop=pop / max(pop.max(initial=0), 1e-9))
        ix.top_pop = list(items.nlargest(50, "pop")[["product_id", "pop"]].itertuples(index=False, name=None))
        ix.subcat_top = _norm_lists(items, "subcategory", "product_id", "pop", 10)
        ix.cat_of = dict(zip(items.product_id, items.category))
        ix.cat_top = _norm_lists(items, "category", "product_id", "pop", 40)
        ix.subcat_of = dict(zip(items.product_id, items.subcategory))
        from ..features import feedback_lift
        ix.fb_lift = feedback_lift(f.get("item_feedback"))

        nb = f["item_neighbors"]
        ix.nb_buy = _norm_lists(nb[nb.kind == "buy"], "p", "q", "n", 20)
        ix.nb_view = _norm_lists(nb[nb.kind == "view"], "p", "q", "n", 20)

        us = f["user_subcat"].sort_values(["user_id", "w"], ascending=[True, False])
        us = us.assign(a=us.w / us.groupby("user_id").w.transform("sum"))
        ix.user_subcat = {u: v[:5] for u, v in _group(us.user_id.tolist(), us.subcategory.tolist(), us.a.tolist()).items()}
        ub = f["user_brand"]
        ub = ub.assign(a=ub.w / ub.groupby("user_id").w.transform("max"))
        ix.user_brand = {u: dict(v) for u, v in _group(ub.user_id.tolist(), ub.brand.tolist(), ub.a.tolist()).items()}
        up = f["user_purchases"]
        t = (pd.to_datetime(up.last_ts, utc=True, errors="coerce") - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
        ix.user_bought = _group(up.user_id.tolist(), up.product_id.tolist(), up.n.clip(upper=4).tolist(), t.tolist())

        # Comprado solo de forma orgánica (nunca en carrusel): en vivo convierte ~4× menos de lo esperado
        nc = up.n_carousel if "n_carousel" in up else 0
        org = up[(up.n - nc > 0) & (nc == 0)]
        ix.user_organic = {u: {q for (q,) in v} for u, v in _group(org.user_id.tolist(), org.product_id.tolist()).items()}
        # Historial útil por (usuario, producto): comprado en carrusel, carrito abandonado, clic antiguo
        ix.user_items = {}
        ui = f.get("user_items")
        if ui is not None and len(ui):
            ix.user_items = _group(ui.user_id.tolist(), ui.product_id.tolist(), ui.carritos.tolist(),
                                   ui.compras_organicas.tolist(), ui.compras_carrusel.tolist(),
                                   ui.clics_carrusel.tolist())

        cp = f["context_popular"].assign(ctx=lambda d: list(zip(d.device, d.region)))
        ix.ctx_pop = _norm_lists(cp, "ctx", "product_id", "n", 30)
        return ix

    def recommend(self, req, state):
        w, ix = self.w, self.ix   # una sola lectura: si el refresh cambia el índice, no nos afecta
        s: dict[str, float] = defaultdict(float)

        views = _views(req)
        for k, v in enumerate(reversed(views)):
            d = w["decay"] ** k
            p = v
            s[p] += w["self"] * d
            for q, n in ix.nb_buy.get(p, ()):
                s[q] += w["buy"] * d * n
            for q, n in ix.nb_view.get(p, ()):
                s[q] += w["view"] * d * n

        user = None if req.get("is_new_user") else req.get("user_id")
        for sub, a in ix.user_subcat.get(user, ()):
            for q, n in ix.subcat_top.get(sub, ()):
                s[q] += w["user"] * a * n

        ctx = req.get("context") or {}
        for q, n in ix.ctx_pop.get((ctx.get("device"), ctx.get("region")), ()):
            s[q] += w["ctx"] * n
        for q, n in ix.top_pop:
            s[q] += w["pop"] * n

        bought = ix.user_bought.get(user, ())
        if bought:
            now = _epoch(req.get("ts"))
            for q, n, t in bought:
                if not all(math.isfinite(float(x)) for x in (n, t)):
                    continue
                s[q] += w["rebuy"] * (0.5 + 0.125 * n) * math.exp(-max(now - t, 0) / 86400 / w["rebuy_days"])

        # Historial del usuario con ese producto (señales medidas en vivo, 141.440 impresiones a conocidos):
        # comprado antes en carrusel 2,15 % por impresión (~30× lo esperado), carrito abandonado 0,30 % (~5×),
        # clic antiguo 0,25 % (~4×). Lo comprado orgánicamente no cuenta aquí (convierte 4× menos).
        hist: set[str] = set()   # productos que han recibido bonus de historial (v5: exentos del filtro)
        if w.get("hist_carousel") or w.get("hist_cart") or w.get("hist_click"):
            for q, carts, org, car, clicks in ix.user_items.get(user, ()):
                if car > 0 or (org <= 0 and (carts > 0 or clicks > 0)):
                    hist.add(q)
                if car > 0:
                    s[q] += w.get("hist_carousel", 0.0)
                elif org > 0:
                    continue
                elif carts > 0:
                    s[q] += w.get("hist_cart", 0.0)
                elif clicks > 0:
                    s[q] += w.get("hist_click", 0.0)
        if w.get("organic", 1.0) != 1.0:
            for q in ix.user_organic.get(user, ()):
                if q in s:
                    s[q] *= w["organic"]

        # Categorías de la sesión: relleno dentro de categoría y penalización (o descarte) de lo de fuera
        cats: dict[str, float] = {}
        rec = w.get("cat_recency", 0.0)   # v4: peso de la vista k-ésima más antigua = rec^k (normalizado)
        rec_total = sum(rec ** k for k in range(len(views))) if rec and views else 0.0
        if views and (w["cat_fill"] or w["off"] != 1):
            for k, v in enumerate(reversed(views)):
                c = ix.cat_of.get(v)
                if c is not None:
                    if rec:
                        cats[c] = cats.get(c, 0.0) + rec ** k / rec_total
                    elif w.get("cat_share"):   # peso = fracción de las vistas que son de esa categoría
                        cats[c] = cats.get(c, 0.0) + 1 / len(views)
                    else:
                        cats[c] = max(cats.get(c, 0.0), w["cat_decay"] ** k)
            for c, a in cats.items():
                for q, n in ix.cat_top.get(c, ()):
                    s[q] += w["cat_fill"] * a * n
        if views and w.get("sub_fill"):   # v4: más vendidos de las subcategorías vistas
            for k, v in enumerate(reversed(views)):
                a = rec ** k / rec_total if rec else 1 / len(views)
                for q, n in ix.subcat_top.get(ix.subcat_of.get(v), ()):
                    s[q] += w["sub_fill"] * a * n

        brands = ix.user_brand.get(user)
        if brands:
            for q in s:
                s[q] *= 1 + w["brand"] * brands.get(ix.brand_of.get(q), 0)

        if w.get("fb") and ix.fb_lift:   # v4: lo que el producto ha demostrado en nuestro carrusel
            fb, lift = w["fb"], ix.fb_lift
            for q in s:
                x = lift.get(q)
                if x is not None:
                    s[q] *= x ** fb

        if w.get("organic", 1.0) == 0.0:   # v4: fuera lo comprado solo de forma orgánica
            for q in ix.user_organic.get(user, ()):
                s.pop(q, None)

        if cats and w["off"] != 1:
            off, hist_off = w["off"], w.get("hist_off", 0.0)
            for q in list(s):
                if ix.cat_of.get(q) not in cats:
                    f = hist_off if (hist_off and q in hist) else off   # v5: el historial no se borra
                    if f:
                        s[q] *= f
                    else:
                        del s[q]

        # v5: ya se lo vendimos en vivo. Con el user_id tal cual: un is_new_user también puede habernos comprado.
        uid = req.get("user_id")
        if w.get("live_rebuy", 1.0) == 0.0 and uid:
            sold = Personal._live_bought
            for q in [q for q in s if (uid, q) in sold]:
                del s[q]
        # top > 10 (v5): finalize() se queda con los 10 primeros válidos; con margen, un agotado lo sustituye
        # nuestro siguiente candidato y no un best seller del relleno (que puede ser algo ya vendido).
        return sorted((p for p in s if math.isfinite(s[p])), key=s.get, reverse=True)[:int(w.get("top", 10))]


class PersonalV3(Personal):
    """Categoría + calidad del producto + recién visto + co-compra, sin perfil de usuario (ver V3)."""
    name = "personal_v3"
    version = "v3"


class PersonalV4(Personal):
    """v3 + recencia de categoría, subcategoría, descarte de lo orgánico y lift del feedback (ver V4)."""
    name = "personal_v4"
    version = "v4"


class PersonalV5(Personal):
    """v4 + el historial del usuario sobrevive al filtro de categoría + no recompra lo vendido en vivo (ver V5).

    refresh() corre cada 2 min: actualiza las ventas en vivo (consulta pequeña, en un hilo) y delega el
    rebuild de features de Personal cada 10 min, como en v3/v4."""
    name = "personal_v5"
    version = "v5"
    refresh_every = 120.0
    _features_at = float("-inf")
    _live_thread: threading.Thread | None = None

    def refresh(self, state):
        if self.until is not None:
            return
        if time.monotonic() - PersonalV5._features_at >= Personal.refresh_every:
            PersonalV5._features_at = time.monotonic()
            super().refresh(state)
        t = PersonalV5._live_thread
        if t is None or not t.is_alive():
            PersonalV5._live_thread = threading.Thread(target=self._update_live_bought, daemon=True)
            PersonalV5._live_thread.start()

    @staticmethod
    def _update_live_bought() -> None:
        from ..client import query
        from ..config import TEAM
        try:
            rows = query(f"""SELECT DISTINCT r.user_id::String AS u, f.purchased::String AS p
                             FROM {TEAM}.feedback AS f INNER JOIN shop.requests AS r
                               ON r.request_id::String = f.request_id::String
                             WHERE f.purchased::String != '' AND f.request_id::String != '__init__'""",
                         timeout=30)
            # Publicación inmutable: recommend() nunca ve el conjunto a medio construir.
            Personal._live_bought = frozenset((r["u"], r["p"]) for r in rows)
        except Exception as e:
            print(f"[personal_v5] ventas en vivo no actualizadas, sigo con las anteriores: {e}")


class PersonalV1(Personal):
    """Leattner.v1 congelada (sin lógica de categoría), para comparar en A/B."""
    name = "personal_v1"
    version = "v1"


def _epoch(ts) -> float:
    """ts de la petición ('YYYY-MM-DD HH:MM:SS.fff…', UTC) en segundos; ahora si falta."""
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()
    except (ValueError, TypeError):
        return datetime.now(timezone.utc).timestamp()


class PersonalNoRebuy(Personal):
    """La política personal de Joan sin el bonus de recompra (para compararla sin esa señal)."""
    name = "personal_norebuy"

    def __init__(self, until: str | None = None, weights: dict | None = None) -> None:
        super().__init__(until, {"rebuy": 0.0, **(weights or {})})
