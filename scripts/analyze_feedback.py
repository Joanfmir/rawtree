"""¿Qué premia el simulador? Análisis de team_NN.feedback (solo lectura).

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/analyze_feedback.py [min_peticiones_por_versión]

Tablas (todas con nº de muestras e intervalo de confianza al 95 %, Wilson):
  1. Recompra   clic/compra de un producto mostrado según si el usuario ya lo había comprado, por slot
  2. Posición   clic/compra por slot
  3. Contexto   conversión por page y device; compras que vienen de recent_views / misma subcategoría
  4. Frío       conversión con is_new_user y con vistas recientes de productos nuevos (new_product)
  5. Lo anterior (resumido) por model_version, si la versión tiene suficientes peticiones

"Ya comprado" se mide en dos orígenes distintos:
  hist  compra en el histórico de 60 días (shop.history_events purchase + shop.history_sessions purchased)
  vivo  compra durante el evento ANTES de la petición (shop.events purchase + nuestras ventas del carrusel)
El "ratio ajustado" compara las compras observadas con las esperadas si esos productos convirtieran como
los no comprados EN EL MISMO SLOT (quita el sesgo de posición). Ratio > 1: se compran más de lo normal.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reco import config  # noqa: E402
from reco.client import query_df  # noqa: E402

config.check()
T = config.TEAM
MIN_REQ = int(sys.argv[1]) if len(sys.argv) > 1 else 1000   # peticiones mínimas para desglosar una versión
pd.set_option("display.width", 200)

# Usuarios con feedback: para no traer el histórico de compras entero
USERS = f"""SELECT r.user_id::String FROM shop.requests AS r
            WHERE r.request_id::String IN (SELECT request_id::String FROM {T}.feedback)"""


# ───────────────────────────── estadística ─────────────────────────────
def wilson(x, n):
    """Intervalo de Wilson al 95 % para x éxitos de n. Devuelve (lo, hi) en proporción."""
    x, n = np.asarray(x, dtype=float), np.asarray(n, dtype=float)
    nn = np.where(n > 0, n, 1.0)
    p, z = x / nn, 1.96
    den = 1 + z * z / nn
    mid = (p + z * z / (2 * nn)) / den
    half = z * np.sqrt(p * (1 - p) / nn + z * z / (4 * nn * nn)) / den
    return np.where(n > 0, np.clip(mid - half, 0, 1), np.nan), np.where(n > 0, np.clip(mid + half, 0, 1), np.nan)


def rate_cols(x, n):
    """'1.23 % [0.90–1.60]' para cada fila."""
    lo, hi = wilson(x, n)
    x, n = np.asarray(x, dtype=float), np.asarray(n, dtype=float)
    return [f"{100 * a / b:6.2f} % [{100 * l:5.2f}–{100 * h:5.2f}]" if b else "     —"
            for a, b, l, h in zip(x, n, lo, hi)]


def rates(df, by, clicks="click", buys="buy", unit="n"):
    """Tabla con n, clics, compras y sus tasas con IC, agrupando por `by`."""
    g = df.groupby(by, dropna=False).agg(**{unit: (buys, "size"), "clics": (clicks, "sum"), "compras": (buys, "sum")})
    g["ctr"] = rate_cols(g.clics, g[unit])
    g["conv"] = rate_cols(g.compras, g[unit])
    return g.reset_index()


def show(df):
    print(df.to_string(index=False) if len(df) else "  (sin datos)")


def adjusted_ratio(imp, mask, ref, col="buy"):
    """Observado / esperado de `col` en imp[mask], donde lo esperado usa la tasa por slot de imp[ref].
    Devuelve (n, observado, esperado, ratio, lo, hi). El IC es log-normal (Poisson en ambos recuentos)."""
    a, b = imp[mask], imp[ref]
    ref_rate = b.groupby("slot")[col].mean()
    exp = float((a.groupby("slot").size() * ref_rate).sum())
    obs, ref_obs = int(a[col].sum()), int(b[col].sum())
    if not obs or not ref_obs or exp <= 0:
        return len(a), obs, exp, np.nan, np.nan, np.nan
    r, se = obs / exp, (1 / obs + 1 / ref_obs) ** 0.5
    return len(a), obs, exp, r, r * np.exp(-1.96 * se), r * np.exp(1.96 * se)


def verdict(r, lo, hi, obs):
    if np.isnan(r):
        return "sin compras: no se puede estimar"
    sure = "fiable" if obs >= 20 else "provisional: menos de 20 eventos"
    if lo > 1:
        return f"convierten MÁS que los no comprados ({sure})"
    if hi < 1:
        return f"convierten MENOS que los no comprados ({sure})"
    return "no concluyente: el intervalo incluye 1 (puede ser azar)"


def diff_test(x1, n1, x2, n2):
    """z de la diferencia de dos proporciones."""
    if not n1 or not n2:
        return 0.0
    pool = (x1 + x2) / (n1 + n2)
    se = (pool * (1 - pool) * (1 / n1 + 1 / n2)) ** 0.5
    return (x1 / n1 - x2 / n2) / se if se else 0.0


def z_text(z):
    return ("diferencia fiable" if abs(z) >= 2.6 else "probable, faltan datos" if abs(z) >= 2
            else "no fiable: puede ser azar")


# ───────────────────────────── datos ─────────────────────────────
def load():
    req = query_df(f"""
        SELECT f.request_id::String AS request_id, r.ts::String AS ts, r.user_id::String AS user_id,
               r.is_new_user::Bool AS is_new_user, r.`context.page`::String AS page,
               r.`context.device`::String AS device, r.recent_views AS recent_views,
               f.shown AS shown, f.clicked::String AS clicked, f.purchased::String AS purchased,
               f.served_by::String AS served_by, f.fallback_reason::String AS reason
        FROM {T}.feedback AS f INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
        WHERE f.request_id::String != '__init__'""")
    req = req.drop_duplicates("request_id").reset_index(drop=True)
    req["ts"] = pd.to_datetime(req.ts)
    req["views"] = [[v["product_id"] for v in (rv or [])] for rv in req.recent_views]

    sub = query_df(f"""SELECT request_id::String AS request_id, any(model_version::String) AS model
                       FROM {T}.submissions WHERE request_id::String != '__init__' GROUP BY request_id""")
    req = req.merge(sub, on="request_id", how="left")
    # Lo que vio el comprador: nuestra lista (versión X) o el fallback (best sellers)
    req["model"] = np.where(req.served_by == "team", req.model.fillna("?"), "fallback (baseline)")

    cat = query_df("""
        SELECT product_id, argMax(subcategory, t) AS subcategory FROM (
            SELECT product_id::String AS product_id, subcategory::String AS subcategory, toDateTime64('2000-01-01', 3) AS t FROM shop.catalog
            UNION ALL
            SELECT product_id::String, subcategory::String, ts::DateTime64(3) FROM shop.catalog_updates)
        GROUP BY product_id""")
    subcat = dict(zip(cat.product_id, cat.subcategory))
    new_products = set(query_df(
        "SELECT DISTINCT product_id::String AS p FROM shop.catalog_updates WHERE change::String = 'new_product'").p)

    hist = query_df(f"""
        SELECT u, p, sum(n) AS n, max(last) AS last FROM (
            SELECT user_id::String AS u, product_id::String AS p, count() AS n, max(ts::DateTime64(3)) AS last
            FROM shop.history_events WHERE event_type::String = 'purchase' AND user_id::String IN ({USERS}) GROUP BY u, p
            UNION ALL
            SELECT user_id::String AS u, purchased::String AS p, count() AS n, max(ts::DateTime64(3)) AS last
            FROM shop.history_sessions WHERE purchased::String != '' AND user_id::String IN ({USERS}) GROUP BY u, p)
        GROUP BY u, p""")
    live = query_df(f"""
        SELECT user_id::String AS u, product_id::String AS p, min(ts::DateTime64(3))::String AS first
        FROM shop.events WHERE event_type::String = 'purchase' AND user_id::String IN ({USERS}) GROUP BY u, p""")
    return req, subcat, new_products, hist, live


def build(req, subcat, new_products, hist, live):
    """Añade columnas por petición y devuelve también la tabla de impresiones (una fila por producto mostrado)."""
    req["click"] = req.clicked != ""
    req["buy"] = req.purchased != ""
    req["n_views"] = req.views.str.len()
    req["last_sub"] = [subcat.get(v[-1]) if v else None for v in req.views]
    req["view_new"] = [any(p in new_products for p in v) for v in req.views]
    req["cold"] = req.is_new_user | req.view_new
    req["buy_in_views"] = [b and p in v for b, p, v in zip(req.buy, req.purchased, req.views)]
    req["buy_same_sub"] = [b and s is not None and subcat.get(p) == s
                           for b, p, s in zip(req.buy, req.purchased, req.last_sub)]
    req["buy_not_shown"] = [b and p not in s for b, p, s in zip(req.buy, req.purchased, req.shown)]

    imp = req[["request_id", "ts", "user_id", "model", "served_by", "shown", "clicked", "purchased",
               "views", "last_sub", "is_new_user"]].explode("shown").rename(columns={"shown": "product"})
    imp["slot"] = imp.groupby("request_id").cumcount() + 1
    imp["click"] = imp["product"] == imp.clicked
    imp["buy"] = imp["product"] == imp.purchased
    imp["in_views"] = [p in v for p, v in zip(imp["product"], imp.views)]
    imp["same_sub"] = [s is not None and subcat.get(p) == s for p, s in zip(imp["product"], imp.last_sub)]

    # ¿Ya lo había comprado? Histórico (60 días) y en vivo (antes de la petición)
    h = hist.rename(columns={"u": "user_id", "p": "product", "n": "hist_n", "last": "hist_last"})
    imp = imp.merge(h, on=["user_id", "product"], how="left")
    imp["in_hist"] = imp.hist_n.notna()
    imp["hist_days"] = (imp.ts - pd.to_datetime(imp.hist_last)).dt.total_seconds() / 86400

    own = req.loc[req.buy, ["user_id", "purchased", "ts"]].rename(columns={"purchased": "product", "ts": "first"})
    lv = live.rename(columns={"u": "user_id", "p": "product"})
    lv["first"] = pd.to_datetime(lv["first"])
    lv = pd.concat([lv, own]).groupby(["user_id", "product"], as_index=False)["first"].min()
    imp = imp.merge(lv, on=["user_id", "product"], how="left")
    imp["live"] = imp["first"].notna() & (imp["first"] < imp.ts)   # estrictamente antes de esta petición
    imp["bought"] = np.select([imp.live, imp.in_hist], ["vivo (durante el evento)", "hist (60 días)"], "nunca")
    return req, imp


# ───────────────────────────── informes ─────────────────────────────
def rebuy_report(imp, detail):
    never = imp.bought == "nunca"
    print("\nPor impresión (producto mostrado), según si el usuario ya lo había comprado:")
    show(rates(imp, "bought", unit="impresiones"))
    print("\nRatio de compra ajustado por slot (observado / esperado si convirtieran como 'nunca'):")
    rows = []
    groups = [("hist (60 días)", imp.bought == "hist (60 días)"), ("vivo (durante el evento)", imp.live),
              ("hist, 1 compra", (imp.bought == "hist (60 días)") & (imp.hist_n == 1)),
              ("hist, 2 compras", (imp.bought == "hist (60 días)") & (imp.hist_n == 2)),
              ("hist, 3+ compras", (imp.bought == "hist (60 días)") & (imp.hist_n >= 3)),
              ("hist, última ≤ 14 días", (imp.bought == "hist (60 días)") & (imp.hist_days <= 14)),
              ("hist, última 15–30 días", (imp.bought == "hist (60 días)") & imp.hist_days.between(14, 30, inclusive="right")),
              ("hist, última > 30 días", (imp.bought == "hist (60 días)") & (imp.hist_days > 30))]
    for name, m in groups:
        for col, label in (("buy", "compra"), ("click", "clic")):
            n, obs, exp, r, lo, hi = adjusted_ratio(imp, m, never, col)
            rows.append({"grupo": name, "evento": label, "impresiones": n, "observado": obs,
                         "esperado": round(exp, 1), "ratio": "—" if np.isnan(r) else f"{r:.2f} [{lo:.2f}–{hi:.2f}]",
                         "lectura": verdict(r, lo, hi, obs)})
    show(pd.DataFrame(rows))
    if detail:
        print("\nPor slot (hist+vivo juntos = 'ya comprado'):")
        d = imp.assign(estado=np.where(never, "nunca", "ya comprado"))
        t = rates(d, ["slot", "estado"], unit="impresiones")
        show(t)


def context_report(req, imp):
    for col in ("page", "device"):
        print(f"\nConversión por {col}:")
        show(rates(req, col, unit="peticiones"))
    nb = int(req.buy.sum())
    print(f"\n¿De dónde salen las compras? ({nb} compras)")
    if nb:
        b = req[req.buy]
        both = int((b.buy_in_views | b.buy_same_sub).sum())
        t = pd.DataFrame({"origen": ["producto que estaba en recent_views", "misma subcategoría que la última vista",
                                     "cualquiera de las dos", "comprado fuera de la lista mostrada"],
                          "compras": [int(b.buy_in_views.sum()), int(b.buy_same_sub.sum()), both, int(b.buy_not_shown.sum())]})
        t["de_las_compras"] = rate_cols(t.compras, [nb] * len(t))
        show(t)
    print("\nPor impresión: ¿convierte más un producto ya visto / de la subcategoría de la última vista?")
    d = imp.assign(tipo=np.select([imp.in_views, imp.same_sub], ["en recent_views", "misma subcat (no visto)"], "otro"))
    t = rates(d, "tipo", unit="impresiones")
    other = d.tipo == "otro"
    ratio = []
    for name in t.tipo:
        if name == "otro":
            ratio.append("1 (referencia)")
            continue
        _, obs, _, r, lo, hi = adjusted_ratio(d, d.tipo == name, other)
        ratio.append("—" if np.isnan(r) else f"{r:.1f}× [{lo:.1f}–{hi:.1f}]")
    t["ratio_compra_ajustado_slot"] = ratio
    show(t)
    print("\nConversión según nº de vistas recientes:")
    show(rates(req.assign(vistas=req.n_views.clip(upper=5).map(lambda k: "5+" if k == 5 else str(k))), "vistas", unit="peticiones"))


def cold_report(req):
    print("\nConversión con is_new_user:")
    show(rates(req, "is_new_user", unit="peticiones"))
    print("\nConversión con un producto nuevo (new_product) en recent_views:")
    show(rates(req.rename(columns={"view_new": "vista_de_producto_nuevo"}), "vista_de_producto_nuevo", unit="peticiones"))
    print("\nPeticiones frías (lo que puntúa en 'cold start' = usuario nuevo O vista de producto nuevo):")
    show(rates(req.rename(columns={"cold": "fria"}), "fria", unit="peticiones"))
    c = req[req.cold]
    print(f"  frías: {len(c)} de {len(req)} ({100 * len(c) / max(len(req), 1):.1f} %) · "
          f"solo usuario nuevo {int((req.is_new_user & ~req.view_new).sum())} · "
          f"solo producto nuevo {int((~req.is_new_user & req.view_new).sum())} · ambas {int((req.is_new_user & req.view_new).sum())}")


def report(req, imp, title, detail):
    print(f"\n{'═' * 100}\n{title} · {len(req)} peticiones · {int(req.click.sum())} clics · {int(req.buy.sum())} compras "
          f"· conversión {rate_cols([req.buy.sum()], [len(req)])[0].strip()}\n{'═' * 100}")
    print("\n── 1. RECOMPRA ──")
    rebuy_report(imp, detail)
    print("\n── 2. POSICIÓN ──")
    show(rates(imp, "slot", unit="impresiones"))
    if detail:
        print("\n── 3. CONTEXTO ──")
        context_report(req, imp)
    else:
        print("\n── 3. CONTEXTO (resumen) ──")
        show(rates(req, "page", unit="peticiones"))
    print("\n── 4. ARRANQUE EN FRÍO ──")
    cold_report(req)


def main():
    req, imp = build(*load())
    print(f"Feedback de {T}: {len(req)} peticiones entre {req.ts.min():%d/%m %H:%M} y {req.ts.max():%d/%m %H:%M} UTC")
    print("Los intervalos entre corchetes son al 95 %. Si dos intervalos se solapan mucho, la diferencia puede ser azar.")

    report(req, imp, "TODO (nuestras listas + fallback)", detail=True)

    print(f"\n{'═' * 100}\n5. POR VERSIÓN DE MODELO\n{'═' * 100}")
    by = rates(req, "model", unit="peticiones").sort_values("peticiones", ascending=False)
    cold = rates(req[req.cold], "model", unit="frias")[["model", "frias", "conv"]].rename(columns={"conv": "conv_frias"})
    show(by.merge(cold, on="model", how="left"))
    rows = by.to_dict("records")
    if len(rows) >= 2:
        print(f"\nConversión de cada versión frente a '{rows[0]['model']}' (la de más tráfico). "
              "Ojo: no es un A/B limpio si corrieron a horas distintas.")
        for r in rows[1:]:
            z = diff_test(r["compras"], r["peticiones"], rows[0]["compras"], rows[0]["peticiones"])
            print(f"  {r['model']}: z = {z:+.2f} → {z_text(z)}")
    for m, n in zip(by.model, by.peticiones):
        if n < MIN_REQ:
            print(f"\n[{m}] solo {n} peticiones (< {MIN_REQ}): sin desglose, no habría muestras suficientes.")
            continue
        report(req[req.model == m], imp[imp.model == m], f"VERSIÓN {m}", detail=False)

    print("\nNotas para leer la recompra:")
    print(" · En 'fallback (baseline)' la lista es la misma para todos (best sellers): que el usuario hubiera comprado")
    print("   o no el producto no depende de nuestro modelo, así que es la comparación más limpia.")
    print(" · En nuestras versiones, los 'ya comprados' los eligió el modelo por afinidad: su ratio mezcla el efecto")
    print("   de recompra con el de 'producto afín al usuario'.")


if __name__ == "__main__":
    main()
