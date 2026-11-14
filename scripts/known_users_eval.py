"""Señales históricas frente a resultados reales; solo SELECT y cachés locales.

Ejecutar con el Python del reto. Las tasas orgánicas son por par con vista en
vivo (no por impresión de carrusel); los intervalos son Wilson al 95 %.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reco.client import query_df
from reco.config import TEAM

ROOT = Path(__file__).resolve().parents[1] / 'data' / 'known_users_eval'


def leer(nombre, sql):
    p = ROOT / f'{nombre}.parquet'
    if p.exists():
        return pd.read_parquet(p)
    t = time.perf_counter()
    d = query_df(sql+'\nLIMIT 500000', timeout=600)
    if len(d) >= 500000:
        raise RuntimeError(f'{nombre}: dividir consulta, {len(d)} filas')
    d.to_parquet(p, index=False)
    print(f'{nombre}: {len(d):,} filas, {time.perf_counter()-t:.2f}s', flush=True)
    return d


def intervalo(k, n):
    if not n:
        return (0., 0.)
    z = 1.96
    p = k/n
    c = (p+z*z/(2*n))/(1+z*z/n)
    h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return max(0., c-h), min(1., c+h)


def tabla(d, grupo, nombre):
    r = d.groupby(grupo, dropna=False, observed=True).agg(n=('y', 'size'), compras=('y', 'sum')).reset_index()
    r['tasa'] = r.compras/r.n
    base = d.y.mean()
    r['lift_global'] = r.tasa/base if base else np.nan
    ci = [intervalo(k, n) for k, n in zip(r.compras, r.n)]
    r['ic95_inf'], r['ic95_sup'] = zip(*ci)
    r.to_csv(ROOT / f'{nombre}.csv', index=False)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--as-of', default='2026-11-14 13:20:00', help='Corte UTC fijo, sin fuga de futuro')
    ap.add_argument('--benchmark-user-items', action='store_true', help='Medir SQL + Parquet y cobertura histórica del top-3')
    args = ap.parse_args()
    # Validación antes de interpolar el corte en SQL.
    corte = str(pd.Timestamp(args.as_of))
    global ROOT
    ROOT = ROOT / corte.replace(':', '-').replace(' ', '_')
    ROOT.mkdir(parents=True, exist_ok=True)
    cat = leer('catalogo', 'SELECT product_id::String AS product_id, category::String AS category, subcategory::String AS subcategory, brand::String AS brand, price_eur::Float64 AS price FROM shop.catalog UNION ALL SELECT product_id::String, category::String, subcategory::String, brand::String, price_eur::Float64 FROM shop.catalog_updates WHERE change::String = \'new_product\'')
    cat = cat.drop_duplicates('product_id', keep='last')
    precios = leer('precios_corte', f"""SELECT product_id, argMax(precio, ts) AS precio FROM (
        SELECT product_id::String AS product_id, price_eur::Float64 AS precio, toDateTime64('1970-01-01',3) AS ts FROM shop.catalog
        UNION ALL SELECT product_id::String, price_eur::Float64, ts::DateTime64(3) FROM shop.catalog_updates
        WHERE ts::DateTime64(3) < '{corte}') GROUP BY product_id""")
    cat['price'] = cat.product_id.map(precios.set_index('product_id').precio)
    req = leer('peticiones', f"""SELECT request_id::String AS request_id, user_id::String AS user_id,
        is_new_user::Bool AS nuevo, context.page::String AS page,
        recent_views::Array(JSON) AS recent_views, ts::DateTime64(3) AS ts
        FROM shop.requests WHERE ts::DateTime64(3) < '{corte}'""")
    fb = leer('feedback', f"""SELECT request_id::String AS request_id, shown::Array(String) AS shown,
        purchased::String AS purchased, served_by::String AS served_by
        FROM {TEAM}.feedback WHERE request_id::String != '__init__' AND ts::DateTime64(3) < '{corte}'""")
    # Una fila por par orgánico: evita contar la misma compra contra varias vistas.
    ev = []
    for i in range(8):
        ev.append(leer(f'eventos_v2_{i}', f"""SELECT user_id::String AS user_id, product_id::String AS product_id,
            countIf(event_type::String='view') AS lv, countIf(event_type::String='purchase') AS lb,
            avgIf(price_eur::Float64,event_type::String='purchase') AS buy_price,
            avgIf(price_eur::Float64,event_type::String='view') AS view_price,
            min(ts::DateTime64(3)) AS first_ts
            FROM shop.events WHERE ts::DateTime64(3) < '{corte}' AND cityHash64(user_id::String)%8={i}
            GROUP BY user_id, product_id"""))
    ev = pd.concat(ev, ignore_index=True)
    # Subconsulta de pares relevantes; no descarga los 8,8 M pares históricos.
    relevant = f"""SELECT user_id::String AS u, product_id::String AS p FROM shop.events WHERE ts::DateTime64(3) < '{corte}'
        UNION DISTINCT SELECT r.user_id::String, arrayJoin(f.shown::Array(String)) FROM {TEAM}.feedback f
        INNER JOIN shop.requests r ON r.request_id::String=f.request_id::String
        WHERE f.request_id::String!='__init__' AND f.ts::DateTime64(3) < '{corte}'"""
    hist = []
    for i in range(8):
        hist.append(leer(f'historial_{i}', f"""SELECT user_id, product_id, sum(vistas) AS vistas,
            sum(carritos) AS carritos, sum(organicas) AS organicas, sum(clics) AS clics,
            sum(carrusel) AS carrusel, max(ultima_compra) AS ultima_compra FROM (
            SELECT user_id::String AS user_id, product_id::String AS product_id,
            countIf(event_type::String='view') AS vistas, countIf(event_type::String='add_to_cart') AS carritos,
            countIf(event_type::String='purchase') AS organicas, 0 AS clics, 0 AS carrusel,
            maxIf(ts::DateTime64(3),event_type::String='purchase') AS ultima_compra
            FROM shop.history_events WHERE cityHash64(user_id::String)%8={i}
            AND (user_id::String,product_id::String) IN ({relevant}) GROUP BY user_id,product_id
            UNION ALL SELECT user_id::String, p, 0, 0, 0, countIf(clicked::String=p), countIf(purchased::String=p),
            toDateTime64('1970-01-01',3) FROM shop.history_sessions
            ARRAY JOIN arrayDistinct([clicked::String,purchased::String]) AS p
            WHERE p!='' AND cityHash64(user_id::String)%8={i}
            AND (user_id::String,p) IN ({relevant}) GROUP BY user_id,p
            ) GROUP BY user_id,product_id"""))
    hist = pd.concat(hist, ignore_index=True)
    # Perfil histórico completo, independiente del producto comprado hoy.
    perfiles = {}
    for dim in ['category', 'subcategory', 'brand']:
        perfiles[dim] = leer(f'perfil_{dim}', f"""SELECT e.user_id::String AS user_id, c.{dim}::String AS {dim},
            countIf(e.event_type::String='view') AS hv, countIf(e.event_type::String='purchase') AS hb
            FROM shop.history_events e INNER JOIN shop.catalog c ON e.product_id::String=c.product_id::String
            GROUP BY user_id, {dim} ORDER BY hv DESC, {dim} ASC LIMIT 1 BY user_id""")
    precio = leer('precio_usuario', """SELECT user_id::String AS user_id,
        avgIf(price_eur::Float64,event_type::String='purchase') AS precio_hist,
        countIf(event_type::String='purchase') AS n_hist FROM shop.history_events GROUP BY user_id""")
    conocidos = set(precio.loc[precio.n_hist >= 0, 'user_id'])
    cm = cat.set_index('product_id')
    def views(v):
        if isinstance(v, str):
            v = json.loads(v)
        return [json.loads(x) if isinstance(x,str) else x for x in v]
    req['vids'] = req.recent_views.map(lambda v: [x['product_id'] for x in views(v)])
    req['nv'] = req.vids.map(len)
    req['last'] = req.vids.map(lambda v: v[-1] if v else '')
    req['last_cat'] = req['last'].map(cm.category)
    req['view_price'] = req['last'].map(cm.price)
    req['nv_bin'] = pd.cut(req.nv, [-1,0,1,3,100], labels=['0','1','2-3','4+'])
    joined = req.merge(fb, on='request_id', validate='one_to_one')
    joined['y'] = joined.purchased.ne('').astype(int)
    for groups, name in [(['nuevo'], 'conversion'), (['nuevo','page'], 'pagina'), (['nuevo','nv_bin'], 'numero_vistas'), (['nuevo','last_cat'], 'categoria_vista')]:
        tabla(joined, groups, name)
        tabla(joined[joined.served_by.eq('team')], groups, name+'_team')
    joined['precio_bin'] = pd.cut(joined.view_price,[0,20,50,100,200,100000])
    tabla(joined,['nuevo','precio_bin'],'precio_visto')
    joined.groupby('nuevo').agg(n=('request_id','size'), vistas=('nv','mean'), precio_visto=('view_price','mean')).to_csv(ROOT/'composicion.csv')
    # Estandarizar las tasas de nuevos a la composición de conocidos.
    ajustes = []
    for grupos in [['page'],['nv_bin'],['last_cat'],['precio_bin'],['page','nv_bin','last_cat']]:
        tasas = joined[joined.nuevo].groupby(grupos,dropna=False,observed=True).y.mean().rename('tasa_nuevos').reset_index()
        k = joined[~joined.nuevo].merge(tasas,on=grupos,how='left')
        ajustes.append({'estratos':'+'.join(grupos),'tasa_conocidos':k.y.mean(),
            'tasa_nuevos_estandarizada':k.tasa_nuevos.mean(),'cobertura':k.tasa_nuevos.notna().mean()})
    pd.DataFrame(ajustes).to_csv(ROOT/'brecha_ajustada.csv',index=False)
    imp = joined[joined.served_by.eq('team')].copy()
    imp['slot'] = imp.shown.map(lambda x: list(range(1,len(x)+1)))
    imp = imp.explode(['shown','slot']).rename(columns={'shown':'product_id'})
    imp['y'] = imp.product_id.eq(imp.purchased).astype(int)
    imp = imp.merge(cat,on='product_id',how='left')
    imp['misma_cat'] = imp.category.eq(imp.last_cat)
    imp['slot_bin'] = pd.cut(imp.slot.astype(int),[0,1,3,7,10],labels=['1','2-3','4-7','8-10'])
    tabla(imp,['nuevo','misma_cat'],'exposicion_categoria')
    imp.groupby('nuevo').agg(n=('y','size'), precio_mostrado=('price','mean'), misma_cat=('misma_cat','mean')).to_csv(ROOT/'mostrado.csv')
    for dim, perfil in perfiles.items():
        favorito = perfil.set_index('user_id')[dim]
        r = req[~req.nuevo & req.nv.gt(0)].copy()
        r['y'] = r['last'].map(cm[dim]).eq(r.user_id.map(favorito)).astype(int)
        tabla(r,['page'],f'coincidencia_vistas_{dim}')
        todas = req[~req.nuevo & req.nv.gt(0)].explode('vids').copy()
        todas['y'] = todas.vids.map(cm[dim]).eq(todas.user_id.map(favorito)).astype(int)
        tabla(todas,['page'],f'coincidencia_todas_vistas_{dim}')
    organic = ev[ev.user_id.isin(conocidos)].merge(cat,on='product_id',how='left')
    organic['y'] = organic.lb.gt(0).astype(int)
    # En orgánico el precio observado en las vistas evita usar el catálogo inicial.
    if 'view_price' in organic:
        organic['price'] = organic.view_price
    # La tasa usa el precio de las vistas para ambos resultados (sin condicionar a la compra).
    organic['sin_vista'] = organic.lv.eq(0)
    tabla(organic,['sin_vista'],'organico_cobertura')
    for nombre, d in [('organico',organic[organic.lv.gt(0)].copy()),('carrusel',imp[~imp.nuevo].copy())]:
        d = d.merge(hist,on=['user_id','product_id'],how='left')
        for col in ['vistas','carritos','organicas','clics','carrusel']:
            d[col] = d[col].fillna(0)
        d['vistas_sin_compra'] = np.where(d.organicas.add(d.carrusel).gt(0),'comprado',pd.cut(d.vistas,[-1,0,1,2,4,100000],labels=['0','1','2','3-4','5+']).astype(str))
        d['carrito_abandonado'] = d.carritos.gt(0) & d.organicas.add(d.carrusel).eq(0)
        d['compras_organicas'] = pd.cut(d.organicas,[-1,0,1,100000],labels=['0','1','2+'])
        d['clic_carrusel'] = d.clics.gt(0)
        d['compra_carrusel'] = d.carrusel.gt(0)
        dias = (pd.Timestamp(corte,tz='UTC')-pd.to_datetime(d.ultima_compra,utc=True,errors='coerce')).dt.total_seconds()/86400
        d['antiguedad_compra'] = pd.cut(dias,[-1,7,30,100],labels=['0-7d','7-30d','30-60d']).astype(object)
        d.loc[d.organicas.eq(0),'antiguedad_compra'] = 'nunca'
        for dim, perfil in perfiles.items():
            d[f'favorito_{dim}'] = d[dim].eq(d.user_id.map(perfil.set_index('user_id')[dim]))
        d = d.merge(precio,on='user_id',how='left')
        d['ratio_precio'] = pd.cut(d.price/d.precio_hist,[0,.5,.8,1.25,2,100000],labels=['<0.5','0.5-0.8','0.8-1.25','1.25-2','2+'])
        d['senal_prioritaria'] = 'resto'
        d.loc[d.vistas.gt(0),'senal_prioritaria'] = 'visto'
        d.loc[d.clics.gt(0),'senal_prioritaria'] = 'clic antiguo'
        d.loc[d.carrito_abandonado,'senal_prioritaria'] = 'carrito abandonado'
        d.loc[d.carrusel.gt(0),'senal_prioritaria'] = 'compra carrusel'
        d.loc[d.organicas.gt(0),'senal_prioritaria'] = 'compra organica'
        for signal in ['senal_prioritaria','vistas_sin_compra','carrito_abandonado','compras_organicas','clic_carrusel','compra_carrusel','antiguedad_compra','favorito_category','favorito_subcategory','favorito_brand','ratio_precio']:
            tabla(d,[signal],f'{nombre}_{signal}')
            if nombre=='carrusel':
                tabla(d,[signal,'slot','misma_cat'],f'{nombre}_{signal}_control')
                # Estandarización indirecta por slot/categoría: compras observadas / esperadas.
                tasa = d.groupby(['slot','misma_cat'],observed=True).y.transform('mean')
                d['esperadas'] = tasa
                a = d.groupby(signal,observed=True).agg(n=('y','size'),compras=('y','sum'),esperadas=('esperadas','sum'))
                a['oe_slot_categoria'] = a.compras/a.esperadas
                a.to_csv(ROOT/f'{nombre}_{signal}_ajustado.csv')
                d['en_vistas'] = [p in v for p,v in zip(d.product_id,d.vids)]
                d['esperadas'] = d.groupby(['slot','misma_cat','page','en_vistas'],observed=True).y.transform('mean')
                a = d.groupby(signal,observed=True).agg(n=('y','size'),compras=('y','sum'),esperadas=('esperadas','sum'))
                a['oe_ampliado'] = a.compras/a.esperadas
                a.to_csv(ROOT/f'{nombre}_{signal}_robusto.csv')
    # Agregados de historial por dimensión, solo para dimensiones observadas en vivo.
    for dim in ['category','subcategory','brand']:
        partes = []
        for i in range(8):
            partes.append(leer(f'dimension_{dim}_{i}', f"""SELECT e.user_id::String AS user_id,
                c.{dim}::String AS {dim}, countIf(e.event_type::String='view') AS vistas,
                countIf(e.event_type::String='add_to_cart') AS carritos,
                countIf(e.event_type::String='purchase') AS compras
                FROM shop.history_events e INNER JOIN shop.catalog c ON e.product_id::String=c.product_id::String
                WHERE cityHash64(e.user_id::String)%8={i} AND (e.user_id::String,c.{dim}::String) IN (
                    SELECT z.u, c.{dim}::String FROM ({relevant}) z
                    INNER JOIN shop.catalog c ON z.p=c.product_id::String)
                GROUP BY user_id,{dim}"""))
        perfil = pd.concat(partes,ignore_index=True)
        for nombre, base in [('organico',organic[organic.lv.gt(0)]),('carrusel',imp[~imp.nuevo])]:
            d = base.merge(perfil,on=['user_id',dim],how='left')
            for col in ['vistas','carritos','compras']:
                d[col+'_hist'] = pd.cut(d[col].fillna(0),[-1,0,1,4,19,10000000],labels=['0','1','2-4','5-19','20+'])
                grupos = [col+'_hist']
                tabla(d,grupos,f'{nombre}_{dim}_{col}')
                if nombre=='carrusel':
                    tabla(d,grupos+['slot','misma_cat'],f'{nombre}_{dim}_{col}_control')
        del partes, perfil
    if args.benchmark_user_items:
        from reco.features import user_items_query
        tiempos = []
        for hasta in ['2026-11-07',str(pd.Timestamp(req.ts.min())),None]:
            t = time.perf_counter()
            ui = query_df(user_items_query(hasta)+' LIMIT 500000',timeout=120)
            assert len(ui) < 500000 and not ui.duplicated(['user_id','product_id']).any()
            assert ui.groupby('user_id').size().max() <= 3
            if hasta:
                assert pd.to_datetime(ui.ultimo_ts).max() < pd.Timestamp(hasta)
            ui.to_parquet(ROOT/f'user_items_{str(hasta).replace(":","-")}.parquet',index=False)
            tiempos.append({'until':hasta,'filas':len(ui),'segundos':time.perf_counter()-t})
            if hasta not in ['2026-11-07',None]:
                expos = imp[~imp.nuevo].merge(hist,on=['user_id','product_id'],how='left')
                for c in ['organicas','carrusel','clics','carritos']:
                    expos[c] = expos[c].fillna(0)
                expos['senal'] = 'resto'
                expos.loc[expos.clics.gt(0),'senal'] = 'clic'
                expos.loc[expos.carritos.gt(0)&(expos.organicas+expos.carrusel).eq(0),'senal'] = 'carrito'
                expos.loc[expos.carrusel.gt(0),'senal'] = 'carrusel'
                expos.loc[expos.organicas.gt(0),'senal'] = 'organico'
                expos['retenido'] = pd.MultiIndex.from_frame(expos[['user_id','product_id']]).isin(
                    pd.MultiIndex.from_frame(ui[['user_id','product_id']]))
                tabla(expos,['senal','retenido'],'cobertura_user_items')
        pd.DataFrame(tiempos).to_csv(ROOT/'benchmark_user_items.csv',index=False)
    escribir_informe(corte)
    print(f'Resultados: {ROOT}',flush=True)
    print(pd.read_csv(ROOT/'conversion.csv').to_string(index=False))


INTEGRACION = r"""
## Parche propuesto para personal.py (no aplicado)

En Personal._index, antes de return ix:
```python
        ix.user_items = {}
        ui = f.get("user_items")
        if ui is not None:
            ix.user_items = _group(
                ui.user_id.tolist(), ui.product_id.tolist(),
                ui.carritos.tolist(), ui.compras_organicas.tolist(),
                ui.compras_carrusel.tolist(), ui.clics_carrusel.tolist())
```
En recommend, justo después de calcular user:
```python
        categorias_vistas = {ix.cat_of[p] for p in views if p in ix.cat_of}
        historicos = []
        for q, car, org, carr, clic in ix.user_items.get(user, ()):
            if org > 0:
                continue
            if categorias_vistas and ix.cat_of.get(q) not in categorias_vistas:
                continue
            peso = 1.0 if carr > 0 else 0.6 if car > 0 else 0.3 if clic > 0 else 0.0
            if peso:
                historicos.append((peso, q))
        for peso, q in sorted(historicos, reverse=True)[:2]:
            s[q] += w.get("history", 0.5) * peso
```
Los pesos son iniciales para experimentar; el lift no determina un peso óptimo.
Para desactivar el bonus antiguo de recompra al probar esta señal:
`Personal(weights={"rebuy": 0.0, "history": 0.5})`. PersonalV3 ya usa rebuy=0.
No eliminar la comprobación final de stock ni la prioridad de categoría.

## Parche propuesto para reco/ranker.py (no aplicado)

Después de definir FEATURES, añadir (al final, conservando los índices existentes):
```python
FEATURES += ["hist_views", "hist_carts", "hist_organic_buys",
             "hist_carousel_buys", "hist_carousel_clicks", "hist_age_days"]
```
Al final de Featurizer.__init__:
```python
        self.user_items = defaultdict(dict)
        ui = feats.get("user_items")
        if ui is not None:
            segundos = pd.to_datetime(ui.ultimo_ts, utc=True).astype("int64") / 1e9
            for u, p, v, car, org, carr, clic, ts in zip(
                    ui.user_id, ui.product_id, ui.vistas, ui.carritos,
                    ui.compras_organicas, ui.compras_carrusel, ui.clics_carrusel, segundos):
                if p in self.idx:
                    self.user_items[u][self.idx[p]] = (v, car, org, carr, clic, ts)
```
En candidates, antes de np.unique y del filtro bought:
```python
        categorias_vistas = {self.cat[v] for v in views}
        for q, (_, car, org, carr, clic, _) in self.user_items.get(user, {}).items():
            if org == 0 and (car or carr or clic):
                if not categorias_vistas or self.cat[q] in categorias_vistas:
                    c.append(q)
```
Sustituir el filtro bought por este bloque para permitir compras antiguas de
carrusel sin compra orgánica. La ausencia en el top-3 mantiene la exclusión anterior:
```python
        bought = self.bought.get(user)
        if bought:
            historial = self.user_items.get(user, {})
            permitir = {q for q, x in historial.items() if x[2] == 0 and x[3] > 0}
            excluir = bought - permitir
            out = out[~np.isin(out, list(excluir))]
```
En matrix, antes del return np.nan_to_num:
```python
        historial = self.user_items.get(user, {})
        filas = np.array([historial.get(int(q), (0, 0, 0, 0, 0, 0))
                          for q in cand], dtype=float).reshape(-1, 6)
        inicio = FEATURES.index("hist_views")
        X[:, inicio:inicio+5] = np.log1p(filas[:, :5])
        instante = pd.Timestamp(req["ts"])
        instante = instante.tz_localize("UTC") if instante.tzinfo is None else instante
        X[:, inicio+5] = np.where(filas[:, 5] > 0,
            np.clip((instante.timestamp()-filas[:, 5])/86400, 0, 365), 365)
```
Obligatorio reentrenar: el modelo actual espera 31 columnas. El conjunto para
entrenar con feedback debe construirse con features anteriores a cada bloque de
validación. El entrenamiento antiguo descarta compras repetidas; para aprender
la excepción de carrusel hay que ajustar ese filtro en el entrenador (parche
pendiente fuera de los ficheros asignados). Los parches no se han integrado ni
validado mediante un entrenamiento; solo user_items y el análisis están ejecutados.
"""


def escribir_informe(corte):
    """Informe portable junto a los CSV; no modifica documentación compartida."""
    partes = [f"# Historial de usuarios conocidos · corte UTC {corte}",
        "Solo SELECT. Historial fijo de 60 días, anterior al tráfico live. "
        "Orgánico: un ensayo por par usuario-producto con >=1 vista hoy; éxito si hubo >=1 compra hoy. "
        "Carrusel: una impresión real de feedback.shown, solo served_by=team y usuario conocido. "
        "Las tablas de conversión por petición incluyen fallback; también se entrega la versión solo team. "
        "Las filas de coincidencia de vistas cuentan coincidencias, no compras.",
        "Los IC95 son Wilson binomiales descriptivos: no corrigen correlación por usuario ni comparaciones múltiples. "
        "lift_global divide por la tasa global del mismo conjunto. OE divide compras observadas por esperadas "
        "según slot exacto y coincidencia con categoría de última vista; robusto añade página y producto recién visto. "
        "No son estimadores causales ni uplift de una política nueva. Precio orgánico: media observada en vistas; "
        "precio del carrusel: instantánea del catálogo al corte, no precio histórico de cada impresión."]
    nombres = ['conversion','conversion_team','pagina','numero_vistas','composicion','mostrado','brecha_ajustada',
               'categoria_vista','precio_visto','exposicion_categoria',
               'coincidencia_vistas_category','coincidencia_todas_vistas_category',
               'coincidencia_vistas_subcategory','coincidencia_vistas_brand']
    senales = ['senal_prioritaria','vistas_sin_compra','carrito_abandonado','compras_organicas',
               'clic_carrusel','compra_carrusel','antiguedad_compra',
               'favorito_category','favorito_subcategory','favorito_brand','ratio_precio']
    for fuente in ['organico','carrusel']:
        nombres += [f'{fuente}_{s}' for s in senales]
        if fuente == 'carrusel':
            nombres += [f'{fuente}_{s}_ajustado' for s in senales]
            nombres += [f'{fuente}_{s}_robusto' for s in senales]
        nombres += [f'{fuente}_{dim}_{s}' for dim in ['category','subcategory','brand'] for s in ['vistas','carritos','compras']]
    for nombre in nombres:
        d = pd.read_csv(ROOT/f'{nombre}.csv')
        partes += [f"## {nombre}", '```text\n'+d.to_string(index=False)+'\n```']
    partes += ["## Límites y propuesta",
        "Priorizar compras antiguas de carrusel SIN compra orgánica, después carritos abandonados y clics "
        "sin compra orgánica. Conservar categoría actual, calidad general y contexto como señales principales. "
        "No añadir repetición orgánica ni un bonus general por muchas vistas históricas; no se observa "
        "preferencia por comprar cerca de la media histórica de precio. Probar como máximo dos candidatos "
        "históricos por petición, primero dentro de la categoría actual. Los pesos son una propuesta, no un resultado A/B.",
        "user_items es un top-3 por usuario: no sirve para excluir todas las compras. Mantener user_purchases "
        "para eso. Los contadores incluyen live cuando until=None; no usar esas features para evaluar tráfico "
        "anterior. La evaluación de este informe utiliza exclusivamente historia antigua como predictor. "
        "No se ha cambiado producción ni entrenado un modelo nuevo."]
    for nombre in ['benchmark_user_items','cobertura_user_items']:
        if (ROOT/f'{nombre}.csv').exists():
            partes += [f'## {nombre}', '```text\n'+pd.read_csv(ROOT/f'{nombre}.csv').to_string(index=False)+'\n```']
    partes.append(INTEGRACION)
    (ROOT/'informe.md').write_text('\n\n'.join(partes)+'\n')


if __name__ == '__main__':
    main()
