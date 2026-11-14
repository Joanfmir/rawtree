"""Análisis SELECT de tendencias/precios; agregados y resultados en data/trends_eval.

    python scripts/trends_eval.py          # descarga agregados y evalúa
    python scripts/trends_eval.py --cached # repite sobre la instantánea local
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reco.client import query_df
from reco.config import DATA_DIR

OUT = DATA_DIR / 'trends_eval'


def descargar():
    OUT.mkdir(parents=True, exist_ok=True)
    fuentes = {
        'catalog': "SELECT product_id::String AS product_id, category::String AS category, price_eur::Float64 AS price FROM shop.catalog",
        'updates': "SELECT product_id::String AS product_id, category::String AS category, ts::DateTime64(3) AS ts, change::String AS change, price_eur::Float64 AS price FROM shop.catalog_updates",
        'live': "SELECT product_id::String AS product_id, toStartOfInterval(ts::DateTime64(3), INTERVAL 5 MINUTE) AS ts, countIf(event_type::String='purchase') AS buys, countIf(event_type::String='view') AS views FROM shop.events GROUP BY product_id, ts",
        'clock': "SELECT max(ts::DateTime64(3)) AS ts FROM shop.events",
    }
    reloj = query_df(fuentes.pop('clock'))
    reloj.to_parquet(OUT / 'clock.parquet', index=False)
    fin = pd.Timestamp(reloj.ts.iloc[0]).floor('D')
    for i in range(8):
        a, b = fin - pd.Timedelta(days=i), fin - pd.Timedelta(days=i-1)
        fuentes[f'history_{i}'] = f"SELECT product_id::String AS product_id, toStartOfHour(ts::DateTime64(3)) AS ts, count() AS buys FROM shop.history_events WHERE event_type::String='purchase' AND ts::DateTime64(3)>='{a}' AND ts::DateTime64(3)<'{b}' GROUP BY product_id, ts"
    # Partir el stream evita superar 500k filas incluso al terminar el reto.
    sql_live = fuentes.pop('live')
    partes = []
    for a in pd.date_range(fin, pd.Timestamp(reloj.ts.iloc[0]), freq='3h'):
        b = a + pd.Timedelta(hours=3)
        sql = sql_live.replace(' GROUP BY', f" WHERE ts::DateTime64(3)>='{a}' AND ts::DateTime64(3)<'{b}' GROUP BY")
        partes.append(query_df(sql))
    pd.concat(partes).to_parquet(OUT / 'live.parquet', index=False)
    for nombre, sql in fuentes.items():
        t = time.perf_counter()
        df = query_df(sql)
        if len(df) > 500000:
            raise ValueError('Agregado demasiado grande')
        df.to_parquet(OUT / f'{nombre}.parquet', index=False)
        print(f'{nombre}: {len(df)} filas, {time.perf_counter()-t:.3f}s', flush=True)


def analizar():
    cat = pd.read_parquet(OUT / 'catalog.parquet')
    up = pd.read_parquet(OUT / 'updates.parquet')
    live = pd.read_parquet(OUT / 'live.parquet')
    fin = pd.Timestamp(pd.read_parquet(OUT / 'clock.parquet').ts.iloc[0])
    up.ts, live.ts = pd.to_datetime(up.ts), pd.to_datetime(live.ts)
    up = up[up.ts <= fin]
    live = live[live.ts < fin.floor('5min')]
    nuevos = up[up.change == 'new_product'][['product_id', 'category', 'price']]
    cat = pd.concat([cat, nuevos]).drop_duplicates('product_id').sort_values('product_id').set_index('product_id')
    ids = cat.index
    hist = pd.concat([pd.read_parquet(OUT / f'history_{i}.parquet') for i in range(8)])
    hist.ts = pd.to_datetime(hist.ts)
    compras = pd.concat([hist, live[['product_id', 'ts', 'buys']]])

    def suma(df, a, b, col='buys'):
        return df.loc[(df.ts >= a) & (df.ts < b)].groupby('product_id')[col].sum().reindex(ids, fill_value=0).astype(float)

    filas = []
    inicio = live.ts.min()
    for corte in pd.date_range(inicio.ceil('h') + pd.Timedelta(hours=1), fin.floor('h') - pd.Timedelta(hours=1), freq='h'):
        y = suma(live, corte, corte + pd.Timedelta(hours=1))
        senales = {nombre: suma(compras, corte-pd.Timedelta(hours=h), corte) / h for nombre, h in [('pop_7d',168), ('pop_1d',24), ('buy_1h',1), ('buy_3h',3)]}
        senales['mix_1h_7d'] = .5*senales['buy_1h']+.5*senales['pop_7d']
        senales['mix_3h_7d'] = .5*senales['buy_3h']+.5*senales['pop_7d']
        senales['mix_3h_1d'] = .5*senales['buy_3h']+.5*senales['pop_1d']
        for nombre, s in senales.items():
            for grupo, ix in [('global', ids)] + [(c, g.index) for c,g in cat.groupby('category')]:
                # Desempate determinista por ID; Spearman usa rangos medios e incluye ceros.
                real = y.loc[ix].sort_values(ascending=False, kind='stable').head(50).index
                pred = s.loc[ix].sort_values(ascending=False, kind='stable').head(50).index
                filas.append(dict(corte=str(corte), grupo=grupo, senal=nombre, recall50=len(set(real)&set(pred))/len(real), spearman=s.loc[ix].rank().corr(y.loc[ix].rank()), compras=float(y.loc[ix].sum())))
    metricas = pd.DataFrame(filas)
    metricas.to_csv(OUT / 'backtest.csv', index=False)
    resumen = metricas.assign(ambito=lambda d: np.where(d.grupo=='global','global','categoria')).groupby(['ambito','senal'])[['recall50','spearman']].mean()

    # Cada cambio se compara con productos de igual categoría y quintil de demanda previa,
    # sin otro cambio en la ventana; controles contemporáneos absorben el efecto de hora.
    efectos = []
    for tipo in ['price_change','back_in_stock']:
        for horizonte in ([1, 2] if tipo == 'price_change' else [1]):
            for ts, lote in up[up.change==tipo].groupby('ts'):
                if ts-pd.Timedelta(hours=1)<inicio or ts+pd.Timedelta(hours=horizonte)>fin:
                    continue
                previo = suma(live,ts-pd.Timedelta(hours=2 if tipo=='back_in_stock' else 1),ts-pd.Timedelta(hours=1 if tipo=='back_in_stock' else 0),'views')
                estrato = cat.category.astype(str)+'_'+pd.qcut(previo.rank(method='first'),5,labels=False).astype(str)
                alterados = set(up.loc[(up.ts>=ts-pd.Timedelta(hours=1))&(up.ts<ts+pd.Timedelta(hours=horizonte)), 'product_id'])
                control = ~ids.isin(alterados)
                for col in ['buys','views']:
                    # Para retorno, comparar con la hora anterior al agotamiento (dura 60 min).
                    if tipo == 'back_in_stock':
                        if ts-pd.Timedelta(hours=2)<inicio:
                            continue
                        antes = suma(live,ts-pd.Timedelta(hours=2),ts-pd.Timedelta(hours=1),col)
                    else:
                        antes = suma(live,ts-pd.Timedelta(hours=1),ts,col)
                    despues = suma(live,ts,ts+pd.Timedelta(hours=horizonte),col)
                    control_pre = antes[control].groupby(estrato[control]).sum()
                    control_post = despues[control].groupby(estrato[control]).sum()
                    factor = estrato.map((control_post+1)/(control_pre+1))
                    # Si toda la categoría cambia (Pets), no existe control intraclase.
                    quintil = pd.qcut(previo.rank(method='first'),5,labels=False)
                    externo = (despues[control].groupby(quintil[control]).sum()+1)/(antes[control].groupby(quintil[control]).sum()+1)
                    factor = factor.fillna(quintil.map(externo))
                    for p in lote.product_id:
                        anterior = up[(up.product_id==p)&(up.ts<ts)].sort_values('ts')
                        precio = anterior.price.iloc[-1] if len(anterior) else cat.loc[p,'price']
                        nuevo = lote.set_index('product_id').loc[p,'price']
                        efectos.append(dict(tipo=tipo,horizonte_h=horizonte,ts=str(ts),product_id=p,metrica=col,antes=antes[p],despues=despues[p],esperado=antes[p]*factor[p],bajada=1-nuevo/precio))
    ef = pd.DataFrame(efectos)
    ef.to_csv(OUT / 'efectos.csv',index=False)
    efecto_res = []
    if len(ef):
        ef['grupo_bajada'] = pd.cut(ef.bajada,[-np.inf,.15,.25,np.inf],labels=['<15%','15-25%','>25%']).astype(str)
        for keys,g in ef.groupby(['tipo','metrica','grupo_bajada','horizonte_h']):
            efecto_res.append(dict(tipo=keys[0],metrica=keys[1],bajada=keys[2],horizonte_h=int(keys[3]),n=len(g),antes=g.antes.sum(),despues=g.despues.sum(),esperado=g.esperado.sum(),ratio_controlado=g.despues.sum()/g.esperado.sum() if g.esperado.sum()>0 else None))
    precios, duraciones = [], []
    for p,g in up.sort_values('ts').groupby('product_id'):
        precio = cat.loc[p,'price']
        fuera = None
        for r in g.itertuples():
            if r.change=='price_change':
                precios.append(dict(product_id=p,ratio=r.price/precio,ratio_inicial=r.price/cat.loc[p,'price']))
            precio=r.price
            if r.change=='out_of_stock': fuera=r.ts
            elif r.change=='back_in_stock' and fuera is not None:
                duraciones.append((r.ts-fuera).total_seconds()/60)
                fuera=None
    pr = pd.DataFrame(precios)
    hoy = suma(live,inicio,fin.floor('h'))
    semana = suma(hist,inicio-pd.Timedelta(days=7),inicio)
    trend = pd.DataFrame({'category':cat.category,'hoy':hoy,'semana':semana})
    trend['ratio_cuota'] = ((hoy+5)/(hoy.sum()+5*len(ids)))/((semana+5)/(semana.sum()+5*len(ids)))
    trend['ratio_suavizado'] = (trend.hoy+5)/(trend.semana*(fin.floor('h')-inicio).total_seconds()/3600/168+5)
    trend.sort_values('ratio_cuota',ascending=False).to_csv(OUT/'tendencias.csv')
    resultado = dict(inicio=str(inicio),fin=str(fin),cortes=metricas.corte.nunique(),backtest=resumen.reset_index().to_dict('records'),efectos=efecto_res,precios=dict(n=len(pr),subidas=int((pr.ratio>1).sum()),bajada_mediana=float(1-pr.ratio.median()),ratio_inicial_mediano=float(pr.ratio_inicial.median()),productos_repetidos=int(pr.product_id.duplicated().sum())),stock=dict(completos=len(duraciones),mediana_min=float(np.median(duraciones)),p90_min=float(np.percentile(duraciones,90)),sin_retorno=int((up.change=='out_of_stock').sum()-len(duraciones))),top_tendencias=trend.sort_values('ratio_cuota',ascending=False).head(10).reset_index().to_dict('records'))
    (OUT/'resultados.json').write_text(json.dumps(resultado,indent=2,ensure_ascii=False))
    print(json.dumps(resultado,indent=2,ensure_ascii=False))


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cached',action='store_true')
    args=ap.parse_args()
    if not args.cached: descargar()
    analizar()
