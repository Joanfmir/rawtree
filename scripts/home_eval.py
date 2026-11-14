"""Análisis reproducible de home. Solo SELECT; resultados y cachés en data/home_eval/."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from reco import features
from reco.client import query
from reco.config import DATA_DIR, TEAM
from reco.state import State
from reco.safety import finalize

OUT = DATA_DIR / 'home_eval'
CORTE = '2026-11-07'


def cache(nombre, sql):
    OUT.mkdir(parents=True, exist_ok=True)
    ruta = OUT / (nombre + '.pkl')
    if ruta.exists():
        return pd.read_pickle(ruta)
    df = pd.DataFrame(query(sql, timeout=600))
    if len(df) >= 499999:
        raise RuntimeError('Consulta truncada: particionar antes de continuar')
    df.to_pickle(ruta)
    return df


def features_evaluacion(live):
    ruta = OUT / ('features_live_eval.pkl' if live else 'features_hist_eval.pkl')
    if ruta.exists():
        return pd.read_pickle(ruta)
    OUT.mkdir(parents=True,exist_ok=True)
    f = features.load(None if live else CORTE)
    pd.to_pickle(f,ruta)
    return f


def sesiones(live):
    pref = 'r' if live else 'h'
    cols = f"""{pref}.request_id::String AS request_id, {pref}.ts::String AS ts,
    {pref}.user_id::String AS user_id, {pref}.context.device::String AS device,
    {pref}.context.region::String AS region, {pref}.context.page::String AS page"""
    if live:
        sql = f"""SELECT {cols}, r.is_new_user::Bool AS is_new_user,
        f.shown::Array(String) AS shown, f.clicked::String AS clicked, f.purchased::String AS purchased
        FROM {TEAM}.feedback f INNER JOIN shop.requests r ON r.request_id::String=f.request_id::String
        WHERE length(r.recent_views::Array(JSON))=0 AND f.request_id::String!='__init__'
        AND f.served_by::String='team' LIMIT 499999"""
    else:
        sql = f"""SELECT {cols}, h.shown::Array(String) AS shown,
        h.propensity::Array(Float64) AS propensity, h.clicked::String AS clicked,
        h.purchased::String AS purchased FROM shop.history_sessions h
        WHERE h.ts::DateTime64(3)>='{CORTE}' AND h.context.page::String='home' LIMIT 499999"""
    df = cache('live' if live else 'historico', sql)
    for col in ['clicked', 'purchased']:
        df[col] = df[col].fillna('')
    df['context'] = [dict(device=d, region=r, page=p) for d,r,p in zip(df.device,df.region,df.page)]
    df['recent_views'] = [[] for _ in range(len(df))]
    return df


def indices(f):
    it = f['item_features'].set_index('product_id')
    cp = f['context_popular']
    ctx = {(d,r,p):float(n) for d,r,p,n in cp[['device','region','product_id','n']].itertuples(index=False,name=None)}
    uc = f['user_cat'].sort_values('w',ascending=False)
    fav = uc.groupby('user_id').category.apply(list).to_dict()
    bought = set(zip(f['user_purchases'].user_id,f['user_purchases'].product_id))
    return it,ctx,fav,bought


def productos_previos(live):
    """Solo pares usuario/producto efectivamente expuestos; nunca descarga eventos en bruto."""
    if live:
        pares = f"""SELECT DISTINCT r.user_id::String AS u,arrayJoin(f.shown::Array(String)) AS p
        FROM {TEAM}.feedback f INNER JOIN shop.requests r ON r.request_id::String=f.request_id::String
        WHERE length(r.recent_views::Array(JSON))=0 AND f.served_by::String='team'
        AND f.request_id::String!='__init__'"""
    else:
        pares = f"""SELECT DISTINCT user_id::String AS u,arrayJoin(shown::Array(String)) AS p
        FROM shop.history_sessions WHERE ts::DateTime64(3)>='{CORTE}' AND context.page::String='home'"""
    corte = '2026-11-14' if live else CORTE
    return cache('pares_previos_live_v2' if live else 'pares_previos_hist_v2',f"""
    SELECT e.user_id::String AS user_id,e.product_id::String AS product_id,
    countIf(e.event_type::String='view') AS vistas,
    countIf(e.event_type::String='add_to_cart') AS carritos,
    countIf(e.event_type::String='purchase') AS compras_organicas,
    countIf(e.event_type::String IN ('purchase','carousel_purchase')) AS compras_totales
    FROM (SELECT user_id::String AS user_id,product_id::String AS product_id,event_type::String AS event_type,ts::DateTime64(3) AS ts FROM shop.history_events
    UNION ALL SELECT user_id::String,purchased::String,'carousel_purchase',ts::DateTime64(3) FROM shop.history_sessions WHERE purchased::String!='') e INNER JOIN ({pares}) p
    ON e.user_id::String=p.u AND e.product_id::String=p.p
    WHERE e.ts::DateTime64(3)<'{corte}'
    GROUP BY e.user_id::String,e.product_id::String LIMIT 499999""")


def impresiones(df,f,live):
    it,ctx,fav,bought = indices(f)
    x = df.explode('shown').rename(columns={'shown':'product_id'})
    x['slot'] = x.groupby(level=0).cumcount()+1
    x = x.join(it[['category','purchases_7d','buy_rate','price_eur']],on='product_id')
    x['ctx_pop'] = [ctx.get((d,r,p),0) for d,r,p in zip(x.device,x.region,x.product_id)]
    x['favorita'] = [c in fav.get(u,[])[:1] and not n for c,u,n in zip(x.category,x.user_id,x.is_new_user)]
    x['comprado_antes'] = [(u,p) in bought and not n for u,p,n in zip(x.user_id,x.product_id,x.is_new_user)]
    previos = productos_previos(live)
    for campo in ['vistas','carritos','compras_organicas','compras_totales']:
        pares = set(zip(previos.loc[previos[campo]>0,'user_id'],previos.loc[previos[campo]>0,'product_id']))
        x[campo] = [(u,p) in pares and not n for u,p,n in zip(x.user_id,x.product_id,x.is_new_user)]
    x['comprado_antes'] = x['compras_totales']
    x['hora_CET'] = (pd.to_datetime(x.ts,utc=True).dt.hour+1)%24
    x['clic'] = x.product_id == x.clicked
    x['compra'] = x.product_id == x.purchased
    tablas = {}
    for col in ['purchases_7d','buy_rate','ctx_pop','price_eur','favorita','comprado_antes','vistas','carritos','compras_organicas','hora_CET','slot']:
        grupo = x[col] if col in ['favorita','comprado_antes','vistas','carritos','compras_organicas','hora_CET','slot'] else pd.qcut(x[col],4,duplicates='drop')
        g = x.groupby([x.is_new_user,grupo],observed=True).agg(n=('compra','size'),clics=('clic','sum'),compras=('compra','sum'))
        g['ctr'] = g.clics/g.n
        g['cvr'] = g.compras/g.n
        tablas[col] = g.reset_index().astype({col:str}).to_dict('records')
    cats = df.shown.apply(lambda s: [it.category.get(p,'?') for p in s])
    tablas['sistema_viejo'] = {'una_categoria':float(cats.apply(lambda c:len(set(c))==1).mean()),
        'distribucion_categoria':cats.apply(lambda c:c[0]).value_counts(normalize=True).to_dict(),
        'categoria_favorita_top1':float(np.mean([c[0] in fav.get(u,[])[:1] for c,u in zip(cats,df.user_id)]))}
    marginal = cats.apply(lambda c:c[0]).value_counts(normalize=True)
    favoritas = df.user_id.map(lambda u: fav.get(u,[None])[0]).value_counts(normalize=True)
    tablas['sistema_viejo']['acierto_favorita_si_independiente'] = float((marginal*favoritas).sum())
    return tablas


def metricas(df,recs):
    out = {'n':len(df)}
    for col in ['clicked','purchased']:
        mask = df[col].ne('').to_numpy()
        targets = df.loc[mask,col].tolist()
        hit = np.array([t in rr for t,rr in zip(targets,[r for r,m in zip(recs,mask) if m])])
        pos = np.array([rr.index(t)+1 if t in rr else 0 for t,rr in zip(targets,[r for r,m in zip(recs,mask) if m])])
        out[col+'_ndcg10'] = float(np.mean(np.where(pos>0,1/np.log2(np.maximum(pos,1)+1),0))) if len(hit) else None
        out[col+'_n'] = len(hit)
        out[col+'_hit10'] = float(hit.mean()) if len(hit) else None
        if 'propensity' in df and len(hit):
            weights = np.array([1/max(float(ps[ss.index(t)]),.001) for ps,ss,t in zip(df.loc[mask,'propensity'],df.loc[mask,'shown'],targets)])
            out[col+'_ips_ess'] = float(weights.sum()**2/(weights**2).sum())
            out[col+'_snips_hit10'] = float(np.sum(weights*hit)/weights.sum())
    return out


def evaluar(df,f,until):
    from reco.policies.home import HomeBest
    from reco.policies.personal import Personal, PersonalV3
    from reco.policies.ranker import Ranker
    Personal._shared[until] = Personal._index(f)
    it = f['item_features']
    state = State(catalog={p:{'is_active':bool(a)} for p,a in zip(it.product_id,it.is_active)}, best=it.sort_values('purchases',ascending=False).product_id.tolist())
    politicas = {'home_best':HomeBest(until),'personal_v3':PersonalV3(until), 'gulias':Ranker(until)}
    for modo in ['global','contexto','mixta','favorita','diversa','una_categoria']:
        politicas[modo] = HomeBest(until, estrategia=modo)
    res = {}
    # Sin respuesta observada no hay hit que medir; mantener denominadores en el análisis de impresiones.
    df = df[df.clicked.ne('') | df.purchased.ne('')].reset_index(drop=True)
    for nombre,p in politicas.items():
        try:
            if nombre == 'gulias' and until and not (DATA_DIR/f'ranker_until_{until}.txt').exists():
                raise FileNotFoundError('Falta modelo ranker del corte; entrenar con scripts/train_ranker.py --until '+until)
            p.load_features = lambda: f
            p.fit(state)
            t = time.perf_counter()
            rr = [finalize(p.recommend(r,state),state) for r in df.to_dict('records')]
            ms = (time.perf_counter()-t)*1000/max(len(df),1)
            res[nombre] = {}
            for seg,mask in [('todos',np.ones(len(df),bool)),('nuevos',df.is_new_user.to_numpy()),('conocidos',~df.is_new_user.to_numpy())]:
                res[nombre][seg] = metricas(df[mask],[r for r,m in zip(rr,mask) if m])
            pares = set(zip(f['user_purchases'].user_id,f['user_purchases'].product_id))
            no_rebuy = np.array([(u,p) not in pares for u,p in zip(df.user_id,df.purchased)])
            if until:
                res[nombre]['sin_recompra'] = metricas(df[no_rebuy],[r for r,m in zip(rr,no_rebuy) if m])
            res[nombre]['ms_req'] = ms
            print(nombre,json.dumps(res[nombre]),flush=True)
        except Exception as e:
            res[nombre] = {'error':str(e)}
            print(nombre,str(e),flush=True)
    return res


def siguientes(live, df, f):
    """Primer evento por sesión y perfiles fijos anteriores al corte, sin ver los objetivos."""
    tabla = 'shop.events' if live else 'shop.history_events'
    filtro = f"WHERE ts::DateTime64(3)<='{(pd.to_datetime(df.ts,utc=True).max()+pd.Timedelta('30min')).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}'" if live else f"WHERE ts::DateTime64(3)>='{CORTE}'"
    ev = cache('primeros_live' if live else 'primeros_hist', f"""
    SELECT session_id::String AS session_id, argMin(user_id::String,ts::DateTime64(3)) AS user_id,
    min(ts::DateTime64(3)) AS inicio, argMin(device::String,ts::DateTime64(3)) AS device,
    argMin(region::String,ts::DateTime64(3)) AS region,
    argMinIf(product_id::String,ts::DateTime64(3),event_type::String='view') AS vista,
    argMinIf(product_id::String,ts::DateTime64(3),event_type::String='purchase') AS compra
    FROM {tabla} {filtro} GROUP BY session_id LIMIT 499999""")
    previo = cache('ultimos_pre_corte', f"""SELECT user_id::String AS user_id,
    argMaxIf(product_id::String,ts::DateTime64(3),event_type::String='view') AS ultima_vista,
    argMaxIf(product_id::String,ts::DateTime64(3),event_type::String='add_to_cart') AS ultimo_carrito
    FROM shop.history_events WHERE ts::DateTime64(3)<'{CORTE}' GROUP BY user_id LIMIT 499999""")
    train = cache('contexto_sesion_train', f"""SELECT device,region,hora,producto,count() AS n FROM (
    SELECT argMin(device::String,ts::DateTime64(3)) AS device,
    argMin(region::String,ts::DateTime64(3)) AS region,
    toHour(min(ts::DateTime64(3))) AS hora,
    argMinIf(product_id::String,ts::DateTime64(3),event_type::String='view') AS producto
    FROM shop.history_events WHERE ts::DateTime64(3)<'{CORTE}'
    AND ts::DateTime64(3)>='2026-10-31' GROUP BY session_id::String)
    GROUP BY device,region,hora,producto LIMIT 499999""")
    ev = ev.rename(columns={'inicio':'ts'})
    base = features.load(CORTE)
    it,ctx,fav,bought = indices(base)
    categoria = it.category.to_dict()
    train['categoria'] = train.producto.map(categoria)
    previo = previo.set_index('user_id')
    ev['hora'] = pd.to_datetime(ev.ts,utc=True).dt.hour
    # request carece de session_id: vínculo aproximado, mismo usuario, después de home y <30 min.
    a = ev.copy()
    a['tiempo'] = pd.to_datetime(a.ts,utc=True).astype('datetime64[ns, UTC]')
    b = df[['user_id','ts']].copy()
    b['categoria_mostrada'] = df.shown.apply(lambda ps:categoria.get(ps[0]) if ps else None)
    b['tiempo_home'] = pd.to_datetime(b.ts,utc=True).astype('datetime64[ns, UTC]')
    vinc = pd.merge_asof(a.sort_values('tiempo'),b[['user_id','tiempo_home','categoria_mostrada']].sort_values('tiempo_home'),
        left_on='tiempo',right_on='tiempo_home',by='user_id',direction='backward',tolerance=pd.Timedelta('30min'))
    ev['vinculado_home'] = ev.session_id.isin(vinc.loc[vinc.tiempo_home.notna(),'session_id'])
    pred = {'favorita':{u:cs[:3] for u,cs in fav.items()}}
    for col in ['ultima_vista','ultimo_carrito']:
        pred[col] = {u:[categoria.get(p)] for u,p in previo[col].items() if p in categoria}
    tablas = {}
    for campos in [['device'],['region'],['hora'],['device','region'],['device','region','hora']]:
        g = train.groupby(campos+['categoria'],as_index=False).n.sum().sort_values('n',ascending=False)
        tablas['+'.join(campos)] = (campos,{k if isinstance(k,tuple) else (k,):list(v.categoria)[:3] for k,v in g.groupby(campos)})
    globales = train.groupby('categoria').n.sum().sort_values(ascending=False).index.tolist()[:3]
    acierto_viejo = {}
    for objetivo in ['vista','compra']:
        vv = vinc[vinc.tiempo_home.notna() & vinc[objetivo].isin(categoria)]
        acierto_viejo[objetivo] = {'n':len(vv),'acierto_categoria':float((vv[objetivo].map(categoria)==vv.categoria_mostrada).mean()) if len(vv) else None}
    salida = {'categoria_mostrada_vs_siguiente':acierto_viejo,'sesiones':len(ev),'vinculadas_home':int(ev.vinculado_home.sum()),'precision':{}}
    for target in ['vista','compra']:
        valido = ev[ev[target].isin(categoria)]
        for segmento,sub in [('todos',valido),('home_vinculado',valido[valido.vinculado_home]),
                               ('conocidos',valido[valido.user_id.isin(fav)]),('nuevos',valido[~valido.user_id.isin(fav)])]:
            objetivos = sub[target].map(categoria).tolist()
            pp = {n:[dic.get(u,globales) for u in sub.user_id] for n,dic in pred.items()}
            for n in ['ultima_vista','ultimo_carrito']:
                pp[n] = [list(dict.fromkeys(ps+fav.get(u,globales)))[:3] for ps,u in zip(pp[n],sub.user_id)]
            pp['global'] = [globales]*len(sub)
            for n,(campos,dic) in tablas.items():
                pp[n] = [dic.get(tuple(row),globales) for row in sub[campos].itertuples(index=False,name=None)]
            salida['precision'][target+'_'+segmento] = {n:{'n':len(sub),'top1':float(np.mean([c in p[:1] for c,p in zip(objetivos,ps)])) if len(sub) else None,
                'top3':float(np.mean([c in p[:3] for c,p in zip(objetivos,ps)])) if len(sub) else None} for n,ps in pp.items()}
    return salida


def informe():
    """Resumen portable; el JSON conserva todas las tasas, tamaños y segmentos."""
    lineas = ['# Evaluación home', '',
        'Estrategia elegida: popularidad global 7d, sin limitar categorías y sin bonus de recompra. '
        'Con vistas delega en personal_v3; variantes de contexto/perfil solo para comparar.', '',
        'Las features de evaluación se congelan en features_*_eval.pkl. '
        'El histórico usa corte 2026-11-07. Live es retrospectivo: su snapshot incluye eventos '
        'posteriores a algunas peticiones; no demuestra una mejora causal.', '',
        'SNIPS-hit pondera el acierto entre respuestas positivas por 1/max(propensity,0.001). '
        'No estima conversión contrafactual ni corrige la exposición de una lista completa: '
        'las propensities del log son condicionales al prefijo original. '
        'El ESS de compras histórico es pequeño. No hay propensity en vivo.', '']
    porcentaje = lambda v: '—' if v is None else f'{100*v:.2f}%'
    for nombre in ['historico','live']:
        ruta = OUT/(nombre+'_resultados.json')
        if not ruta.exists():
            continue
        d = json.loads(ruta.read_text())
        lineas += [f'## {nombre}: {d["n"]} peticiones', '',
            '| Política | Compra hit@10 | Compra SNIPS | Clic hit@10 | Clic SNIPS | ms/req |',
            '|---|---:|---:|---:|---:|---:|']
        for pol,datos in d['evaluacion'].items():
            if 'error' in datos:
                lineas.append(f'| {pol} | ERROR: {datos["error"]} | | | | |')
                continue
            m = datos['todos']
            lineas.append('| '+pol+' | '+' | '.join(porcentaje(m.get(k)) for k in
                ['purchased_hit10','purchased_snips_hit10','clicked_hit10','clicked_snips_hit10'])+f' | {datos["ms_req"]:.3f} |')
        lineas += ['', '| Política | Compras conocidos | Compras nuevos | Clics conocidos | Clics nuevos |',
            '|---|---:|---:|---:|---:|']
        for pol,datos in d['evaluacion'].items():
            if 'error' in datos:
                continue
            lineas.append('| '+pol+' | '+' | '.join(porcentaje(datos[seg].get(k)) for seg,k in
                [('conocidos','purchased_hit10'),('nuevos','purchased_hit10'),('conocidos','clicked_hit10'),('nuevos','clicked_hit10')])+' |')
        extra = OUT/(nombre+'_siguientes.json')
        if extra.exists():
            e = json.loads(extra.read_text())
            lineas += ['',f'Sesiones orgánicas: {e["sesiones"]}; vinculadas aproximadamente a home: {e["vinculadas_home"]}.',
                'El vínculo usa mismo usuario y primer evento posterior a home a menos de 30 minutos. '
                'No hay session_id en requests: el vínculo no es una atribución exacta. '
                'Última vista/carrito usa la categoría de ese producto para top-1, y completa top-3 con favoritas sin repetir.', '',
                '| Predictor | Primera vista top1/top3 | Primera compra top1/top3 |','|---|---:|---:|']
            for predictor in e['precision']['vista_home_vinculado']:
                vals=[]
                for target in ['vista','compra']:
                    m=e['precision'][target+'_home_vinculado'][predictor]
                    vals.append(porcentaje(m['top1'])+' / '+porcentaje(m['top3']))
                lineas.append('| '+predictor+' | '+' | '.join(vals)+' |')
            lineas += ['', 'Acierto categoría mostrada frente a siguiente categoría: '+json.dumps(e.get('categoria_mostrada_vs_siguiente',{})), '']
        imp = d['impresiones']
        lineas += ['', '| Señal / grupo | Nuevo | Impresiones | Clics | Compras | CTR | CVR |', '|---|---|---:|---:|---:|---:|---:|']
        for senal,filas in imp.items():
            if not isinstance(filas,list):
                continue
            for r in filas:
                lineas.append(f'| {senal}: {r[senal]} | {r["is_new_user"]} | {r["n"]} | {r["clics"]} | {r["compras"]} | {porcentaje(r["ctr"])} | {porcentaje(r["cvr"])} |')
        lineas += ['', 'Sistema antiguo / distribución de categoría: '+json.dumps(imp['sistema_viejo'],ensure_ascii=False), '']
    prueba = OUT/'comprobacion.json'
    if prueba.exists():
        lineas += ['', 'Comprobación local: '+prueba.read_text(), '']
    log = OUT/'dry_run.log'
    if log.exists():
        lineas += ['', 'Dry-run (ventana de stats de 60 s):']+[linea for linea in log.read_text().splitlines() if '[stats]' in linea]+['']
    lineas += ['## Límites', '',
        'No afirmar un ganador estadístico con 17 compras live. La selección de estrategia y la evaluación '
        'usan la misma muestra; falta validación futura. Histórico y live divergen en recompra. '
        'Los flags visto/carrito/comprado usan solo información anterior al inicio del periodo '
        '(7/11 histórico, 14/11 live), excluyendo novedades durante el periodo. '
        'La etiqueta nuevo en histórico/eventos significa sin perfil previo; en feedback usa is_new_user. '
        'Las tasas por señal no controlan posición, política ni otras variables. '
        'No confundir asociación con efecto. Una categoría favorita no equivale a intención de sesión. '
        'La lista global ya incluye varias categorías; imponer una cuota de diversidad tampoco mejora el replay.', '']
    (OUT/'informe.md').write_text('\n'.join(lineas))
    print('Informe:',OUT/'informe.md',flush=True)


def dry_run(segundos):
    """Lanza y termina únicamente nuestro proceso; nunca escribe submissions."""
    import subprocess
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'dry_run.log').open('w') as log:
        p = subprocess.Popen([sys.executable,'-u','-m','reco.serve','--policy','home_best','--dry-run'],stdout=log,stderr=subprocess.STDOUT)
        try:
            p.wait(timeout=segundos+25)
        except subprocess.TimeoutExpired:
            p.terminate()
            p.wait(timeout=10)
    for linea in (OUT/'dry_run.log').read_text().splitlines():
        if '[stats]' in linea or '[error]' in linea:
            print(linea)


def comprobar():
    """Contrato de delegación, validez con cambios de stock y coste local de home."""
    from reco.policies.home import HomeBest
    f = features.load()
    it = f['item_features']
    state = State(catalog={p:{'is_active':bool(a)} for p,a in zip(it.product_id,it.is_active)},
                  best=it.sort_values('purchases',ascending=False).product_id.tolist())
    p = HomeBest()
    p.fit(state)
    req = {'user_id':'usuario_prueba','context':{'device':'ios','region':'Catalunya'},
           'recent_views':[],'is_new_user':True}
    rr = p.recommend(req,state)
    assert len(rr)==10 and len(set(rr))==10
    state.unavailable = set(rr[:7])
    rr2 = finalize(p.recommend(req,state),state)
    assert len(rr2)==10 and not set(rr2)&state.unavailable
    req['recent_views'] = [{'product_id':rr[0]}]
    assert p.recommend(req,state)==p.interna.recommend(req,state)
    req['recent_views'] = []
    tiempos=[]
    for _ in range(1000):
        inicio=time.perf_counter()
        p.recommend(req,state)
        tiempos.append((time.perf_counter()-inicio)*1000)
    res={'n':1000,'mediana_ms':float(np.median(tiempos)),
         'p95_ms':float(np.percentile(tiempos,95)),'max_ms':max(tiempos),'contratos':'OK'}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'comprobacion.json').write_text(json.dumps(res,indent=2))
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--comprobar',action='store_true')
    ap.add_argument('--informe',action='store_true')
    ap.add_argument('--dry-run-seconds',type=int)
    ap.add_argument('--solo-impresiones',action='store_true')
    ap.add_argument('--solo-siguientes',action='store_true')
    ap.add_argument('--fuente',choices=['live','historico','ambas'],default='ambas')
    args = ap.parse_args()
    if args.comprobar:
        comprobar()
        return
    if args.informe:
        informe()
        return
    if args.dry_run_seconds is not None:
        dry_run(args.dry_run_seconds)
        return
    for live in [False,True]:
        if args.fuente != 'ambas' and (args.fuente=='live') != live:
            continue
        nombre = 'live' if live else 'historico'
        f = features_evaluacion(live)
        df = sesiones(live)
        if not live:
            df['is_new_user'] = ~df.user_id.isin(f['user_cat'].user_id)
        if args.solo_impresiones:
            (OUT/(nombre+'_impresiones.json')).write_text(json.dumps(impresiones(df,f,live),indent=2,default=str))
            continue
        if args.solo_siguientes:
            (OUT/(nombre+'_siguientes.json')).write_text(json.dumps(siguientes(live,df,f),indent=2))
            continue
        print(nombre,len(df),'clics',df.clicked.ne('').sum(),'compras',df.purchased.ne('').sum(),flush=True)
        resumen = {'n':len(df),'impresiones':impresiones(df,f,live),'evaluacion':evaluar(df,f,None if live else CORTE),'siguientes':siguientes(live,df,f)}
        (OUT/(nombre+'_resultados.json')).write_text(json.dumps(resumen,indent=2,default=str))

    if not args.solo_siguientes and not args.solo_impresiones:
        informe()

if __name__ == '__main__':
    main()
