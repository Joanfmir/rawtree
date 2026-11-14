"""Preferencia relativa A/B: cuota A > 0,5 favorece A, con IC Wilson del 95 %.

El test de signos es binomial exacto bilateral sobre eventos atribuibles, por
separado para clic y compra. Un prefijo común no vota. No estima conversión
absoluta ni uplift: la mezcla cambia la lista y puede introducir interacciones.
La aleatorización equilibra posiciones en expectativa, no en cada petición.
Los segmentos se solapan (home y nuevo); muchos contrastes requieren corrección.
Clic y compra no son observaciones independientes entre sí. Usuarios repetidos
pueden invalidar la independencia binomial; para decisiones finales conviene
agrupar por usuario y confirmar en A/B. Un dry-run no tiene feedback propio.

--simulate usa un modelo sintético de un clic como máximo, base 1 %, A con
probabilidad 1+X veces B y exposición igual. No presupone ventaja estadística
de interleaving: con este modelo ambos diseños tienen información casi igual.
Se estima el N para potencia 80 % y alfa bilateral 5 % a horizonte fijo;
no es una garantía de potencia real ni permite parar al primer p < 0,05.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reco import config
from reco.client import query


def estadistica(a, b):
    from scipy.stats import binomtest
    n = a + b
    if not n:
        return dict(A=a, B=b, cuota_A=None, ic95=None, p=None)
    z = 1.959963984540054
    p = a / n
    centro = (p + z*z/(2*n)) / (1 + z*z/n)
    radio = z * (p*(1-p)/n + z*z/(4*n*n))**0.5 / (1 + z*z/n)
    return dict(A=a, B=b, cuota_A=p, ic95=[centro-radio, centro+radio],
                p=binomtest(a, n, 0.5).pvalue)


def resumir(logs, feedback):
    """Descarta duplicados ambiguos; cada petición aporta como máximo un voto/evento."""
    por_id = defaultdict(list)
    for r in logs:
        if not r.get('dry_run'):
            por_id[r['request_id']].append(r)
    grupos = {}
    aceptadas = 0
    for rid, registros in por_id.items():
        fs = feedback.get(rid, [])
        if len(registros) != 1 or len(fs) != 1:
            continue
        r, f = registros[0], fs[0]
        if (f.get('served_by') != 'team' or f.get('shown') != r.get('shown')
                or len(r.get('shown', [])) != 10 or len(set(r['shown'])) != 10
                or len(r.get('owners', [])) != 10
                or any(o not in ('A', 'B', 'ambas') for o in r['owners'])):
            continue
        aceptadas += 1
        segmentos = ['total', 'nuevo' if r.get('is_new_user') else 'conocido']
        if r.get('home'):
            segmentos.append('home')
        for segmento, slot in [(s, None) for s in segmentos] + [('slot', i) for i in range(1, 11)]:
            key = (r['policy_a'], r['policy_b'], segmento, slot)
            g = grupos.setdefault(key, dict(peticiones=0, clic=[0, 0], compra=[0, 0]))
            g['peticiones'] += 1
            for evento, campo in [('clic', 'clicked'), ('compra', 'purchased')]:
                producto = f.get(campo)
                if producto not in r['shown']:
                    continue
                pos = r['shown'].index(producto)
                autor = r['owners'][pos]
                if autor in ('A', 'B') and (slot is None or slot == pos+1):
                    g[evento][autor == 'B'] += 1
    out = []
    for (a, b, segmento, slot), g in grupos.items():
        out.append(dict(policy_a=a, policy_b=b, segmento=segmento, slot=slot,
                        peticiones=g['peticiones'],
                        clic=estadistica(*g['clic']), compra=estadistica(*g['compra'])))
    return aceptadas, out


def simular(repeticiones=3000):
    import numpy as np
    from scipy.stats import binom, norm
    rng = np.random.default_rng(17)
    print('Modelo: CTR B=1%, potencia objetivo 80%, alfa=5%, horizonte fijo; N total.')
    print('mejora_pct | peticiones_interleaving | peticiones_AB | potencia_IL | potencia_AB')
    for x in (.1, .2, .3):
        p = .01
        aproximado = (norm.ppf(.975)+norm.ppf(.8))**2 * 4*(1+x/2)/(p*x*x)
        resultados = []
        for modo in ('IL', 'AB'):
            for n in np.ceil(np.linspace(.8, 1.3, 26)*aproximado/2).astype(int)*2:
                if modo == 'IL':
                    muestras = rng.multinomial(int(n), [p*(1+x)/2, p/2, 1-p*(1+x/2)], size=repeticiones)
                    a, b = muestras[:, 0], muestras[:, 1]
                else:
                    a = rng.binomial(n//2, p*(1+x), repeticiones)
                    b = rng.binomial(n//2, p, repeticiones)
                valor = np.minimum(1, 2*binom.cdf(np.minimum(a, b), a+b, .5))
                potencia = float(np.mean(valor < .05))
                if potencia >= .8:
                    break
            resultados.append((int(n), potencia))
        print(f'{100*x:.0f} | {resultados[0][0]} | {resultados[1][0]} | {resultados[0][1]:.3f} | {resultados[1][1]:.3f}')


def fecha(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', choices=('local', 'scratch'), default='local')
    ap.add_argument('--log', type=Path, default=config.LOGS_DIR / 'interleave.jsonl')
    ap.add_argument('--since', default='2026-11-14T00:00:00Z')
    ap.add_argument('--until')
    ap.add_argument('--simulate', action='store_true')
    args = ap.parse_args()
    if args.simulate:
        simular()
        return
    since, until = fecha(args.since), fecha(args.until) if args.until else None
    cut = f"ts::DateTime64(3) >= '{since.strftime('%Y-%m-%d %H:%M:%S')}'"
    if until:
        cut += f" AND ts::DateTime64(3) < '{until.strftime('%Y-%m-%d %H:%M:%S')}'"
    if args.source == 'scratch':
        campos = ', '.join(f'{c}::{t} AS {c}' for c, t in (
            ('request_id', 'String'), ('shown', 'Array(String)'), ('owners', 'Array(String)'),
            ('policy_a', 'String'), ('policy_b', 'String'), ('ts', 'String'),
            ('dry_run', 'Bool'), ('is_new_user', 'Bool'), ('home', 'Bool')))
        logs = query(f"SELECT {campos} FROM {config.TEAM}.scratch WHERE kind::String = 'interleave' AND {cut} LIMIT 499999", timeout=60)
        if len(logs) >= 499999:
            raise SystemExit('Acota la ventana: se alcanzó el límite de filas.')
    else:
        logs = []
        if args.log.exists():
            with args.log.open() as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        ts = fecha(r['ts'])
                        if ts >= since and (until is None or ts < until):
                            logs.append(r)
                    except (ValueError, KeyError, TypeError):
                        print('Aviso: línea inválida omitida', file=sys.stderr)
    live = [r for r in logs if not r.get('dry_run')]
    print(f'Registros={len(logs)}, dry-run excluidos={len(logs)-len(live)}')
    feedback = defaultdict(list)
    ids = sorted({r['request_id'] for r in live if r['request_id'] != '__init__'})
    for start in range(0, len(ids), 500):
        quoted = ','.join("'"+s.replace('\\', '\\\\').replace("'", "\\'")+"'" for s in ids[start:start+500])
        rows = query(f"SELECT request_id::String AS request_id, shown::Array(String) AS shown, clicked::String AS clicked, purchased::String AS purchased, served_by::String AS served_by FROM {config.TEAM}.feedback WHERE request_id::String IN ({quoted}) LIMIT 499999", timeout=60)
        if len(rows) >= 499999:
            raise SystemExit('Demasiadas filas de feedback.')
        for r in rows:
            feedback[r['request_id']].append(r)
    n, resumen = resumir(live, feedback)
    print(f'Peticiones verificadas={n}/{len(live)}; sin coincidencia o ambiguas descartadas')
    for r in resumen:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == '__main__':
    main()
