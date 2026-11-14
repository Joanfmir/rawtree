"""Informe reproducible de solo lectura: python scripts/report.py (horas UTC)."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Caché fuera del repositorio, también con HOME de solo lectura.
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'rawtree-report-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from reco import config
from reco.client import query
from reco.catalog import CURRENT_CATALOG

OUT = ROOT / 'docs' / 'figs'
CONSULTAS = []


def leer(sql):
    """Nunca escribe en RawTree; limita cada descarga y conserva el SQL exacto."""
    CONSULTAS.append(sql)
    filas = query(sql + '\nLIMIT 499999', timeout=180)
    if len(filas) >= 499999:
        raise RuntimeError('Consulta truncada: reducir el intervalo antes de continuar.')
    return filas


def tabla(df):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    for row in df.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v) for v in row) + ' |')
    return '\n'.join(lines)


def guardar(fig, nombre):
    fig.savefig(OUT / f'{nombre}.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def eje_tiempo(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M', tz=timezone.utc))
    ax.set_xlabel('Hora UTC (CET = UTC + 1)')
    ax.grid(alpha=.25)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--since', help='Inicio inclusivo, ISO UTC; por defecto todo el feedback')
    ap.add_argument('--until', help='Fin exclusivo, ISO UTC; por defecto máximo feedback + 1 ms')
    args = ap.parse_args()
    config.check()
    OUT.mkdir(parents=True, exist_ok=True)
    t = config.TEAM
    limites = leer(f"SELECT min(ts::DateTime64(3)) AS inicio, max(ts::DateTime64(3)) AS fin FROM {t}.feedback WHERE request_id::String != '__init__'")[0]
    inicio = pd.to_datetime(args.since or limites['inicio'], utc=True)
    fin = pd.to_datetime(args.until, utc=True) if args.until else pd.to_datetime(limites['fin'], utc=True) + pd.Timedelta(milliseconds=1)
    if fin <= inicio:
        ap.error('--until debe ser posterior a --since')
    registros = []
    vistos = set()
    catalogo = {r['product_id']: r['category'] for r in leer(CURRENT_CATALOG)}
    duplicados = 0
    for a in pd.date_range(inicio, fin, freq='1h', inclusive='left'):
        b = min(a + pd.Timedelta(hours=1), fin)
        filtro = f"request_id::String != '__init__' AND ts::DateTime64(3) >= '{a.tz_localize(None)}' AND ts::DateTime64(3) < '{b.tz_localize(None)}'"
        ids = f'SELECT request_id::String FROM {t}.feedback WHERE {filtro}'
        feedback = leer(f"SELECT request_id::String AS id, ts::DateTime64(3) AS ts, shown::Array(String) AS shown, served_by::String AS served_by, fallback_reason::String AS motivo, latency_ms::Float64 AS latencia, clicked::String AS clicked, purchased::String AS purchased FROM {t}.feedback WHERE {filtro}")
        req = {r['id']: r for r in leer(f'SELECT request_id::String AS id, recent_views::Array(JSON) AS vistas FROM shop.requests WHERE request_id::String IN ({ids})')}
        envios = defaultdict(list)
        for s in leer(f'SELECT request_id::String AS id, items::Array(String) AS items, model_version::String AS version FROM {t}.submissions WHERE request_id::String IN ({ids})'):
            envios[s['id']].append(s)
        for f in feedback:
            if f['id'] in vistos:
                duplicados += 1
                continue
            vistos.add(f['id'])
            versiones = {s['version'] for s in envios[f['id']] if s['items'] == f['shown']}
            f['version'] = ('fallback' if f['served_by'] != 'team' else next(iter(versiones)) if len(versiones) == 1 else 'sin atribución inequívoca')
            f['vistas'] = req.get(f['id'], {}).get('vistas', [])
            f['peticion_encontrada'] = f['id'] in req
            registros.append(f)
        print(f'{a} → {b}: {len(feedback)} feedback', flush=True)
    if not registros:
        raise SystemExit('No hay feedback en el intervalo solicitado.')
    df = pd.DataFrame(registros)
    df['ts'] = pd.to_datetime(df.ts, utc=True)
    df['ventana'] = df.ts.dt.floor('15min')
    df['clic'] = df.clicked.ne('').astype(int)
    df['compra'] = df.purchased.ne('').astype(int)
    df['a_tiempo'] = df.served_by.eq('team').astype(int)
    g = df.groupby(['ventana', 'version']).agg(n=('id', 'size'), clics=('clic', 'sum'), compras=('compra', 'sum')).reset_index()
    g['CTR_pct'] = 100 * g.clics / g.n
    g['conversion_pct'] = 100 * g.compras / g.n
    plt.rcParams.update({'font.size': 13, 'axes.titlesize': 17, 'figure.figsize': (13, 7)})
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    for version, d in g.groupby('version'):
        d = d.set_index('ventana').reindex(pd.date_range(df.ventana.min(), df.ventana.max(), freq='15min')).rename_axis('ventana').reset_index()
        for ax, col in zip(axs, ['conversion_pct', 'CTR_pct', 'n']):
            ax.plot(d.ventana, d[col], marker='o', label=version)
    for ax, titulo in zip(axs, ['Compra por petición (%)', 'Clic por petición (%)', 'Peticiones con feedback']):
        ax.set_ylabel(titulo); eje_tiempo(ax)
    axs[0].legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
    axs[0].set_title('Resultados observados por versión · ventanas de 15 minutos')
    fig.tight_layout(); guardar(fig, 'conversion_ctr')
    lat = df.groupby('ventana').agg(n=('id', 'size'), p50_ms=('latencia', 'median'), p95_ms=('latencia', lambda s: s.quantile(.95)), a_tiempo_pct=('a_tiempo', lambda s: 100 * s.mean())).reset_index()
    # El fallback missing puede tener latencia centinela: percentiles solo de respuestas team.
    team_lat = df[df.served_by.eq('team')].groupby('ventana').latencia.quantile([.5, .95]).unstack()
    lat['p50_ms'] = lat.ventana.map(team_lat.get(.5, pd.Series(dtype=float)))
    lat['p95_ms'] = lat.ventana.map(team_lat.get(.95, pd.Series(dtype=float)))
    fig, axs = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    axs[0].plot(lat.ventana, lat.p50_ms, label='p50 (team)'); axs[0].plot(lat.ventana, lat.p95_ms, label='p95 (team)')
    axs[0].axhline(3000, color='red', linestyle='--', label='Límite: 3 s'); axs[0].set_ylabel('Latencia (ms)'); axs[0].legend()
    axs[1].plot(lat.ventana, lat.a_tiempo_pct); axs[1].set_ylim(0, 101); axs[1].set_ylabel('Válidas y a tiempo (%)')
    for ax in axs: eje_tiempo(ax)
    axs[0].set_title('Fiabilidad · todas las peticiones con feedback')
    fig.tight_layout(); guardar(fig, 'latencia')
    relaciones, slots = defaultdict(lambda: [0, 0, 0]), defaultdict(lambda: [0, 0, 0])
    sin_vistas = desconocidas = 0
    for f in registros:
        vistas = [json.loads(v) if isinstance(v, str) else v for v in f['vistas']]
        productos = {v['product_id'] for v in vistas}
        cats = {catalogo[p] for p in productos if p in catalogo}
        sin_vistas += not bool(productos)
        for slot, p in enumerate(f['shown'], 1):
            valores = [1, int(p == f['clicked']), int(p == f['purchased'])]
            slots[slot] = [x + y for x, y in zip(slots[slot], valores)]
            if not productos or not f['peticion_encontrada']:
                continue
            if p in productos: relacion = 'Recién visto'
            elif p not in catalogo or len(cats) == 0 or any(v not in catalogo for v in productos):
                desconocidas += 1
                continue
            else: relacion = 'Categoría vista' if catalogo[p] in cats else 'Otra categoría'
            relaciones[relacion] = [x + y for x, y in zip(relaciones[relacion], valores)]
    def tasas(datos, clave):
        d = pd.DataFrame([(k, *v) for k, v in datos.items()], columns=[clave, 'impresiones', 'clics', 'compras'])
        d['CTR_pct'] = 100 * d.clics / d.impresiones
        d['conversion_pct'] = 100 * d.compras / d.impresiones
        return d
    rel = tasas(relaciones, 'relacion').set_index('relacion').reindex(['Recién visto', 'Categoría vista', 'Otra categoría']).reset_index()
    pos = tasas(slots, 'slot').sort_values('slot')
    fig, ax = plt.subplots()
    ax.bar(rel.relacion, rel.conversion_pct, color=['#167d9a', '#40a58b', '#db9953'])
    for i, row in rel.iterrows(): ax.text(i, row.conversion_pct, f'{row.conversion_pct:.3f}%\nn={row.impresiones:,.0f}', ha='center', va='bottom')
    ax.set_ylim(0, max(rel.conversion_pct.max() * 1.4, .01)); ax.set_ylabel('Compras / impresiones (%)'); ax.set_title('La intención de esta sesión importa'); ax.grid(axis='y', alpha=.2)
    guardar(fig, 'relacion_vistas')
    fig, ax = plt.subplots()
    ax.plot(pos.slot, pos.CTR_pct, marker='o', label='Clic'); ax.plot(pos.slot, pos.conversion_pct, marker='o', label='Compra')
    ax.set_xticks(pos.slot); ax.set_xlabel('Posición en el carrusel'); ax.set_ylabel('Eventos / impresiones (%)'); ax.set_title('Resultados observados según posición'); ax.legend(); ax.grid(alpha=.2)
    guardar(fig, 'posicion')
    reparto = g.pivot(index='ventana', columns='version', values='n').fillna(0)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.stackplot(reparto.index, reparto.to_numpy().T, labels=reparto.columns)
    ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1)); ax.set_ylabel('Peticiones con feedback'); ax.set_title('Qué versión se mostró realmente'); eje_tiempo(ax)
    guardar(fig, 'versiones')
    tot = df.groupby('version').agg(n=('id', 'size'), clics=('clic', 'sum'), compras=('compra', 'sum'), inicio=('ts', 'min'), fin=('ts', 'max')).reset_index()
    tot['CTR_pct'] = 100 * tot.clics / tot.n; tot['conversion_pct'] = 100 * tot.compras / tot.n
    resumen = f'''# Cifras del informe\n\nGenerado: {datetime.now(timezone.utc).isoformat()}. Equipo: {t}.\nIntervalo consultado: [{inicio}, {fin}). Ventanas de 15 minutos por **timestamp de feedback**, UTC; extremos potencialmente parciales.\n\nFuente: SELECT mediante reco.client.query. SQL exacto en consultas.sql; tablas agregadas en CSV. No se escribe en RawTree.\n\nPeticiones: {len(df)}; clics: {df.clic.sum()}; compras: {df.compra.sum()}. CTR: {100*df.clic.mean():.4f}%; conversión observada: {100*df.compra.mean():.4f}%.\nVálidas y a tiempo (served_by=team): {100*df.a_tiempo.mean():.4f}%. Duplicados de feedback descartados por ID: {duplicados}.\n\n## Conversión, CTR y reparto por versión\n\n{tabla(tot)}\n\nLas versiones se atribuyen solo si items coincide exactamente, incluido el orden, con feedback.shown y hay una sola etiqueta distinta coincidente. Fallback se separa. No se multiplica el feedback por envíos duplicados. Comparar periodos distintos no demuestra mejora causal.\n\n## Latencia y fiabilidad por ventana\n\n{tabla(lat)}\n\nPercentiles empíricos pandas (interpolación lineal), únicamente served_by=team. Fiabilidad usa todo el feedback, incluidos missing, late e invalid. No mide peticiones cuyo feedback aún no haya llegado.\n\n## Relación con vistas\n\n{tabla(rel)}\n\nGrupos excluyentes: recién visto = cualquier producto en recent_views; categoría vista excluye recién vistos. Solo sesiones con vistas y petición encontrada. Sin vistas: {sin_vistas} peticiones; sin petición encontrada: {sum(not f['peticion_encontrada'] for f in registros)}; impresiones con categoría desconocida excluidas: {desconocidas}. Catálogo vigente al generar el informe; no reconstruye cambios históricos de categoría. Asociación descriptiva, confundida por posición y política.\n\n## Posición\n\n{tabla(pos)}\n\nDenominador: una impresión por elemento de feedback.shown, incluido fallback. La asignación de posiciones no es aleatoria: esto no estima el efecto causal de mover un producto.\n\n## Reparto temporal\n\nVer versiones.csv y versiones.png (conteos por ventana).\n\n## Pendientes\n\nResultado final y score oficial: PENDIENTE. Las compras realizadas no son la probabilidad esperada con la que puntúa el simulador.\n'''
    for nombre, d in [('conversion_ctr', g), ('latencia', lat), ('relacion_vistas', rel), ('posicion', pos), ('versiones', reparto.reset_index()), ('totales_version', tot)]:
        d.to_csv(OUT / f'{nombre}.csv', index=False)
    (OUT / 'resumen.md').write_text(resumen)
    (OUT / 'consultas.sql').write_text(';\n\n'.join(CONSULTAS) + ';\n')
    print(tot.to_string(index=False)); print(f'Informe generado en {OUT}')


if __name__ == '__main__':
    main()
