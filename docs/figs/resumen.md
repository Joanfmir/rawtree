# Cifras del informe

Generado: 2026-11-14T13:08:14.084248+00:00. Equipo: team_17.
Intervalo consultado: [2026-11-14 10:36:49.934000+00:00, 2026-11-14 13:05:36.592000+00:00). Ventanas de 15 minutos por **timestamp de feedback**, UTC; extremos potencialmente parciales.

Fuente: SELECT mediante reco.client.query. SQL exacto en consultas.sql; tablas agregadas en CSV. No se escribe en RawTree.

Peticiones: 26781; clics: 716; compras: 223. CTR: 2.6735%; conversión observada: 0.8327%.
Válidas y a tiempo (served_by=team): 66.2410%. Duplicados de feedback descartados por ID: 0.

## Conversión, CTR y reparto por versión

| version | n | clics | compras | inicio | fin | CTR_pct | conversion_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Leattner.v1 | 4562 | 139 | 51 | 2026-11-14 12:27:14.590000+00:00 | 2026-11-14 13:05:36.285000+00:00 | 3.0469 | 1.1179 |
| Leattner.v2 | 1183 | 58 | 14 | 2026-11-14 12:52:25.592000+00:00 | 2026-11-14 12:59:08.270000+00:00 | 4.9028 | 1.1834 |
| Leattner.v2-cat | 352 | 9 | 4 | 2026-11-14 12:59:29.943000+00:00 | 2026-11-14 13:05:35.595000+00:00 | 2.5568 | 1.1364 |
| Leattner.v2-gulias | 386 | 12 | 3 | 2026-11-14 12:59:28.914000+00:00 | 2026-11-14 13:05:36.591000+00:00 | 3.1088 | 0.7772 |
| baseline | 11257 | 276 | 85 | 2026-11-14 11:24:37.307000+00:00 | 2026-11-14 12:27:12.266000+00:00 | 2.4518 | 0.7551 |
| fallback | 9041 | 222 | 66 | 2026-11-14 10:36:49.934000+00:00 | 2026-11-14 12:59:28.238000+00:00 | 2.4555 | 0.7300 |

Las versiones se atribuyen solo si items coincide exactamente, incluido el orden, con feedback.shown y hay una sola etiqueta distinta coincidente. Fallback se separa. No se multiplica el feedback por envíos duplicados. Comparar periodos distintos no demuestra mejora causal.

## Latencia y fiabilidad por ventana

| ventana | n | p50_ms | p95_ms | a_tiempo_pct |
| --- | --- | --- | --- | --- |
| 2026-11-14 10:30:00+00:00 | 1471 | nan | nan | 0.0000 |
| 2026-11-14 10:45:00+00:00 | 2700 | nan | nan | 0.0000 |
| 2026-11-14 11:00:00+00:00 | 2700 | nan | nan | 0.0000 |
| 2026-11-14 11:15:00+00:00 | 2700 | 1196.0000 | 1934.2000 | 35.8889 |
| 2026-11-14 11:30:00+00:00 | 2700 | 1174.5000 | 1853.2000 | 100.0000 |
| 2026-11-14 11:45:00+00:00 | 2700 | 1199.0000 | 1889.0000 | 99.6667 |
| 2026-11-14 12:00:00+00:00 | 2700 | 1238.0000 | 1992.3000 | 100.0000 |
| 2026-11-14 12:15:00+00:00 | 2700 | 1303.0000 | 2044.7000 | 99.7778 |
| 2026-11-14 12:30:00+00:00 | 2700 | 1216.0000 | 1860.1500 | 99.9259 |
| 2026-11-14 12:45:00+00:00 | 2700 | 1202.0000 | 1872.4500 | 84.3704 |
| 2026-11-14 13:00:00+00:00 | 1010 | 1247.0000 | 1934.8500 | 100.0000 |

Percentiles empíricos pandas (interpolación lineal), únicamente served_by=team. Fiabilidad usa todo el feedback, incluidos missing, late e invalid. No mide peticiones cuyo feedback aún no haya llegado.

## Relación con vistas

| relacion | impresiones | clics | compras | CTR_pct | conversion_pct |
| --- | --- | --- | --- | --- | --- |
| Recién visto | 8316 | 71 | 34 | 0.8538 | 0.4089 |
| Categoría vista | 44956 | 267 | 96 | 0.5939 | 0.2135 |
| Otra categoría | 162078 | 305 | 77 | 0.1882 | 0.0475 |

Grupos excluyentes: recién visto = cualquier producto en recent_views; categoría vista excluye recién vistos. Solo sesiones con vistas y petición encontrada. Sin vistas: 5246 peticiones; sin petición encontrada: 0; impresiones con categoría desconocida excluidas: 0. Catálogo vigente al generar el informe; no reconstruye cambios históricos de categoría. Asociación descriptiva, confundida por posición y política.

## Posición

| slot | impresiones | clics | compras | CTR_pct | conversion_pct |
| --- | --- | --- | --- | --- | --- |
| 1 | 26781 | 206 | 77 | 0.7692 | 0.2875 |
| 2 | 26781 | 127 | 40 | 0.4742 | 0.1494 |
| 3 | 26781 | 83 | 24 | 0.3099 | 0.0896 |
| 4 | 26781 | 71 | 21 | 0.2651 | 0.0784 |
| 5 | 26781 | 50 | 15 | 0.1867 | 0.0560 |
| 6 | 26781 | 33 | 10 | 0.1232 | 0.0373 |
| 7 | 26781 | 44 | 9 | 0.1643 | 0.0336 |
| 8 | 26781 | 33 | 11 | 0.1232 | 0.0411 |
| 9 | 26781 | 36 | 8 | 0.1344 | 0.0299 |
| 10 | 26781 | 33 | 8 | 0.1232 | 0.0299 |

Denominador: una impresión por elemento de feedback.shown, incluido fallback. La asignación de posiciones no es aleatoria: esto no estima el efecto causal de mover un producto.

## Reparto temporal

Ver versiones.csv y versiones.png (conteos por ventana).

## Pendientes

Resultado final y score oficial: PENDIENTE. Las compras realizadas no son la probabilidad esperada con la que puntúa el simulador.
