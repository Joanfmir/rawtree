# Guion para el jurado · team_17

Duración prevista: seis minutos, más preguntas. Público: producto. Antes de presentar, ejecutar `python scripts/report.py`, revisar [las cifras actualizadas](figs/resumen.md) y sustituir los PENDIENTE. Las imágenes y CSV conservan denominadores y periodo; `figs/consultas.sql` conserva las consultas ejecutadas. No confundir la foto de práctica con el resultado final.

## 0:00–0:45 · El problema de negocio

«Un cliente llega a una tienda y debemos elegir qué diez productos enseñarle, en menos de tres segundos. Queremos vender más, pero una gran recomendación que llega tarde no sirve: el cliente recibe el escaparate genérico. Por eso diseñamos juntos la relevancia y la fiabilidad.»

Mostrar el reto en una frase: **entender la intención de esta visita y responder a tiempo**. Fuente de diez productos y tres segundos: `README.md`, reglas de entrega. La ronda que puntúa es domingo 15/11, 10:00–12:00 CET (`README.md`, calendario). El resultado oficial de esa ronda: **PENDIENTE**.

## 0:45–1:35 · Partimos de datos, pero desconfiamos de sus sesgos

«Comenzamos con el histórico. Separamos entrenamiento y evaluación por fecha, para no validar con información del futuro. El escaparate anterior enseñaba sobre todo productos populares; acertar sobre ese registro puede premiar imitarlo. Por eso miramos también métricas que corrigen la probabilidad con que se mostró cada producto.»

Resultado offline documentado: con corte 07/11 y 3.530 compras, `ips_hit` pasa de 0,080 con baseline a 0,340 con personal ajustada; `ips_ndcg` de personal es 0,198. Fuente: `CLAUDE.md`, «Resultados offline». Son resultados documentados, no recalculados por este informe; **no equivalen a conversión en vivo**.

«Una advertencia: el histórico sugería mucha recompra. No la tratamos como una verdad del cliente futuro. La contrastamos con el feedback real y mantenemos políticas con decisiones diferentes sobre ese punto.» La señal histórica de 62 % de compras repetidas y 6,5 veces más probabilidad consta en `CLAUDE.md`; no afirmar que se mantiene en vivo. `README.md` advierte que una compra puede reducir el interés posterior.

## 1:35–2:30 · Lo que nos enseñó el stream

Mostrar [relación con vistas](figs/relacion_vistas.png) y [posición](figs/posicion.png).

«Ahora podemos observar qué productos se mostraron de verdad y si hubo clic o compra. Separamos tres casos: el propio producto visto, otro de una categoría vista, y otra categoría. Las cifras actuales están en pantalla con el número de impresiones. Esta evidencia nos llevó a priorizar la intención de la sesión y a probar el filtro de categoría.»

Leer las tasas exactas de `figs/resumen.md`, sección «Relación con vistas»; fuente: SELECT de `feedback`, `requests`, `catalog` y `catalog_updates`. Evitar un multiplicador fijo que pueda quedarse obsoleto. Los tres grupos son excluyentes, y las sesiones sin vistas se excluyen de esa comparación.

«También vemos que las posiciones no tienen el mismo rendimiento. Ordenar es parte de la recomendación. Pero estos gráficos son asociaciones: los mejores candidatos ya se colocan arriba, por lo que no atribuímos toda la diferencia a la posición.» Fuente: `figs/posicion.csv`.

## 2:30–3:30 · Cómo decide el sistema

Mostrar [arquitectura](ARQUITECTURA.md).

«Primero reunimos candidatos: lo que acaba de mirar, productos relacionados, preferencias conocidas y productos populares en su contexto. Después los ordenamos. Comparamos una política de señales ponderadas con una variante centrada en la categoría y con un modelo LightGBM que aprende a ordenar.»

«Para alguien nuevo usamos la sesión y el contexto, sin exigir un historial personal. Para un producto nuevo, el ranker puede usar categoría, precio y señales de sesión; sus estadísticas de popularidad parten de valores de su subcategoría. Así no necesita esperar a acumular compras para ser candidato.» Fuentes de implementación: `reco/policies/personal.py`, `reco/ranker.py`.

«Antes de enviar, una última barrera elimina duplicados, productos inactivos y agotados, y completa la lista con productos válidos. Si el catálogo no tuviera oferta suficiente, esta barrera no puede fabricar productos: seguimos midiendo la fiabilidad en feedback.» Fuente: `reco/safety.py`.

## 3:30–4:35 · Cómo aprendemos y para qué usamos RawTree

Mostrar [reparto de versiones](figs/versiones.png).

«Asignamos las peticiones entre brazos mediante un hash estable del identificador. Así podemos comparar alternativas simultáneamente sobre tráfico comparable. Medimos compras y clics, y revisamos tamaños de muestra antes de decidir. Un periodo anterior contra uno posterior no es un experimento controlado.» Fuente de asignación: `reco/serve.py`. Un mismo usuario puede aparecer en varios brazos: no prometemos independencia perfecta ni significación por el simple hecho de tener una tasa mayor.

«RawTree es el centro de datos operativos: leemos peticiones, catálogo, actividad orgánica y feedback; agregamos señales con SQL; conservamos los derivados en Parquet para servir desde memoria; y el servicio escribe la lista elegida por lotes. Actualizamos señales del stream sin poner el trabajo pesado dentro de cada recomendación.» Fuentes: `reco/features.py`, `reco/client.py`, `reco/serve.py`.

«Actualizar las señales no es reentrenar el modelo. El feedback incorpora compras a la personalización y las exclusiones; los eventos actualizan popularidad y tendencias; el stock y el catálogo se refrescan. El entrenamiento de LightGBM se ejecuta por separado. No hemos implementado una asignación automática de tráfico al brazo ganador.»

## 4:35–5:30 · Resultados y fiabilidad

Mostrar [conversión y CTR](figs/conversion_ctr.png) y [latencia](figs/latencia.png).

«Esta es la evolución observada en ventanas de quince minutos, con el número de peticiones de cada brazo. Mostramos también el fallback. No asignamos una venta a una versión solo porque envió una respuesta: comprobamos que su lista ordenada coincide con lo que el simulador mostró.»

Leer cifras vigentes de `figs/resumen.md`: periodo, peticiones, compras, CTR, conversión, porcentaje válido y a tiempo. Los percentiles de latencia se calculan sobre respuestas `team`; el porcentaje a tiempo incluye todo el feedback, también fallos. Las ventanas de los extremos pueden estar incompletas.

«El feedback contiene compras realizadas. El ranking oficial utiliza probabilidades esperadas del simulador: nuestras tasas son un diagnóstico, no una reproducción exacta de su score.» Fuente: `README.md`, «No luck».

**Resultado final, cold start oficial y comparación con referencia/oráculo: PENDIENTE.** No declarar un ganador global usando tasas de versiones que se ejecutaron en horarios diferentes. Ganador A/B y su incertidumbre durante un periodo simultáneo: **PENDIENTE de análisis específico**.

## 5:30–6:00 · Siguiente paso y demo

«Lo siguiente sería confirmar los brazos en un periodo simultáneo, medir por separado clientes y productos nuevos, y decidir cuánto explorar sin perder ventas. También mediríamos la frescura de las señales y el coste de refrescarlas. El objetivo es adaptar el escaparate manteniendo el plazo de respuesta.»

Demo segura: enseñar los PNG, `figs/resumen.md` y el diagrama. Para actualizar en directo, ejecutar únicamente `python scripts/report.py`. No arrancar otro servicio. Cierre: «La principal lección es que la intención actual del cliente y la operación fiable deben validarse con el stream, no darse por resueltas por el histórico».

## Preguntas probables

**¿Habéis ganado al baseline?** El offline documentado mejora `ips_hit`; la comparación en vivo se lee en los resultados actualizados y debe limitarse a periodos comparables. La mejora oficial final sigue PENDIENTE.

**¿Por qué no recomendar siempre lo más vendido?** Es un respaldo robusto, pero no representa necesariamente la intención de esta sesión. La relación entre vistas y compras se muestra con denominadores en el informe.

**¿Cómo evitáis el sesgo del histórico?** Corte temporal e indicadores ponderados por propensión. Eso no elimina todo el cambio de distribución; por eso usamos feedback y experimentos simultáneos. No afirmamos haber calculado una estimación causal sin restricciones.

**¿Qué aprende automáticamente?** Se refrescan features y estado; se incorporan compras. No hay reentrenamiento continuo de LightGBM ni bandido adaptativo implementado. Los cambios de política los decide el equipo.

**¿Cómo tratáis el cold start?** Sesión, contexto, catálogo y valores por subcategoría para productos nuevos. La métrica oficial incluye usuarios nuevos o vistas de productos añadidos durante el evento (`README.md`). El desglose cuantitativo oficial sigue PENDIENTE.

**¿Y los precios?** Se aplican actualizaciones del catálogo; el ranker usa precio y su relación con el perfil. No afirmamos disponer de un experimento causal sobre descuentos ni de un optimizador de promociones.

**¿Por qué SQL y ficheros locales?** RawTree agrega los datos cerca de su origen; Parquet e índices en memoria permiten separar analítica de respuesta. El informe conserva SQL exacto para auditar las cifras.

**¿Por qué mostrar fallback?** También es una experiencia real del cliente. Excluirlo ocultaría fallos. En cambio, para comparar políticas distinguimos quién se sirvió realmente.

**¿Una tasa mayor demuestra que el brazo gana?** No: hacen falta periodo simultáneo, muestra suficiente e incertidumbre; además puede haber dependencia entre peticiones del mismo usuario. Las figuras son descriptivas.

**¿Cómo sabemos que no se duplican resultados?** La atribución compara `items` con `feedback.shown` incluido el orden, y exige una etiqueta inequívoca. Una respuesta ambigua queda separada. Los duplicados de feedback se contabilizan y descartan por ID.

**¿Cuál es la limitación principal?** El histórico y el stream difieren; las compras son escasas y aleatorias, y la última ventana puede estar incompleta. Los resultados finales no pueden adelantarse a la ronda final.

## Foto de práctica verificada (sustituir tras actualizar)

Fuente de todas las cifras de este bloque: `figs/resumen.md`, intervalo UTC del 14/11 **10:36:49.934–13:05:36.592**, generado con consultas ejecutadas por `scripts/report.py`.

- 26.781 peticiones, 716 clics y 223 compras: CTR **2,6735 %**, conversión **0,8327 %**.
- Compra por impresión: recién visto **0,4089 %** (8.316 impresiones); categoría vista **0,2135 %** (44.956); otra categoría **0,0475 %** (162.078).
- `Leattner.v1`: 51 compras / 4.562 peticiones, **1,1179 %**. Baseline: 85 / 11.257, **0,7551 %**. Son horarios distintos: no presentar esta diferencia como uplift causal.
- Los brazos nuevos tienen solo 4 y 3 compras, respectivamente, en esta foto: insuficiente para declarar ganador.
- Fiabilidad acumulada: **66,2410 %**, incluyendo un tramo inicial servido enteramente por fallback. No describe por sí sola el estado operativo actual. En la ventana parcial 13:00–13:05:36 UTC: **100 %** a tiempo sobre 1.010 peticiones; latencia team p50 **1.247 ms**, p95 **1.934,85 ms**.

Para regenerar todo desde la raíz, con el Python autorizado:

```bash
/Users/joanfernandezmir/Desktop/DATA/rawtree/.venv/bin/python scripts/report.py
```

Para reproducir el intervalo de esta foto (el feedback disponible puede crecer por llegadas tardías):

```bash
/Users/joanfernandezmir/Desktop/DATA/rawtree/.venv/bin/python scripts/report.py --since '2026-11-14T10:36:49.934Z' --until '2026-11-14T13:05:36.592Z'
```

Para aislar la ronda final, cuando haya feedback disponible:

```bash
/Users/joanfernandezmir/Desktop/DATA/rawtree/.venv/bin/python scripts/report.py --since '2026-11-15T09:00:00Z' --until '2026-11-15T11:00:00Z'
```

El filtro usa hora de **feedback**, no hora de petición: este último comando es un diagnóstico aproximado de la ronda; para una auditoría exacta del score hay que filtrar por `shop.requests.ts` y esperar todo el feedback. Los comandos sobrescriben los artefactos anteriores.
