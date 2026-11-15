# Guion de la demo · DreamTeam (team_17)

Seis minutos más preguntas. Los sponsors han pedido **el cómo y el porqué, no el qué**: por qué hicimos cada
cosa, cómo entrenamos, qué agentes de código usamos y qué hizo cada persona. Por eso el guion va por decisiones
y no por componentes. Cada bloque sigue el mismo patrón: **qué vimos → qué decidimos → qué pasó**.

Antes de presentar hay que rellenar los PENDIENTE: resultado final, qué era la versión `mj` y quién usó cada
agente.

| Criterio del jurado | Puntos | Dónde se responde |
|---|---|---|
| Método (justificado y validado offline) | 10 | Bloques 2 y 3 |
| Aprender del stream | 10 | Bloques 4 y 5 |
| Uso de RawTree | 5 | Bloques 4 y 7 (demo) |
| Comunicación | 5 | Una idea por bloque, sin jerga |
| Petición de los sponsors: agentes y reparto | — | Bloque 6 |

---

## 1 · El problema y el resultado (0:00–0:30)

«Cada tercio de segundo llega un cliente y tenemos tres segundos para elegir los diez productos que verá.
Empezamos el sábado vendiendo 0,74 de cada 100 visitas con los más vendidos. Acabamos en torno a 2,2. Os
contamos las cinco decisiones que explican ese salto, incluida una idea que descartamos.»

- Resultado de la ronda final: **PENDIENTE (sale a las 12:00)**.
- Práctica, últimas tres horas: 5.º–6.º de 42 equipos, 2,13 compras esperadas por 100 peticiones.

Enseñar la curva de versiones (tabla del bloque 5).

## 2 · Decisión 1: no recomendar lo que el cliente ya compró (0:30–1:20)

**Qué vimos.** En el histórico, el 62 % de las compras repiten un producto ya comprado, y uno ya comprado se
compra 6,5 veces más. Nuestra primera política personalizada lo explotaba y ganaba todas las métricas offline.

**Qué pasó en vivo.** Lo contrario: en nuestro feedback, un producto ya comprado convertía unas 9 veces menos.

**Qué decidimos.** Ninguna versión desde `v2-bird` propone algo ya comprado, y los tests offline excluyen las
recompras.

Frase para el jurado: «El histórico describe a los clientes del sistema viejo; el simulador es otro cliente.
Aprendimos pronto a no fiarnos del histórico sin contrastarlo».

## 3 · Decisión 2: validar la validación (1:20–2:20)

**Qué vimos.** Después de lo anterior no sabíamos qué número offline creer.

**Qué decidimos.** Cogimos las 8 versiones que ya habíamos servido, con su conversión real en el leaderboard,
y pedimos a cada test offline que las ordenara.

| Test offline · métrica | Correlación con el leaderboard | Parejas bien ordenadas |
|---|---|---|
| **Orgánico · `ndcg@10`** | **0,98** | **27 de 28** |
| Histórico · `ips_ndcg` sin recompras | 0,95 | 26 de 28 |
| Histórico · `ips_ndcg` con recompras | −0,14 | 12 de 28 |
| Histórico · `hit@10` con recompras | −0,60 | 8 de 28 |

**Qué pasó.** La métrica con la que empezamos ordenaba nuestras versiones al revés. El test de referencia pasó
a ser el **orgánico**: peticiones reales del stream, y el acierto es lo que ese usuario compró por su cuenta en
los 30 minutos siguientes. `make offline C="<política>"` lo ejecuta y da un veredicto: peor → descartar,
empate → solo el vivo decide, mejor → A/B.

Es la respuesta directa a «¿cómo lo validasteis offline antes de salir en vivo?».

## 4 · Decisiones 3 y 4: cómo entrenamos (2:20–3:40)

**Qué vimos antes de tocar el modelo** (`docs/DATASET_V5.md`, sección 1):

- El vivo tiene unas 29 veces más eventos por hora que el histórico: un recuento no significa lo mismo al
  entrenar que al servir.
- En el histórico no hay ni un cambio de precio; en vivo hay promociones de dos horas, con un 42 % más de
  conversión en nuestro carrusel.
- Entran 100 productos nuevos cada cuatro horas y venden desde el primer minuto.

**Qué decidimos.**

- **Features que valen igual en los dos mundos**: 81, ninguna es un recuento absoluto (cuotas, cocientes y lifts
  suavizados). Un test lo comprueba.
- **Una sola implementación de cada fórmula** para entrenar y para servir, con una comprobación de paridad valor
  a valor.
- **Nuestro feedback como dato de entrenamiento**, con peso triple: es la única medida directa del simulador.
  Dos features salen de ahí: clics y compras de cada producto en nuestro carrusel, corregidos por la posición.
- **Cold start desde el entrenamiento**: en el 30 % de las peticiones del histórico se esconde la historia del
  usuario, y los productos nuevos heredan las tasas de su subcategoría.

**El modelo.** CatBoost que estima la probabilidad de compra de cada producto para esta visita; puntúa todo el
catálogo en stock y devuelve los diez mejores. Lo entrenamos tres veces según llegaba feedback: sábado por la
noche, domingo de madrugada y domingo a las 10:00 con todo el vivo acumulado.

## 5 · Decisión 5: solo sube a producción lo que gana un A/B simultáneo (3:40–4:40)

**Por qué.** Las tendencias y el stock cambian durante el día: comparar una versión de las 15:00 con otra de las
18:00 no dice nada. Repartimos las peticiones por un hash del identificador y cada brazo firma con su versión.

**La idea que descartamos.** `v8-bird` añadía compras orgánicas al entrenamiento. Offline empataba. Tres horas
con cuatro brazos a partes iguales:

| Versión | Peticiones | Compras por 100 peticiones |
|---|---|---|
| v7-bird | 7.764 | 2,29 ± 0,17 |
| v5-bird | 7.883 | 2,21 ± 0,17 |
| v6-bird | 7.724 | 2,20 ± 0,17 |
| v8-bird | 7.866 | 1,92 ± 0,15 |

«v5, v6 y v7 son indistinguibles; v8 fue la última las tres horas y no entró en la final. También miramos qué
hacía cada una con los productos nuevos: v7 enseñaba los de una tanda recién llegada unas tres veces antes que
v5 y v6, y v8 casi los ignoraba.»

**Evolución en vivo** (compras reales por 100 peticiones mientras cada versión estuvo activa):

| Versión | Sábado/domingo (CET) | Conversión |
|---|---|---|
| baseline | sáb 12:24 | 0,74 |
| Leattner.v1 → v3 (señales ponderadas) | sáb 13:27–17:19 | 1,04 → 1,45 |
| v2-bird sin recompra (CatBoost, 56 features) | sáb 18:00–22:31 | 1,62 |
| v3-bird (heurística + ranking) | sáb 19:30–23:09 | 1,53 |
| v5/v6/v7-bird (81 features + feedback) | dom 01:20–10:23 | 1,94–2,30 |
| Ronda final (`mj`) | dom 10:24– | **PENDIENTE** |

Avisar de que son horas distintas y no comparables una a una; las decisiones se tomaron con los A/B.

## 6 · Cómo trabajamos: agentes y equipo (4:40–5:30)

**Agentes de código.**

- Claude Code fue la herramienta principal (46 commits llevan su co-autoría) y usamos agentes de Codex el sábado
  para tareas en paralelo (evaluador en vivo, análisis de cold start y tendencias, informe).
  **PENDIENTE: confirmar quién usó qué.**
- **`CLAUDE.md` como memoria compartida del equipo.** Cada sesión de agente empieza leyéndolo: estructura,
  reglas, tabla de versiones y cada hallazgo con su número. Cuando alguien aprendía algo (la recompra penaliza,
  qué test offline creer), lo escribía ahí y el agente del siguiente partía de eso.
- **Reglas para trabajar cerca de producción:** una sola instancia para todo el equipo y `make dry` para probar;
  análisis solo con `SELECT`; cada modelo en su rama hasta ganar un A/B.
- **Qué hicieron bien los agentes:** explorar los datos con SQL, mantener iguales las features de entrenar y
  servir, la revisión de fiabilidad y los análisis durante la ronda.
- **Qué nos quedamos nosotros:** qué hipótesis probar, leer los A/B con su margen de error y cada cambio de
  producción.

**Quién hizo qué.** Dos subequipos; el reparto es aproximado porque nos fuimos ayudando.

| Subequipo | Personas | En qué se centró |
|---|---|---|
| Algoritmos | Lucas y Joan | Políticas y modelos de recomendación, servicio y A/B |
| Dataset | Santi y Jordi | Exploración de datos, features y datasets de entrenamiento |

## 7 · Demo (5:30–6:00)

Una sola cosa en pantalla: **el A/B es una consulta**. Ejecutar `make monitor`, que une
`team_17.submissions` (qué versión respondió) con `team_17.feedback` (qué pasó) y enseña conversión, respuestas
a tiempo y latencia por versión. Con eso se ve RawTree, el feedback y cómo decidimos, todo a la vez.

Plan B si no hay red: la tabla del bloque 5.

No arrancar ningún servicio durante la demo.

Cierre: «El modelo es la parte menos original. Lo que nos hizo subir fue comprobar cada creencia contra el
stream: la recompra, la métrica offline y cada versión nueva».

---

## Preguntas probables

**¿Por qué CatBoost y no algo más complejo?** Trata bien las categóricas (marca, categoría, atributos) sin
fugas y nos permitió reentrenar con feedback nuevo durante el evento. Probamos dos variantes en dos etapas: v3
perdió en vivo frente a v2 y v4 no llegó a servirse.

**¿Aprende solo?** Las señales sí: popularidad y tendencias en tramos de 10 minutos, historia de cada usuario
al minuto, catálogo y stock. El modelo se reentrena a mano y los cambios de versión los decide el equipo; no
hay un bandido automático.

**¿Cómo tratáis a un cliente nuevo?** El 97 % de los marcados como nuevos ya tienen actividad de hoy antes de
la petición, y la usamos. Para los que no tienen nada quedan el dispositivo, la región y la página. En práctica,
con usuarios nuevos las cuatro versiones convertían igual (2,9–3,0 %).

**¿Y un producto nuevo?** Hereda las tasas de su subcategoría y tiene una feature de tendencia de la última
hora. En nuestro carrusel los productos nuevos convertían 2,6 veces más que el resto.

**¿Y los precios?** Usamos el precio en el instante de la petición y el descuento activo. No optimizamos
ingresos, solo conversión.

**¿Cuánto tarda?** Mediana de 1,6 s y percentil 95 de 2,4 s sobre los 3 s permitidos, con el sondeo incluido.
En práctica respondimos a tiempo el 97–98 % de las peticiones; con una sola versión en la final, el 99,7 %.

**¿Por qué guardáis cosas en local?** Nuestra clave solo permite `SELECT` e insertar JSON, así que lo derivado
no puede vivir en RawTree: lo agregamos con SQL y lo guardamos en Parquet para servir desde memoria.

**¿Qué haríais con más tiempo?** Reentreno automático con comprobación previa, medir cold start con un test
propio (los nuestros no eran fiables para ese matiz) y bajar la latencia para ganar fiabilidad.

**¿Qué salió mal?** Empezamos optimizando una métrica que ordenaba al revés; el sábado por la mañana perdimos
un tramo entero en fallback; y mantuvimos cuatro brazos hasta entrada la ronda final.

## Fuentes de las cifras

- Recompra, fiabilidad de los tests offline y nota de v8: `CLAUDE.md`.
- Mediciones previas al dataset v5, filas y pesos de entrenamiento: `docs/DATASET_V5.md`.
- A/B de cuatro brazos: leaderboard público (`/api/state?by=model`), 15/11 de 07:00 a 09:56 CET.
- Evolución por versión, latencia y respuestas a tiempo: `team_17.feedback` unido a `team_17.submissions`,
  consultado el 15/11 a las 10:40 CET. Es compra real, no la esperada que usa el leaderboard.
- Productos nuevos por versión: análisis del 15/11 sobre las peticiones de 05:00 a 10:00 CET.
