# RawTree · Datathon FME 2026 (equipo DreamTeam)

Reto: recomendar 10 productos por petición en < 3 s. Enunciado completo en `README.md`.
Ronda que puntúa: **domingo 15/11, 10:00–12:00 CET**. La mejor versión debe estar corriendo antes de las 10:00.

## Estructura
- `starter/`: kit original de los organizadores. No se toca (sirve de referencia).
- `reco/`: nuestro sistema.
  - `serve.py`: bucle en vivo (pull → recommend → finalize → push).
  - `policies/`: una clase por estrategia (`fit`, `refresh`, `recommend`); se registran en `policies/__init__.py`.
  - `safety.py`: `finalize()` asegura 10 IDs distintos, activos y en stock. Ninguna respuesta sale sin pasar por aquí.
  - `features.py`: tablas para entrenar y servir (item_features, user_subcat, user_brand, user_purchases, item_neighbors, context_popular). Se calculan con SQL en RawTree y se guardan en `data/features_live/` o `data/features_until_<fecha>/` (Parquet).
  - `evaluate.py`: evaluación offline (`python -m reco.evaluate`). Métrica principal `ips_ndcg`/`ips_hit` **sin recompras** (corrigen el sesgo del log); `hit@10` favorece a los best sellers.
  - `offline_live.py` + `scripts/offline_vs_live.py` (`make offline C="pol"`): contrasta la evaluación offline con el leaderboard (fiabilidad) y sitúa políticas nuevas. Ver "Evaluar un modelo nuevo".
  - `policies/personal.py`: política principal. Pesos en `DEFAULT_WEIGHTS`, ajustados con `scripts/tune.py search`.
  - `state.py`, `catalog.py`, `client.py`, `config.py`: estado, SQL del catálogo, acceso a la API, `.env`.
- `scripts/tune.py` (`ablation` | `search`), `scripts/build_features.py` (`--until YYYY-MM-DD` para la versión offline, ~15 s), `scripts/check.py` (conexión y tamaño de las tablas), `scripts/monitor.py` (feedback en vivo).
- `notebooks/`: análisis y evaluación offline. `data/` y `logs/` son locales (gitignored).

## Reglas
- `.env` (la API key) nunca se sube a git ni se comparte.
- Solo **una** instancia de `make serve` en producción para todo el equipo (si hay dos, se duplican las respuestas). Para probar, usar `make dry`.
- `recommend()` debe tardar milisegundos; el trabajo pesado va en `fit()`/`refresh()` o en SQL.
- Antes de cambiar la versión de producción: `make offline C="<política>"`, luego `make dry`, y si gana offline, A/B **simultáneo** en vivo.
- Las columnas de RawTree son `Dynamic`: los JOIN y `argMax` necesitan casts (`product_id::String`, `ts::DateTime64(3)`).

- Permisos de la API key: solo `SELECT` e inserción de JSON por la API. No se pueden crear tablas, ni hacer `INSERT ... SELECT`, ni borrar. Lo derivado se guarda en local (`data/`).
- Evaluación offline: features `--until 2026-11-07` y test con `shop.history_sessions` desde esa fecha (el histórico acaba el 14/11 a las 07:08 UTC).

## Versiones de Bird (nombres estandarizados)
Las versiones de Bird se llaman **`vN-bird`**, con `N` = número de versión (v1-bird, v2-bird, v3-bird…). Ese es el
nombre que se usa en el leaderboard (`model_version`), en el A/B y al hablar del modelo. Cada versión nueva suma 1
y se añade a esta tabla y a `BIRD_VERSIONS` en `reco/policies/__init__.py`.

| Versión | Política | Qué es |
|---|---|---|
| v1-bird | `bird_v1` | CatBoost con las 5 features del contrato inicial (sin personalización) |
| v2-bird | `bird_v2` | CatBoost con 56 features (dataset v2), sin recompra; puntúa todo el catálogo |
| v3-bird | `bird_blend_rank` | `personal_v5` propone 50 candidatos y el modelo de ranking (YetiRank) los ordena |

El nombre de versión funciona como política: `make dry POLICY=v3-bird`, `make up AB=<otros brazos>,v3-bird=v3-bird`.
Código en `reco/bird/` y `reco/policies/bird.py`; entrenamiento y evaluación con `scripts/bird.py` y
`scripts/bird_blend.py`. La siguiente versión será **v4-bird**.

## Fiabilidad de la evaluación offline (14/11, 8 versiones servidas)
Verdad: conversión **esperada** de cada `model_version` en el leaderboard (`/api/state?by=model`, sin suerte).
Offline: `history_sessions` desde el 07/11, **sin recompras** (1.096 compras).

| Métrica offline | Spearman con el leaderboard | Parejas bien | Parejas claras bien |
|---|---|---|---|
| `ips_ndcg` sin recompras | **0,95** | **26/28** | **13/13** |
| `hit@10` sin recompras | 0,74 | 23/28 | 10/13 |
| `ips_ndcg` CON recompras | −0,14 | 12/28 | 6/14 |
| `hit@10` CON recompras | −0,60 | 8/28 | 3/14 |
| `nuevo_ips_ndcg` vs conv. de nuevos | 0,78 | 22/28 | 5/5 |

- **Sirve para descartar**: si una política sale claramente peor offline, no hace falta llevarla al vivo.
- **No sirve para decidir entre dos buenas**: falla justo arriba: v3-bird 0,260 > v2-bird-norebuy 0,207 offline, pero
  1,49 < 1,79 en el leaderboard (diferencia en el límite de "clara"). La mezcla v3-bird es la más sobrevalorada
  (residuo −0,24 en la calibración) y bird_v2 la más infravalorada (+0,25).
- "Parejas claras": el leaderboard las separa por más de 2 errores estándar (binomial, conservador).
- Las cifras con recompras son de `reco.evaluate --with-rebuy`; el resto, de `make offline`.
- **Nunca** mirar métricas con recompras (`--with-rebuy`): el simulador penaliza recomprar y el orden sale al revés.
- La fiabilidad se recalcula sola con `make offline`; cada versión nueva en vivo se añade a `LIVE_VERSIONS`.

## Evaluar un modelo nuevo
1. Traer el código de la branch del modelo (`git merge origin/<branch>`) y sus modelos de `data/` (están en
   `.gitignore`: hay que subirlos a propósito o copiarlos). El modelo para offline debe estar entrenado **solo con
   datos anteriores al 07/11**; si no, hay fuga y saldrá inflado.
2. Registrar la política en `reco/policies/__init__.py`.
3. `make offline C="<política>"`: tabla en vivo vs offline, fiabilidad actual, y para el candidato Δ `ips_ndcg`
   (IC 95 % bootstrap) frente a la mejor versión en vivo, en conocidos y en nuevos, con veredicto:
   PEOR → descartar · EMPATE → solo el vivo decide · MEJOR → A/B simultáneo antes de pasar a producción.
4. Si va al vivo, añadir su `model_version` → política a `LIVE_VERSIONS` en `reco/offline_live.py`.

## Recompra
En el histórico, el 62 % de las compras repiten un producto ya comprado y uno ya comprado se compra 6,5 veces más.
**En vivo es al revés**: el simulador penaliza recomprar (lo ya comprado convierte ~9 veces menos). Por eso el test
offline excluye las recompras y las políticas buenas no recomiendan lo ya comprado.
