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
  - `evaluate.py`: evaluación offline (`python -m reco.evaluate`). Métrica principal `ips_ndcg`/`ips_hit` (corrigen el sesgo del log); `hit@10` favorece a los best sellers.
  - `policies/personal.py`: política principal. Pesos en `DEFAULT_WEIGHTS`, ajustados con `scripts/tune.py search`.
  - `state.py`, `catalog.py`, `client.py`, `config.py`: estado, SQL del catálogo, acceso a la API, `.env`.
- `scripts/tune.py` (`ablation` | `search`), `scripts/build_features.py` (`--until YYYY-MM-DD` para la versión offline, ~15 s), `scripts/check.py` (conexión y tamaño de las tablas), `scripts/monitor.py` (feedback en vivo).
- `notebooks/`: análisis y evaluación offline. `data/` y `logs/` son locales (gitignored).

## Reglas
- `.env` (la API key) nunca se sube a git ni se comparte.
- Solo **una** instancia de `make serve` en producción para todo el equipo (si hay dos, se duplican las respuestas). Para probar, usar `make dry`.
- `recommend()` debe tardar milisegundos; el trabajo pesado va en `fit()`/`refresh()` o en SQL.
- Antes de cambiar la versión de producción: evaluarla offline y luego probarla con `make dry`.
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

## Resultados offline (corte 2026-11-07, 3.530 compras)
| Política | ips_hit | ips_ndcg | hit@10 |
|---|---|---|---|
| baseline | 0,080 | — | 0,520 |
| coview | 0,108 | — | 0,209 |
| personal (ajustada) | **0,340** | **0,198** | **0,754** |

Hallazgo clave: **recompra**. El 62 % de las compras repiten un producto ya comprado por el usuario, y un producto ya comprado se compra 6,5 veces más.
