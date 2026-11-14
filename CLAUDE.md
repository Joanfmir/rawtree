# RawTree · Datathon FME 2026 (equipo DreamTeam)

Reto: recomendar 10 productos por petición en < 3 s. Enunciado completo en `README.md`.
Ronda que puntúa: **domingo 15/11, 10:00–12:00 CET**. La mejor versión debe estar corriendo antes de las 10:00.

## Estructura
- `starter/`: kit original de los organizadores. No se toca (sirve de referencia).
- `reco/`: nuestro sistema.
  - `serve.py`: bucle en vivo (pull → recommend → finalize → push).
  - `policies/`: una clase por estrategia (`fit`, `refresh`, `recommend`); se registran en `policies/__init__.py`.
  - `safety.py`: `finalize()` asegura 10 IDs distintos, activos y en stock. Ninguna respuesta sale sin pasar por aquí.
  - `storage.py`: guarda cada decisión (petición + respuesta + `lag_ms`) en `team_17.scratch` con `kind='decision'` y en `logs/decisions.jsonl`, en segundo plano. En dry-run solo en local.
  - `state.py`, `catalog.py`, `client.py`, `config.py`: estado, SQL del catálogo, acceso a la API, `.env`.
- `scripts/check.py` (conexión y tamaño de las tablas), `scripts/monitor.py` (feedback en vivo).
- `notebooks/`: análisis y evaluación offline. `data/` y `logs/` son locales (gitignored).

## Reglas
- `.env` (la API key) nunca se sube a git ni se comparte.
- Solo **una** instancia de `make serve` en producción para todo el equipo (si hay dos, se duplican las respuestas). Para probar, usar `make dry`.
- `recommend()` debe tardar milisegundos; el trabajo pesado va en `fit()`/`refresh()` o en SQL.
- Antes de cambiar la versión de producción: evaluarla offline y luego probarla con `make dry`.
- Las columnas de RawTree son `Dynamic`: los JOIN y `argMax` necesitan casts (`product_id::String`, `ts::DateTime64(3)`).

## Logs de decisiones
```sql
-- cruzar lo que respondimos con lo que pasó
SELECT d.model_version, count() AS n, avg(d.lag_ms::Float64) AS lag_ms,
       100 * countIf(f.purchased::String != '') / count() AS conv_pct
FROM team_17.scratch AS d INNER JOIN team_17.feedback AS f ON f.request_id::String = d.request_id::String
WHERE d.kind = 'decision' GROUP BY d.model_version
```
