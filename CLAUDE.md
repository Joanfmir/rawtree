# RawTree · Datathon FME 2026 (equipo DreamTeam)

Reto: recomendar 10 productos por petición en < 3 s. Enunciado completo en `README.md`.
Ronda que puntúa: **domingo 15/11, 10:00–12:00 CET**. La mejor versión debe estar corriendo antes de las 10:00.

## Estructura
- `starter/`: kit original de los organizadores. No se toca (sirve de referencia).
- `reco/`: nuestro sistema.
  - `serve.py`: bucle en vivo (pull → recommend → finalize → push).
  - `policies/`: una clase por estrategia (`fit`, `refresh`, `recommend`); se registran en `policies/__init__.py`.
  - `safety.py`: `finalize()` asegura 10 IDs distintos, activos y en stock. Ninguna respuesta sale sin pasar por aquí.
  - `state.py`, `catalog.py`, `client.py`, `config.py`: estado, SQL del catálogo, acceso a la API, `.env`.
- `scripts/check.py` (conexión y tamaño de las tablas), `scripts/monitor.py` (feedback en vivo).
- `notebooks/`: análisis y evaluación offline. `data/` y `logs/` son locales (gitignored).

## Reglas
- `.env` (la API key) nunca se sube a git ni se comparte.
- Solo **una** instancia de `make serve` en producción para todo el equipo (si hay dos, se duplican las respuestas). Para probar, usar `make dry`.
- `recommend()` debe tardar milisegundos; el trabajo pesado va en `fit()`/`refresh()` o en SQL.
- Antes de cambiar la versión de producción: evaluarla offline y luego probarla con `make dry`.
- Las columnas de RawTree son `Dynamic`: los JOIN y `argMax` necesitan casts (`product_id::String`, `ts::DateTime64(3)`).

