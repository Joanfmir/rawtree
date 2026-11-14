# Revisión de fiabilidad · team_17

Validación del 14/11/2026. Solo SELECT; no se ha arrancado ni cambiado producción, ni hecho commit.
Las líneas se refieren al árbol modificado; los escenarios describen el comportamiento anterior.

## Hallazgos corregidos

| Lugar | Gravedad | Escenario anterior y corrección |
|---|---|---|
| reco/catalog.py:43,55 | Alta | Timeout de stock devolvía conjunto vacío; timeout de catálogo sustituía el catálogo actualizado por el inicial (perdía new_product y reactivaba productos). Propagar errores salvo esquema aún inexistente antes del stream; State conserva el estado anterior. |
| reco/state.py:26 | Alta | Una consulta fallida dejaba catálogo nuevo con stock viejo. Publicación conjunta al completar todas las consultas, sin mantener el bloqueo durante SQL. |
| reco/state.py:21 y reco/safety.py:12 | Media | Refresh podía cambiar stock entre dos productos de una misma respuesta. finalize trabaja sobre snapshot estable; refresh sustituye las colecciones. |
| reco/state.py:41 | Alta | Catálogo vacío aceptaba IDs inventados; IDs dict/list causaban TypeError; actividad nula era ambigua. Rechazar IDs no textuales, desconocidos, sin actividad confirmada y agotados. |
| reco/state.py:49 y reco/safety.py:12 | Alta | Best sellers insuficientes tras filtrar stock generaban menos de diez, aunque hubiera otros productos. Rellenar desde todo el catálogo y eliminar duplicados. Si hay menos de diez válidos en total, lanzar ValueError explícito: imposible fabricar una respuesta válida. |
| reco/policies/personal.py:155 | Alta | Vistas nulas, vacías o sin product_id causaban KeyError/TypeError. Normalización compartida con ranker; desconocidos se eliminan al finalizar. |
| reco/policies/personal.py:124,144,180,208 y reco/ranker.py:43,105, última línea | Media | Fechas con zona, compras con fecha nula, precios completamente nulos, usuario con n_events=None o catálogo enteramente nuevo producían excepción o NaN. Fechas UTC, conservación de zona y milisegundos, descarte de puntuaciones no finitas y matriz numérica finita. |
| reco/policies/ranker.py:47 | Alta | refresh consultaba compras por red en el bucle de serve, con timeout de 30 s. Ejecutarlo en un único hilo de fondo; test bloquea la consulta y comprueba que refresh ya retornó. |
| reco/policies/ranker.py:65,89 | Media | Actualización mutaba fz.bought mientras recommend/rebuild lo leían; conjunto acumulado crecía sin límite temporal. Publicar frozenset separado; ventana SQL de dos días y LIMIT 500000; no modificar Featurizer publicado. |
| reco/features.py:145 | Media | Rebuild concurrente podía leer un Parquet incompleto. Temporal .tmp, replace atómico y limpieza incluso con disco lleno. Test comprueba que durante escritura solo se ve la versión antigua. |
| reco/state.py:27 y reco/policies/ranker.py:44,61 | Media | Cambios del reloj del sistema retrasaban intervalos de refresco. Usar monotonic para intervalos; UTC para timestamps del negocio. |

## Pruebas y datos reales

```sh
/Users/joanfernandezmir/Desktop/DATA/rawtree/.venv/bin/python -m unittest discover -s tests
/Users/joanfernandezmir/Desktop/DATA/rawtree/.venv/bin/python tests/validate_live.py
git diff --check
```

18 tests pasan (0,468 s). Las primeras 12 regresiones, antes del parche, dieron 7 fallos y 2 errores; esas mismas 12 pasaron después. La prueba adicional de n_events=None también falló antes de su corrección. Se prueban ambas versiones Personal, categoría v2, usuario nuevo/nulo, vistas desconocidas/nulas, new_product, inactivos y stock, fallback, errores de red simulados, snapshot, reloj, refresco asíncrono y Parquet atómico. El test de A/B extrae por AST la expresión real de serve.py y comprueba 9000 IDs con crc32 y tres brazos; no arranca serve.

Estado real: 5100 productos, 36 sin stock. Una consulta de 5000 solicitudes recientes; cada política usa las features locales y se valida contra el mismo State().refresh(). No se invoca policy.refresh ni se escriben tablas.

| Política | Respuestas | Inválidas | Excepciones | p50 ms | p99 ms | Máximo ms |
|---|---:|---:|---:|---:|---:|---:|
| personal_v1 | 5000 | 0 | 0 | 0,223 | 1,884 | 34,884 |
| personal | 5000 | 0 | 0 | 0,244 | 0,957 | 10,225 |
| gulias | 5000 | 0 | 0 | 3,861 | 8,606 | 15,712 |

Tiempo medido: recommend + finalize, excluye red, carga inicial, consultas y envío. No demuestra latencia extremo a extremo <3 s. No hubo request_id inválidos que reportar; listas de inválidos y excepciones en live_results.json. La verificación es relativa al snapshot cargado: el stock puede cambiar después.

## Riesgos y pendientes

- No se ha ejecutado una prueba de 24 h ni medido RSS bajo refrescos completos. Se ha limitado el conjunto de compras en vivo y evitado su copia acumulativa en Featurizer, pero los índices de usuarios crecen con los datos reales. La ventana de dos días olvida compras antiguas del carrusel que no estén en las features históricas.
- Personal._index y Featurizer siguen construyéndose en el hilo de refresco tras el subprocess de descarga; pueden disputar GIL. No se ha medido este efecto bajo carga ni se ha aislado toda la construcción del índice en otro proceso. Pendiente antes de afirmar garantía de latencia durante refresh.
- Reemplazo de Parquet es atómico por tabla, no por generación completa: dos rebuilds pueden mezclar tablas de generaciones distintas. Hace falta publicar un directorio de generación con manifiesto y un único constructor. No se ha afirmado consistencia transaccional entre tablas.
- Los índices de features pueden tardar 10–20 min en incorporar new_product. State sí valida estos productos y fallback puede usarlos; gulias no los genera hasta actualizar sus features. Stock y catálogo se consultan por separado en RawTree; snapshot local coherente no equivale a transacción remota.
- Si catálogo_updates no tiene esquema se conserva compatibilidad de arranque mediante el error específico Unknown expression identifier. Otro fallo de esquema que comparta ese texto puede activar también ese fallback.
- Sin diez productos válidos no existe solución local. ValueError evita devolver silenciosamente una lista inválida; serve debe contenerlo por petición (parche propuesto abajo).
- Falta validar en vivo la nueva consulta de compras (ventana/ORDER BY), sin lanzar refrescos ni producción. Se cubren su publicación y asincronía con datos sintéticos.

## Parche textual propuesto para reco/serve.py (NO aplicado)

1. En el try por petición, incluir finalize; si la política falla, intentar fallback en un segundo try. Si tampoco es posible, omitir solo esa petición para que no se pierda el lote completo:

```python
try:
    items = finalize(policy.recommend(req, state), state)
except Exception as e:
    errors += 1
    print(f"[recommend] {req['request_id']}: {e}")
    try:
        items = finalize([], state)
    except ValueError as e:
        print(f"[sin catálogo válido] {req['request_id']}: {e}")
        continue
answers.append({"request_id": req["request_id"], "items": items,
                "model_version": version})
```

2. La condición `now - _epoch(...) < 4` acepta solicitudes ya fuera del plazo de 3 s. Tras cada consulta, calcular edad con hora actual, rechazar fechas inválidas y usar un margen medido para el envío (por ejemplo 2,3 s, parametrizable); repetir la comprobación antes de enviar. El `_epoch` actual trunca offsets; usar parser que conserve zona como Personal._epoch, pero devolver error si falta ts para no tratar una fecha inválida como reciente. La calibración del reloj frente al servidor y la latencia de inserción siguen siendo necesarias.

3. `seen.add` antes de insert pierde peticiones si el envío falla. Mover el marcado al resultado satisfactorio; para fallos ambiguos (timeout tras aceptación) reconciliar por SELECT de submissions antes de reintentar, pues el reto prohíbe duplicados. Sustituir el set con poda arbitraria por mapa request_id→timestamp y caducidad de solicitudes fuera de plazo. Esta reconciliación requiere diseño del responsable de serve/client y no se ha implementado aquí.

4. `policy.fit` está fuera de protección: un modelo ausente/corrupto impide arrancar incluso baseline. Capturar por brazo, registrar el error y sustituir ese brazo por Baseline().fit(state), conservando una etiqueta que indique fallback. No continuar con una política parcialmente inicializada.

No se proponen cambios a Makefile. No se ha editado serve.py ni client.py.
