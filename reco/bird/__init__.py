"""Bird: ranker de compra con CatBoost (P(compra) por candidato → top 10).

    schema.py    qué columnas son features, cuáles categóricas, cuál es el target (una entrada por versión del dataset)
    dataset.py   carga, validación del contrato de datos, preparación de X y datos sintéticos
    model.py     entrenamiento, guardado y carga (modelo + esquema + métricas van siempre juntos)
    metrics.py   clasificación (logloss, PR-AUC, calibración) y ranking dentro de los 10 enseñados
    builder.py   build_features(request, candidatos, estado): provisional hasta que llegue el del equipo de datos

Añadir features = añadir un esquema nuevo en schema.py (y que el builder las entregue). Nada más cambia.
"""
