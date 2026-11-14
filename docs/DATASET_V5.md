# Dataset v5 de Bird (v5-bird)

Mismo modelo que v2-bird sin recompra (CatBoost, P(compra) por impresión, puntúa todo el catálogo y nunca
ofrece lo ya comprado). Lo que cambia es **el dataset**: qué filas, qué features y cómo se calculan.

- Fórmulas: `reco/bird/feat5.py` (una sola implementación para entrenar y para servir).
- Entrenamiento: `reco/bird/build_v5.py` → `data/bird/v5/{train,validation,test,live}_v5.parquet`.
- Servicio: `reco/bird/live5.py` (`LiveState5`) y política `bird_v5` / `v5-bird` (`reco/policies/bird.py`).
- Todo en uno: `python scripts/bird_v5.py dataset | compare | train | offline | state | parity`.

## 1. Qué se midió antes de tocar nada (14/11)

| Hallazgo | Medida | Consecuencia en el dataset |
|---|---|---|
| El stream en vivo tiene ~29 veces más eventos por hora que el histórico | 240.000 eventos/h en `shop.events` frente a 200.000/día en `shop.history_events` | Un recuento absoluto no significa lo mismo al entrenar que al servir. Todas las features de popularidad y de usuario pasan a ser cuotas, cocientes o lifts |
| Las features de v2 que eran recuentos derivan mucho entre histórico y vivo | deriva = \|2·AUC−1\| de la feature sola para distinguir una fila del histórico de una en vivo: `it_trend` 0,99 · `it_buys_1d` 0,89 · `u_days_since_last` 0,75 · `it_views_7d` 0,74 · `it_buys_7d` 0,58. Media de las 12 que se quitan: **0,44**; media de las 72 numéricas de v5: **0,12** | Se quitan o se sustituyen por su versión invariante al volumen |
| En el histórico no hay ni un cambio de precio; en vivo hay promociones de 2 h con un 11–20 % de descuento | 4.596 `price_change` sobre 1.799 productos. Mismo producto con y sin descuento: vistas +20 %, compras +32 % (orgánico). En nuestro carrusel: CTR +41 %, CVR +42 % | `it_discount`; el efecto solo se puede aprender de las filas de hoy |
| Entran 100 productos nuevos cada 4 h y venden desde el primer minuto | 44 vistas/h por producto nuevo (media del catálogo: 39) y 6,5 % de compra por vista (media: 5,3 %). En nuestro carrusel convierten 2,6 veces más que el resto | Para un modelo del histórico un producto sin historia vale 0. Tasas y lifts encogidos hacia su subcategoría; `it_trend_1h` |
| En el histórico la popularidad no tiene dinámica; en vivo sí | compras por día del histórico: varianza/media = 1,0 (Poisson puro). Vivo por hora: la desviación de la última hora predice la siguiente (correlación residual 0,33 frente a 0,04) | Ventanas de 1 h y 24 h en forma de lift suavizado: en el histórico valen ~0 y no molestan |
| El dispositivo y la región cambian lo que se compra, y un usuario siempre usa el mismo | dispositivo × categoría: lift 0,63–1,38; × marca 0,34–2,18; × atributo 0,74–1,32. Región × categoría: 0,72–1,18 | Para un comprador nuevo en la home es lo único que se sabe de él: 10 features `ctx_*` |
| El 97 % de las peticiones marcadas `is_new_user` ya tienen eventos de hoy antes de la petición | filas en vivo del dataset | v2 tiraba esa historia (trataba al usuario como desconocido). v5 la usa |
| Señal en nuestro carrusel (AUC de cada feature sola frente a la compra, sin recompras) | afinidad por atributos 0,78 · cuota de compras del usuario 0,78 · categoría 0,76 · sesión (atributos) 0,69 | Más detalle justo donde está la señal: atributos, sesión, afinidad reciente |

## 2. Filas

| Parte | Origen | Peticiones | Compras | Muestreo |
|---|---|---|---|---|
| `train` | `shop.history_sessions` 26/09 → 30/10 | 88.702 | 16.372 | todos los carruseles con clic + 1 de cada 25 sin clic (`sample_weight` 25) |
| `validation` | ídem 31/10 → 06/11 | 17.699 | 3.238 | ídem |
| `test` | ídem 07/11 → 14/11 | 18.805 | 3.530 | ídem |
| `live` | `team_17.feedback` × `shop.requests`, 14/11 10:36 → 22:30 UTC | 17.739 | 1.623 | todos los carruseles con clic + 1 de cada 10 sin clic (`sample_weight` 10) |

Al entrenar se quitan las impresiones de productos que el usuario ya había comprado (opción sin recompra) y, en
el 30 % de las peticiones del histórico, se esconde la historia del usuario (cold start simulado).
Las filas en vivo pesan el triple (`--live-weight 3`): son las únicas que miden al simulador de verdad.

## 3. Las 81 features

Leyenda de la columna «Tablas»: HE = `shop.history_events`, EV = `shop.events`, HS = `shop.history_sessions`,
RQ = `shop.requests`, CA = `shop.catalog`, CU = `shop.catalog_updates`, FB = `team_17.feedback`.
«v2» = ya estaba en v2-bird con la misma definición; «v2*» = estaba, con un cambio que se explica; «nueva».

### 3.1 Producto: catálogo y precio (9)

| Feature | Qué es | Tablas | Estado | Por qué |
|---|---|---|---|---|
| `category`, `subcategory`, `brand` | categóricas del catálogo | CA, CU | v2 | CatBoost las convierte en estadísticas del objetivo ordenadas, sin fuga [7]. Con las filas en vivo aprenden qué grupos premia el simulador |
| `attr1`, `attr2` | los 2 atributos (de 20) que aparecen en la descripción | CA, CU | v2 | Es lo único del texto con señal y lo tienen también los productos nuevos: base del cold start de producto [9] |
| `price_eur` | precio **en el instante de la petición** | CA, CU | v2* | Antes era el precio del catálogo. En vivo cambia cada 2 h |
| `it_price_vs_sub` | log(precio base) − mediana de su subcategoría | CA | v2 | El mismo precio es caro o barato según la categoría [14] |
| `it_age_days` | días desde `created_at` | CA, CU | v2 | Distingue los productos nuevos de hoy (menos de un día) |
| `it_discount` | 1 − precio actual / precio base | CU | **nueva** | Promoción activa. Medido: +42 % de conversión en nuestro carrusel |

### 3.2 Producto: popularidad orgánica (10)

Recuentos de vistas, carritos y compras orgánicas hasta el último tramo completo anterior a la petición
(1 h en el histórico, 10 min en vivo), convertidos en magnitudes que no dependen del volumen de tráfico.

| Feature | Qué es | Tablas | Estado | Por qué |
|---|---|---|---|---|
| `it_buy_share`, `it_view_share` | cuota de las compras / vistas de siempre (×10⁴) | HE, EV | v2 | La popularidad es la señal de base más fuerte del histórico [2] |
| `it_buy_rate`, `it_cart_rate` | compras (carritos) por vista, encogidas hacia la tasa de su **subcategoría** con 50 vistas ficticias | HE, EV | v2* | Antes el prior era una constante: un producto nuevo salía con la tasa más baja del catálogo. Encoger hacia el grupo es Bayes empírico jerárquico [10] |
| `it_buy_share_7d`, `it_view_share_7d`, `it_buy_share_1d` | cuota dentro de los últimos 7 días / 24 h | HE, EV | **nuevas** (sustituyen a `it_buys_7d`, `it_views_7d`, `it_buys_1d`) | La popularidad reciente predice mejor que la de siempre [5]; como cuota vale igual con 200.000 eventos al día que a la hora |
| `it_trend_1d` | log de las vistas de las últimas 24 h entre las que le tocarían por su cuota de siempre (20 vistas ficticias) | HE, EV | **nueva** (sustituye a `it_trend`) | La `it_trend` de v2 era un cociente de recuentos sin normalizar: deriva 0,99. Esta, 0,40 |
| `it_trend_1h` | log de las vistas de la última hora entre las que le tocarían por su cuota de la semana (5 vistas ficticias) | HE, EV | **nueva** | Detecta en minutos productos nuevos, promociones y vueltas de stock. Las preferencias y la popularidad cambian con el tiempo [13] |
| `it_sub_rank` | puesto dentro de su subcategoría por compras de 7 días | HE, EV | v2 | Comparación entre sustitutos directos |

### 3.3 Producto: respuesta en el carrusel (4)

| Feature | Qué es | Tablas | Estado | Por qué |
|---|---|---|---|---|
| `it_c_ctr`, `it_c_cvr` | clics y compras por impresión en el carrusel del sistema viejo, encogidos hacia la media con 200 impresiones ficticias | HS | v2 | La respuesta pasada del producto en la misma superficie es de lo más predictivo que hay [2] |
| `it_fb_click_lift`, `it_fb_buy_lift` | clics (compras) reales en **nuestro** carrusel entre los que habría tenido un producto medio en esas mismas posiciones (COEC); 1 si no hay impresiones | FB | **nuevas** | Es la única medida directa de lo que premia el simulador. Normalizar por posición evita premiar a quien salió arriba [3][4] |

### 3.4 Contexto (13)

| Feature | Qué es | Tablas | Estado | Por qué |
|---|---|---|---|---|
| `page`, `device`, `region` | categóricas de la petición | RQ, HS | v2 | En la home no hay vistas: el CTR es menos de la mitad que en una página de producto |
| `ctx_dev_item`, `ctx_dev_sub`, `ctx_dev_cat`, `ctx_dev_brand`, `ctx_dev_attr`, `ctx_dev_price` | log-lift de P(producto o grupo \| dispositivo) frente a P(producto o grupo), con compras orgánicas; el de producto se encoge hacia el de su subcategoría | HE, EV | **nuevas** | Recomendación sensible al contexto [18]. En v2 `device` y `region` tenían importancia 0: como categóricas sueltas no dicen nada de un candidato concreto; cruzadas con el candidato sí |
| `ctx_reg_item`, `ctx_reg_sub`, `ctx_reg_cat`, `ctx_reg_attr` | lo mismo por región | HE, EV | **nuevas** | Para un comprador nuevo en la home es todo lo que se sabe de él (cold start, 14 % de la nota) |

### 3.5 Sesión: `recent_views` frente al candidato (21)

| Feature | Qué es | Estado | Por qué |
|---|---|---|---|
| `n_recent_views` | nº de vistas de la sesión | v2 | Cuánta evidencia hay de la intención |
| `recently_viewed`, `sv_recency`, `sv_count` | el candidato está entre lo visto, cómo de reciente (1/k) y cuántas veces | v2 | Lo visto y no comprado es el candidato natural |
| `sv_same_sub_last`, `sv_same_cat_last`, `sv_same_brand_last` | coincide con la última vista | v2 | La última acción es la que más pesa en recomendación por sesión [6] |
| `sv_sub_share`, `sv_cat_share`, `sv_brand_share`, `sv_attr_share` | parte de las vistas que comparten subcategoría / categoría / marca / atributos con el candidato | v2 | `sv_attr_share` y `sv_cat_share` son las features con más señal en nuestro carrusel |
| `sv_logp_diff` | precio del candidato frente a la media de lo visto | v2 | Nivel de precio de la sesión |
| `view_to_purchase_score`, `coview_score` | vecinos item-item: «quien miró X compró Y» y «vistos en la misma sesión», ponderados por recencia | v2 | Filtrado colaborativo por vecinos entre productos [19]; en sesiones cortas iguala o supera a redes recurrentes [6] |
| `sv_secs_since_last` | segundos entre la última vista y la petición | **nueva** | Frescura de la intención |
| `sv_attr_last` | cuántos atributos comparte con la última vista (0–2) | **nueva** | La media de la sesión diluye la última vista |
| `sv_attr_w`, `sv_sub_w`, `sv_cat_w` | las cuotas anteriores ponderando cada vista por 0,6^k | **nuevas** | Peso decreciente con la antigüedad dentro de la sesión [6] |
| `sv_logp_diff_last` | precio frente a la última vista | **nueva** | Ídem para el precio |
| `sv_n_subs` | subcategorías distintas vistas | **nueva** | Sesión enfocada frente a sesión que curiosea |

### 3.6 Usuario (6)

| Feature | Qué es | Estado | Por qué |
|---|---|---|---|
| `is_new_user` | marca de la petición (en el histórico: sin eventos anteriores) | v2* | Ya no borra la historia: un comprador nuevo con eventos de hoy la usa |
| `u_hist_conf` | eventos / (eventos + 50) | **nueva** (sustituye a `u_events`) | Cuánto fiarse del perfil. Saturada: 29 veces más actividad la mueve de 0,87 a 0,99, no ×29 |
| `u_buy_ratio` | compras / eventos | **nueva** (sustituye a `u_buys`) | Propensión a comprar, sin depender de cuánta historia hay |
| `u_logp_diff`, `u_buy_logp_diff` | precio del candidato frente a lo que el usuario mira / compra | v2 | Sensibilidad al precio por usuario [14] |
| `u_logp_z` | lo mismo en desviaciones típicas del usuario | **nueva** | Un usuario que mira de todo tolera más diferencia |

### 3.7 Afinidad usuario × candidato (13)

Historia del usuario **anterior a la petición** (vista 1, carrito 3, compra 5; clics y compras de carruseles
anteriores), resumida frente al candidato. Es la versión manual de «activar los intereses del usuario que
tienen que ver con el candidato» de Deep Interest Network [8].

| Feature | Qué es | Estado | Por qué |
|---|---|---|---|
| `u_sub_share`, `u_brand_share`, `u_cat_share`, `u_attr_share` | parte del peso de su historia en la subcategoría / marca / categoría / atributos del candidato | v2 | `u_attr_share` y `u_cat_share` eran el 31 % de la importancia de v2 |
| `u_attr_max_share`, `u_attr_min_share` | el atributo del candidato que más (y que menos) le gusta | **nuevas** | La media esconde si encajan los dos atributos o solo uno |
| `u_sub_share_rec`, `u_cat_share_rec`, `u_attr_share_rec` | las mismas cuotas con cada evento descontado con semivida de 2 días | **nuevas** (sustituyen a `u_sub_w_7d`) | Las preferencias derivan con el tiempo [13]. Como cociente de sumas descontadas no depende del volumen ni del instante en que se descarga el estado |
| `u_sub_buy_share`, `u_brand_buy_share`, `u_cat_buy_share`, `u_attr_buy_share` | las mismas cuotas contando solo compras | **nuevas** (sustituyen a `u_sub_buys`, `u_brand_buys`, `u_sub_w`, `u_brand_w`) | Comprar es una señal más limpia que mirar |

### 3.8 Usuario × producto (4) y posición (1)

| Feature | Qué es | Estado | Por qué |
|---|---|---|---|
| `up_views`, `up_carts`, `up_c_clicks`, `up_days_since_seen` | vistas, carritos y clics de carrusel de ESE producto por ese usuario, y días desde la última vez | v2 | Interés explícito sin compra |
| `position` | hueco del carrusel (1–10) al entrenar; al servir, todos los candidatos con la posición 1 | v2 | Sesgo de posición: se aprende como feature y se fija al servir [1, regla 36][16] |

**Fuera a propósito** (grupo de recompra): `up_buys`, `up_c_buys`, `previous_purchase_count`, `already_purchased`,
`days_since_last_purchase`. En el histórico predicen muchísimo; en vivo el simulador penaliza recomprar.

**Se quitan de v2** (recuentos absolutos, deriva media 0,44): `it_buys_7d`, `it_views_7d`, `it_buys_1d`, `it_trend`,
`u_events`, `u_buys`, `u_days_since_last`, `u_sub_w`, `u_brand_w`, `u_sub_w_7d`, `u_sub_buys`, `u_brand_buys`.
Siguen en los Parquet (no como features) para poder entrenar el esquema de v2 con las mismas filas.

## 4. Entrenar y servir dicen lo mismo

- Las fórmulas están una sola vez (`feat5.py`); entrenamiento y servicio solo consiguen los recuentos [1, reglas 29 y 32].
- `python scripts/bird_v5.py parity`: descarga el estado a fecha de un corte y compara, para peticiones reales de
  los minutos siguientes, las features que daría el servicio con las del dataset [1, regla 37]. Resultado en §5.
- Cada 2 min el servicio vuelve a pedir catálogo, popularidad y feedback (3 consultas de segundos): una promoción
  de 2 h o un producto nuevo se ven casi al momento. En v2 esas tablas se refrescaban cada 30 min.

## 5. Resultados

**Paridad entrenamiento ↔ servicio** (`parity`, antes de las dos últimas correcciones):
- Corte 07/11 (80 peticiones): 99,1 % de los valores iguales. Corte en vivo 14/11 20:00 UTC (96 peticiones):
  popularidad, precio, promoción, sesión, usuario y afinidad, 100 %.
- Diferencias encontradas y corregidas: la mediana de precio de la subcategoría se movía al entrar productos
  nuevos (ahora solo usa el catálogo inicial) y los pares usuario × producto de una sola vista no existen en el
  estado de servicio (ahora el dataset aplica la misma regla, `feat5.stored_pair`).
- Diferencias que quedan, por diseño: `ctx_*` (al entrenar, compras hasta el inicio de la semana; al servir, hasta
  ahora; correlación ≥ 0,98), `it_fb_*` (curva de posición de todo el día frente a la del momento; 0,9995) y los
  vecinos de co-vista (desempates del `LIMIT 30 BY`; ≥ 0,98).

**Modelos**

| Modelo | Datos | Árboles | Validación | ROC-AUC | PR-AUC | NDCG en los 10 mostrados |
|---|---|---|---|---|---|---|
| `v2_fresh` (referencia, 56 features) | histórico < 31/10 | 131 | histórico 31/10–06/11 | 0,945 | 0,025 | 0,781 |
| `v5_fresh` (81 features) | histórico < 31/10 | 132 | histórico 31/10–06/11 | 0,951 | 0,029 | 0,775 |
| `v5_live` (81 features, producción) | histórico completo + vivo | 255 | últimas horas en vivo (210 compras) | 0,948 | 0,089 | 0,777 |

En el histórico v5 y v2 empatan: es donde los recuentos absolutos que se quitan no derivan. Las métricas de
`v5_live` son de la pasada de prueba que eligió el número de árboles (el último 15 % del vivo fuera).

**Pendiente (no se ha medido):** `bird_v5` frente a `bird_v2` sobre el catálogo completo en los tests `organic` e
`history` (`python scripts/bird_v5.py offline` o `make offline C="bird_v5"`), la comparación con las mismas filas
(`compare`) y la latencia de `recommend()`. Hasta tener eso, v5-bird solo debe ir al vivo en A/B simultáneo.

## 6. Fuentes

1. M. Zinkevich. *Rules of Machine Learning: Best Practices for ML Engineering* (Google). Reglas 29, 31, 32, 36 y 37. <https://developers.google.com/machine-learning/guides/rules-of-ml>
2. X. He et al. *Practical Lessons from Predicting Clicks on Ads at Facebook*. ADKDD 2014. <https://ai.meta.com/research/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/>
3. T. Joachims, A. Swaminathan, T. Schnabel. *Unbiased Learning-to-Rank with Biased Feedback*. WSDM 2017 (mejor artículo). <https://arxiv.org/abs/1608.04468>
4. R. Demsyn-Jones. *Position bias in features*. 2024 (describe COEC y lo atribuye a Zhang y Jones, 2007). <https://arxiv.org/abs/2402.02626> · W. V. Zhang, R. Jones. *Comparing click logs and editorial labels for training query rewriting*. WWW 2007 Query Log Analysis Workshop.
5. Y. Ji, A. Sun, J. Zhang, C. Li. *A Re-visit of the Popularity Baseline in Recommender Systems*. SIGIR 2020. <https://arxiv.org/abs/2005.13829>
6. M. Ludewig, D. Jannach. *Evaluation of Session-based Recommendation Algorithms*. UMUAI 28, 2018. <https://arxiv.org/abs/1803.09587>
7. L. Prokhorenkova et al. *CatBoost: unbiased boosting with categorical features*. NeurIPS 2018. <https://arxiv.org/abs/1706.09516>
8. G. Zhou et al. *Deep Interest Network for Click-Through Rate Prediction*. KDD 2018. <https://arxiv.org/abs/1706.06978>
9. M. Volkovs, G. Yu, T. Poutanen. *DropoutNet: Addressing Cold Start in Recommender Systems*. NeurIPS 2017. <https://proceedings.neurips.cc/paper/2017/file/dbd22ba3bd0df8f385bdac3e9f8be207-Paper.pdf>
10. D. Micci-Barreca. *A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems*. SIGKDD Explorations 3(1), 2001. <https://doi.org/10.1145/507533.507538>
11. P. Romov, E. Sokolov. *RecSys Challenge 2015: ensemble learning with categorical features* (solución ganadora: predicción de compra en sesiones de comercio electrónico con gradient boosting). <https://doi.org/10.1145/2813448.2813511>
12. L. Grinsztajn, E. Oyallon, G. Varoquaux. *Why do tree-based models still outperform deep learning on typical tabular data?* NeurIPS 2022. <https://arxiv.org/abs/2207.08815>
13. Y. Koren. *Collaborative Filtering with Temporal Dynamics*. KDD 2009 (mejor artículo). <https://doi.org/10.1145/1557019.1557072>
14. Y. Zheng et al. *Price-aware Recommendation with Graph Convolutional Networks*. ICDE 2020. <https://arxiv.org/abs/2003.03975>
15. P. Covington, J. Adams, E. Sargin. *Deep Neural Networks for YouTube Recommendations*. RecSys 2016. <https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/>
16. H. Guo et al. *PAL: a position-bias aware learning framework for CTR prediction in live recommender systems*. RecSys 2019. <https://doi.org/10.1145/3298689.3347033>
17. J. Pan et al. *Adversarial Validation Approach to Concept Drift Problem in User Targeting Automation Systems at Uber*. AdKDD 2020. <https://arxiv.org/abs/2004.03045>
18. G. Adomavicius, B. Mobasher, F. Ricci, A. Tuzhilin. *Context-Aware Recommender Systems*. AI Magazine 32(3), 2011. <https://ojs.aaai.org/index.php/aimagazine/article/view/2364>
19. B. Sarwar, G. Karypis, J. Konstan, J. Riedl. *Item-based Collaborative Filtering Recommendation Algorithms*. WWW 2001. <https://doi.org/10.1145/371920.372071>
