# Comparación de entrenamiento Bird v5

Dataset reconstruido el 15 de noviembre de 2026 a partir del histórico y datos en vivo disponibles hasta las 04:20 UTC. Prueba temporal con 737 compras de carrusel y 4.074 compras orgánicas, excluyendo recompras. Comando: `python scripts/bird_v5.py compare-org --org-weights 0.03 0.1 0.3`.

| Peso orgánico | NDCG compra carrusel | Diferencia frente a cero (IC 95 %) | NDCG compra orgánica |
|---:|---:|---:|---:|
| 0 | 0,6866 | referencia | 0,7119 |
| 0,03 | 0,6839 | −0,0027 [−0,0177, +0,0126] | 0,7215 |
| 0,1 | 0,6788 | −0,0078 [−0,0226, +0,0074] | 0,7347 |
| 0,3 | 0,6866 | −0,0000 [−0,0150, +0,0156] | 0,7403 |

Ninguna variante demuestra una mejora en compras del carrusel. En clics, los pesos 0,03 y 0,3 empeoran frente a la referencia con intervalos que excluyen cero. La métrica orgánica mejora, pero no basta para justificar un despliegue. Estos resultados son exploratorios: seleccionar y confirmar variantes requiere un test temporal independiente y, antes de producción, el protocolo A/B del proyecto.

## Segunda iteración

Con `compare-org --org-weights 0.1 --cold-share 0.1 --live-weight 6`, la referencia sin orgánicas obtuvo NDCG de compra 0,6719 y la variante orgánica 0,6945: diferencia +0,0225, IC 95 % [+0,0074, +0,0382]. En clics la diferencia fue +0,0069 [−0,0011, +0,0148]; el NDCG orgánico pasó de 0,7141 a 0,7316. Esta comparación es frente a la referencia con los mismos pesos de esta iteración, no frente a la configuración original. Se ha usado el mismo test para explorar ambas iteraciones, así que falta una comparación pareada directa y confirmación independiente antes de considerar el candidato mejor.
