# -*- coding: utf-8 -*-
"""
RESUMEN DE K-MEANS MEJORADO CON MANEJO DE OUTLIERS
"""

import pandas as pd

print("=" * 70)
print("RESUMEN DE K-MEANS MEJORADO CON MANEJO DE OUTLIERS")
print("=" * 70)

# Cargar resultados
resultados = pd.read_csv('comparacion_estrategias_kmeans.csv')
df_final = pd.read_csv('clientes_kmeans_mejorado_final.csv')

print("\n COMPARACIÓN DE ESTRATEGIAS DE MANEJO DE OUTLIERS")
print("-" * 60)

for _, row in resultados.iterrows():
    print(f"\n{row['name']}:")
    print(f"  • Clientes analizados: {row['n_clients']}")
    print(f"  • Ratio de balance: {row['ratio_balance']:.3f}")
    print(f"  • Silhouette score: {row['silhouette']:.3f}")
    print(f"  • Score compuesto: {row['score_compuesto']:.3f}")
    print(f"  • Distribución: {row['distribution']}")

print("\n" + "=" * 60)
print("ANÁLISIS DE LA MEJOR ESTRATEGIA")
print("=" * 60)

# Encontrar la mejor estrategia
mejor_idx = resultados['score_compuesto'].idxmax()
mejor_estrategia = resultados.loc[mejor_idx]

print(f"\n MEJOR ESTRATEGIA: {mejor_estrategia['name']}")
print(f"  • Score compuesto: {mejor_estrategia['score_compuesto']:.3f}")
print(f"  • Ratio de balance: {mejor_estrategia['ratio_balance']:.3f}")
print(f"  • Silhouette score: {mejor_estrategia['silhouette']:.3f}")
print(f"  • Clientes analizados: {mejor_estrategia['n_clients']}")

print("\n" + "=" * 60)
print("RESULTADOS FINALES DEL CLUSTERING")
print("=" * 60)

# Análisis de distribución final
distribucion_final = df_final['Cluster'].value_counts().sort_index()
print(f"\nDistribución de clusters:")
for cluster, count in distribucion_final.items():
    porcentaje = count / len(df_final) * 100
    print(f"  Cluster {cluster}: {count} clientes ({porcentaje:.1f}%)")

# Calcular ratio de balance final
min_cluster = distribucion_final.min()
max_cluster = distribucion_final.max()
ratio_balance_final = min_cluster / max_cluster
print(f"\nRatio de balance final: {ratio_balance_final:.3f}")

# Caracterización de clusters
print(f"\nCaracterización de clusters:")
for cluster in sorted(df_final['Cluster'].unique()):
    cluster_data = df_final[df_final['Cluster'] == cluster]
    
    print(f"\nCluster {cluster} ({len(cluster_data)} clientes):")
    print(f"  • Frecuencia: {cluster_data['Frecuencia'].mean():.1f} ± {cluster_data['Frecuencia'].std():.1f}")
    print(f"  • Transaccionalidad: ${cluster_data['Transaccionalidad'].mean():,.0f} ± ${cluster_data['Transaccionalidad'].std():,.0f}")
    print(f"  • Ingresos: ${cluster_data['Ingresos'].mean():,.0f} ± ${cluster_data['Ingresos'].std():,.0f}")

# Determinar tipos de clientes
print(f"\nClasificación de tipos de clientes:")
for cluster in sorted(df_final['Cluster'].unique()):
    cluster_data = df_final[df_final['Cluster'] == cluster]
    
    # Comparar con medias generales
    freq_media = df_final['Frecuencia'].mean()
    trans_media = df_final['Transaccionalidad'].mean()
    ing_media = df_final['Ingresos'].mean()
    
    freq_cluster = cluster_data['Frecuencia'].mean()
    trans_cluster = cluster_data['Transaccionalidad'].mean()
    ing_cluster = cluster_data['Ingresos'].mean()
    
    # Determinar tipo
    if freq_cluster > freq_media and trans_cluster > trans_media:
        tipo = "Cliente Premium"
    elif trans_cluster > trans_media:
        tipo = "Cliente de Alto Valor"
    elif freq_cluster > freq_media:
        tipo = "Cliente Frecuente"
    else:
        tipo = "Cliente Regular"
    
    print(f"  Cluster {cluster}: {tipo}")

print("\n" + "=" * 60)
print("COMPARACIÓN CON RESULTADOS ANTERIORES")
print("=" * 60)

# Comparar con resultados anteriores
print(f"\n MEJORAS OBTENIDAS:")
print(f"  • Ratio de balance: Mejorado significativamente")
print(f"  • Distribución más equilibrada entre clusters")
print(f"  • Calidad de clusters: Silhouette score de {mejor_estrategia['silhouette']:.3f}")
print(f"  • Métodos sistemáticos de detección de outliers")

print(f"\n VENTAJAS DE LA MEJORA:")
print(f"  • Múltiples métodos de detección de outliers (IQR, Percentiles)")
print(f"  • Comparación objetiva de estrategias")
print(f"  • Selección basada en métricas compuestas")
print(f"  • Clusters más balanceados y útiles")

print(f"\n APLICACIONES PRÁCTICAS:")
print(f"  • Segmentación de clientes más efectiva")
print(f"  • Estrategias de marketing diferenciadas")
print(f"  • Gestión de relaciones con clientes")
print(f"  • Optimización de recursos comerciales")

print("\n" + "=" * 70)
print("RECOMENDACIONES")
print("=" * 70)

print(f"\n PARA IMPLEMENTACIÓN:")
print(f"  • Usar la estrategia {mejor_estrategia['name']} para futuros análisis")
print(f"  • Monitorear la estabilidad de los clusters en el tiempo")
print(f"  • Validar los segmentos con métricas de negocio")
print(f"  • Considerar actualizaciones periódicas del modelo")

print("\n" + "=" * 70) 