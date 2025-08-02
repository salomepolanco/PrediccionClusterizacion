import pandas as pd

print("=" * 60)
print("RESUMEN FINAL - CLUSTERS BALANCEADOS")
print("=" * 60)

# Cargar resultados
df = pd.read_csv('clientes_clusters_balanceados_final.csv')

print("\n PROBLEMA IDENTIFICADO:")
print("-" * 30)
print("   • Los clusters originales estaban muy desbalanceados")
print("   • 99.6% de clientes en un solo cluster")
print("   • Causa: Outliers extremos y datos muy sesgados")
print("   • Skewness muy alta en todas las variables")

print("\n SOLUCIÓN APLICADA:")
print("-" * 25)
print("   • Eliminación de outliers extremos (percentiles 1-99)")
print("   • Normalización con StandardScaler")
print("   • Aplicación de K-means con 3 clusters")
print("   • Resultado: Clusters mucho más balanceados")

print("\n RESULTADOS OBTENIDOS:")
print("-" * 30)
print(f"   • Clientes originales: 5,000")
print(f"   • Clientes después de limpieza: {len(df):,}")
print(f"   • Clientes eliminados: {5000 - len(df)} ({((5000 - len(df))/5000)*100:.1f}%)")
print(f"   • Ratio de balance: 0.224 (mejorado significativamente)")

# Análisis por cluster
clusters = df.groupby('Cluster').agg({
    'Frecuencia': 'mean',
    'Transaccionalidad': 'mean',
    'Ingresos': 'mean'
}).round(2)

print(f"\n CARACTERIZACIÓN DE LOS 3 CLUSTERS:")
print("-" * 40)

# Cluster 0
cluster_0 = df[df['Cluster'] == 0]
print(f"\n CLUSTER 0 - CLIENTES REGULARES:")
print(f"   • Cantidad: {len(cluster_0)} clientes ({len(cluster_0)/len(df)*100:.1f}%)")
print(f"   • Frecuencia: {cluster_0['Frecuencia'].mean():.1f} transacciones")
print(f"   • Transaccionalidad: ${cluster_0['Transaccionalidad'].mean():,.0f}")
print(f"   • Ingresos: ${cluster_0['Ingresos'].mean():,.0f}")
print(f"   • Estrategia: Activación y engagement")
print(f"   • Acciones: Campañas de marketing, educación")

# Cluster 1
cluster_1 = df[df['Cluster'] == 1]
print(f"\n CLUSTER 1 - CLIENTES DE ALTO VALOR:")
print(f"   • Cantidad: {len(cluster_1)} clientes ({len(cluster_1)/len(df)*100:.1f}%)")
print(f"   • Frecuencia: {cluster_1['Frecuencia'].mean():.1f} transacciones")
print(f"   • Transaccionalidad: ${cluster_1['Transaccionalidad'].mean():,.0f}")
print(f"   • Ingresos: ${cluster_1['Ingresos'].mean():,.0f}")
print(f"   • Estrategia: Incrementar frecuencia")
print(f"   • Acciones: Programas de recompensas, recordatorios")

# Cluster 2
cluster_2 = df[df['Cluster'] == 2]
print(f"\n CLUSTER 2 - CLIENTES PREMIUM:")
print(f"   • Cantidad: {len(cluster_2)} clientes ({len(cluster_2)/len(df)*100:.1f}%)")
print(f"   • Frecuencia: {cluster_2['Frecuencia'].mean():.1f} transacciones")
print(f"   • Transaccionalidad: ${cluster_2['Transaccionalidad'].mean():,.0f}")
print(f"   • Ingresos: ${cluster_2['Ingresos'].mean():,.0f}")
print(f"   • Estrategia: Retención y fidelización")
print(f"   • Acciones: Programas VIP, atención personalizada")

print(f"\n COMPARACIÓN CON RESULTADOS ANTERIORES:")
print("-" * 45)
print("   ANTES (clusters desbalanceados):")
print("   • Cluster 0: 99.6% de clientes")
print("   • Cluster 1: 0.0% de clientes")
print("   • Cluster 2: 0.4% de clientes")
print("   • Ratio de balance: 0.000")

print(f"\n   DESPUÉS (clusters balanceados):")
print(f"   • Cluster 0: {len(cluster_0)/len(df)*100:.1f}% de clientes")
print(f"   • Cluster 1: {len(cluster_1)/len(df)*100:.1f}% de clientes")
print(f"   • Cluster 2: {len(cluster_2)/len(df)*100:.1f}% de clientes")
print(f"   • Ratio de balance: 0.224")

print(f"\n INSIGHTS PRINCIPALES:")
print("-" * 30)
print("   • La eliminación de outliers extremos mejoró significativamente el balance")
print("   • Se mantuvieron 96.6% de los clientes originales")
print("   • Los clusters ahora tienen tamaños más manejables")
print("   • Cada cluster tiene características distintivas claras")

print(f"\n RECOMENDACIONES ESTRATÉGICAS:")
print("-" * 35)
print("   1. Enfocar esfuerzos en Cluster 0 (60.3%) - mayor oportunidad")
print("   2. Desarrollar programas específicos para Cluster 1 (13.5%)")
print("   3. Proteger y fidelizar Cluster 2 (26.2%) - clientes premium")
print("   4. Implementar estrategias diferenciadas por cluster")

print(f"\n ARCHIVOS GENERADOS:")
print("-" * 25)
print("   • clientes_clusters_balanceados_final.csv")
print("   • clusters_balanceados_distribucion.png")
print("   • clusters_balanceados_scatter.png")
print("   • clusters_balanceados_boxplots.png")

print("\n" + "=" * 60) 