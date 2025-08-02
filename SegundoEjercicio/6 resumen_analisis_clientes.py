import pandas as pd

print("=" * 60)
print("RESUMEN EJECUTIVO - ANÁLISIS DE CLIENTES")
print("=" * 60)

# Cargar resultados
df = pd.read_csv('clientes_con_indice_y_clusters.csv')

print("\n DATOS GENERALES:")
print(f"   • Total de clientes analizados: {len(df):,}")
print(f"   • Variables consideradas: Frecuencia, Transaccionalidad, Ingresos")
print(f"   • Índice de cliente creado con pesos ponderados")

print("\n ÍNDICE DE CLIENTE:")
print("   • Frecuencia: 30% del índice")
print("   • Transaccionalidad: 40% del índice (más importante)")
print("   • Ingresos: 30% del índice")

# Análisis por cluster
clusters = df.groupby('Cluster').agg({
    'Frecuencia': 'mean',
    'Transaccionalidad': 'mean',
    'Ingresos': 'mean',
    'Indice_Cliente': 'mean'
}).round(2)

print(f"\n CARACTERIZACIÓN DE GRUPOS DE CLIENTES:")
print("-" * 50)

# Cluster 0 - Regulares
cluster_0 = df[df['Cluster'] == 0]
print(f"\n CLIENTES REGULARES (Cluster 0):")
print(f"   • Cantidad: {len(cluster_0)} clientes ({len(cluster_0)/len(df)*100:.1f}%)")
print(f"   • Frecuencia media: {cluster_0['Frecuencia'].mean():.1f} transacciones")
print(f"   • Transaccionalidad media: ${cluster_0['Transaccionalidad'].mean():,.2f}")
print(f"   • Ingresos medios: ${cluster_0['Ingresos'].mean():,.2f}")
print(f"   • Índice medio: {cluster_0['Indice_Cliente'].mean():.2f}")
print(f"   • Estrategia: Activación y engagement")
print(f"   • Acciones: Campañas de marketing, educación")

# Cluster 1 - Caso especial
cluster_1 = df[df['Cluster'] == 1]
if len(cluster_1) > 0:
    print(f"\n CLIENTE ESPECIAL (Cluster 1):")
    print(f"   • Cantidad: {len(cluster_1)} cliente ({len(cluster_1)/len(df)*100:.1f}%)")
    print(f"   • Características únicas: Ingresos muy altos")
    print(f"   • Estrategia: Gestión personalizada")

# Cluster 2 - Alto Valor
cluster_2 = df[df['Cluster'] == 2]
print(f"\n CLIENTES DE ALTO VALOR (Cluster 2):")
print(f"   • Cantidad: {len(cluster_2)} clientes ({len(cluster_2)/len(df)*100:.1f}%)")
print(f"   • Frecuencia media: {cluster_2['Frecuencia'].mean():.1f} transacciones")
print(f"   • Transaccionalidad media: ${cluster_2['Transaccionalidad'].mean():,.2f}")
print(f"   • Ingresos medios: ${cluster_2['Ingresos'].mean():,.2f}")
print(f"   • Índice medio: {cluster_2['Indice_Cliente'].mean():.2f}")
print(f"   • Estrategia: Incrementar frecuencia")
print(f"   • Acciones: Programas de recompensas, recordatorios")

print(f"\n INSIGHTS PRINCIPALES:")
print("-" * 30)
print("   • La mayoría de clientes (99.6%) son regulares")
print("   • Solo 0.4% son clientes de alto valor")
print("   • Hay un cliente con características únicas")
print("   • Gran oportunidad de mejora en clientes regulares")

print(f"\n RECOMENDACIONES ESTRATÉGICAS:")
print("-" * 35)
print("   1. Enfocar esfuerzos en activar clientes regulares (99.6%)")
print("   2. Desarrollar programas específicos para clientes de alto valor")
print("   3. Analizar el caso especial del cliente único")
print("   4. Implementar estrategias de up-selling y cross-selling")

print(f"\n ARCHIVOS GENERADOS:")
print("-" * 25)
print("   • clientes_con_indice_y_clusters.csv")
print("   • metodo_codo_clusters.png")
print("   • analisis_clientes_general.png")
print("   • analisis_detallado_clusters.png")
print("   • matriz_correlacion_clientes.png")

print("\n" + "=" * 60) 