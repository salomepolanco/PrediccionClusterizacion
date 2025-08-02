# -*- coding: utf-8 -*-
"""
K-MEANS SIMPLIFICADO CON MEJOR MANEJO DE OUTLIERS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("K-MEANS MEJORADO - MANEJO DE OUTLIERS")
print("=" * 60)

# 1. CARGAR DATOS
print("\n1. CARGANDO DATOS...")
with open("fri.csv", 'r', encoding='utf-8') as file:
    lines = file.readlines()

header_line = lines[0].strip()
header = header_line.replace('"', '').split(',')

data = []
for line in lines[1:]:
    line = line.strip()
    if line:
        values = line.split(',')
        if len(values) == 3:
            try:
                freq = float(values[0])
                trans = float(values[1])
                ing = float(values[2])
                data.append([freq, trans, ing])
            except ValueError:
                continue

df = pd.DataFrame(data, columns=header)
print(f"Dataset original: {df.shape[0]} clientes")

# 2. ANÁLISIS DE OUTLIERS
print("\n2. ANALIZANDO OUTLIERS...")

# Método IQR mejorado
def remove_outliers_iqr(df, columns, factor=2.0):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        df_clean = df_clean[mask]
    return df_clean

# Método percentiles
def remove_outliers_percentile(df, columns, low=2, high=98):
    df_clean = df.copy()
    for col in columns:
        low_bound = df_clean[col].quantile(low / 100)
        high_bound = df_clean[col].quantile(high / 100)
        mask = (df_clean[col] >= low_bound) & (df_clean[col] <= high_bound)
        df_clean = df_clean[mask]
    return df_clean

# Aplicar diferentes estrategias
print("Aplicando diferentes estrategias de limpieza...")

# Estrategia 1: IQR factor 2.0
df_iqr = remove_outliers_iqr(df, df.columns, factor=2.0)
print(f"  IQR (factor=2.0): {len(df_iqr)} clientes ({len(df_iqr)/len(df)*100:.1f}%)")

# Estrategia 2: Percentiles 2-98
df_percentile = remove_outliers_percentile(df, df.columns, low=2, high=98)
print(f"  Percentiles (2-98): {len(df_percentile)} clientes ({len(df_percentile)/len(df)*100:.1f}%)")

# Estrategia 3: Percentiles 5-95
df_percentile_aggressive = remove_outliers_percentile(df, df.columns, low=5, high=95)
print(f"  Percentiles (5-95): {len(df_percentile_aggressive)} clientes ({len(df_percentile_aggressive)/len(df)*100:.1f}%)")

# 3. EVALUAR CLUSTERING
print("\n3. EVALUANDO CLUSTERING...")

def evaluate_clustering(df_clean, name):
    if len(df_clean) < 10:
        return None
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_clean)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)
    
    distribucion = pd.Series(clusters).value_counts().sort_index()
    ratio_balance = distribucion.min() / distribucion.max()
    silhouette = silhouette_score(df_scaled, clusters)
    
    return {
        'name': name,
        'n_clients': len(df_clean),
        'ratio_balance': ratio_balance,
        'silhouette': silhouette,
        'distribution': distribucion.to_dict()
    }

# Evaluar todas las estrategias
estrategias = [
    (df_iqr, "IQR (factor=2.0)"),
    (df_percentile, "Percentiles (2-98)"),
    (df_percentile_aggressive, "Percentiles (5-95)")
]

resultados = []
for df_clean, nombre in estrategias:
    resultado = evaluate_clustering(df_clean, nombre)
    if resultado:
        resultados.append(resultado)

# Mostrar resultados
print("Resultados por estrategia:")
for res in resultados:
    print(f"\n{res['name']}:")
    print(f"  • Clientes: {res['n_clients']}")
    print(f"  • Ratio de balance: {res['ratio_balance']:.3f}")
    print(f"  • Silhouette score: {res['silhouette']:.3f}")
    print(f"  • Distribución: {res['distribution']}")

# 4. SELECCIONAR MEJOR ESTRATEGIA
print("\n4. SELECCIONANDO MEJOR ESTRATEGIA...")

# Crear DataFrame de resultados
df_resultados = pd.DataFrame(resultados)

# Calcular score compuesto
df_resultados['score_compuesto'] = (
    df_resultados['ratio_balance'] * 0.6 +  # Balance es muy importante
    df_resultados['silhouette'] * 0.4        # Calidad de clusters
)

mejor_idx = df_resultados['score_compuesto'].idxmax()
mejor_estrategia = df_resultados.loc[mejor_idx]

print(f" MEJOR ESTRATEGIA: {mejor_estrategia['name']}")
print(f"  • Score compuesto: {mejor_estrategia['score_compuesto']:.3f}")
print(f"  • Ratio de balance: {mejor_estrategia['ratio_balance']:.3f}")
print(f"  • Silhouette score: {mejor_estrategia['silhouette']:.3f}")

# 5. CLUSTERING FINAL
print("\n5. APLICANDO CLUSTERING FINAL...")

# Seleccionar el mejor dataset
if "IQR" in mejor_estrategia['name']:
    df_final = df_iqr
elif "Percentiles (2-98)" in mejor_estrategia['name']:
    df_final = df_percentile
else:
    df_final = df_percentile_aggressive

# Aplicar clustering final
scaler = StandardScaler()
df_scaled_final = scaler.fit_transform(df_final)

kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
df_final['Cluster'] = kmeans_final.fit_predict(df_scaled_final)

# Análisis final
print("Distribución final de clusters:")
distribucion_final = df_final['Cluster'].value_counts().sort_index()
for cluster, count in distribucion_final.items():
    porcentaje = count / len(df_final) * 100
    print(f"  Cluster {cluster}: {count} clientes ({porcentaje:.1f}%)")

# 6. CARACTERIZACIÓN
print("\n6. CARACTERIZACIÓN DE CLUSTERS:")
for cluster in sorted(df_final['Cluster'].unique()):
    cluster_data = df_final[df_final['Cluster'] == cluster]
    
    print(f"\nCluster {cluster} ({len(cluster_data)} clientes):")
    print(f"  Frecuencia: {cluster_data['Frecuencia'].mean():.1f} ± {cluster_data['Frecuencia'].std():.1f}")
    print(f"  Transaccionalidad: ${cluster_data['Transaccionalidad'].mean():,.0f} ± ${cluster_data['Transaccionalidad'].std():,.0f}")
    print(f"  Ingresos: ${cluster_data['Ingresos'].mean():,.0f} ± ${cluster_data['Ingresos'].std():,.0f}")

# 7. VISUALIZACIONES
print("\n7. GENERANDO VISUALIZACIONES...")

# Distribución de clusters
plt.figure(figsize=(10, 6))
bars = plt.bar(distribucion_final.index, distribucion_final.values, color=['skyblue', 'lightgreen', 'orange'])
plt.title('Distribución de Clusters Mejorados')
plt.xlabel('Cluster')
plt.ylabel('Número de Clientes')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + max(distribucion_final.values)*0.01,
             f'{int(height)}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('kmeans_mejorado_distribucion.png', dpi=300, bbox_inches='tight')
plt.close()

# Scatter plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df_final['Frecuencia'], df_final['Transaccionalidad'], 
                     c=df_final['Cluster'], cmap='viridis', alpha=0.6, s=50)
plt.xlabel('Frecuencia')
plt.ylabel('Transaccionalidad')
plt.title('Clusters Mejorados - Frecuencia vs Transaccionalidad')
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig('kmeans_mejorado_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. GUARDAR RESULTADOS
print("\n8. GUARDANDO RESULTADOS...")

df_final.to_csv('clientes_kmeans_mejorado_final.csv')
df_resultados.to_csv('comparacion_estrategias_kmeans.csv', index=False)

print(" Archivos guardados:")
print("  • clientes_kmeans_mejorado_final.csv")
print("  • comparacion_estrategias_kmeans.csv")
print("  • kmeans_mejorado_distribucion.png")
print("  • kmeans_mejorado_scatter.png")

# 9. RESUMEN FINAL
print("\n9. RESUMEN FINAL")
print("-" * 40)

print(f" ESTRATEGIA SELECCIONADA: {mejor_estrategia['name']}")
print(f" RESULTADOS:")
print(f"  • Clientes analizados: {len(df_final)} de {len(df)} originales")
print(f"  • Ratio de balance: {mejor_estrategia['ratio_balance']:.3f}")
print(f"  • Silhouette score: {mejor_estrategia['silhouette']:.3f}")
print(f"  • Calidad: {'Excelente' if mejor_estrategia['silhouette'] > 0.5 else 'Buena' if mejor_estrategia['silhouette'] > 0.3 else 'Aceptable'}")

print(f"\n MEJORAS OBTENIDAS:")
print(f"  • Múltiples métodos de detección de outliers")
print(f"  • Selección objetiva de la mejor estrategia")
print(f"  • Clusters más balanceados")
print(f"  • Mejor calidad de segmentación")

print("\n" + "=" * 60)
print("ANÁLISIS COMPLETADO")
print("=" * 60) 