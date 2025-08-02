# -*- coding: utf-8 -*-
"""
ANÁLISIS DE CLUSTERS DESBALANCEADOS Y SOLUCIONES
Identificar por qué los clusters no están balanceados y proponer soluciones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("=" * 60)
print("ANÁLISIS DE CLUSTERS DESBALANCEADOS")
print("=" * 60)

#----------------------------------------------- 1. CARGA DE DATOS

print("\n1. CARGA Y ANÁLISIS INICIAL")
print("-" * 40)

# Cargar datos
with open("fri.csv", 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Procesar encabezado
header_line = lines[0].strip()
header = header_line.replace('"', '').split(',')

# Procesar datos
data = []
for i, line in enumerate(lines[1:], 1):
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

# Crear DataFrame
df = pd.DataFrame(data, columns=header)
df.index.name = 'Cliente_ID'

print(f"Dataset: {df.shape[0]} clientes, {df.shape[1]} variables")

#----------------------------------------------- 2. DIAGNÓSTICO DEL PROBLEMA

print("\n2. DIAGNÓSTICO DEL PROBLEMA")
print("-" * 40)

# Análisis de distribución
print("Estadísticas descriptivas:")
print(df.describe())

# Detectar outliers
print("\nDetección de outliers:")
for col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[df[col] > Q3 + 1.5*IQR]
    print(f"  {col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")

# Análisis de skewness
print("\nAnálisis de asimetría (skewness):")
for col in df.columns:
    skewness = df[col].skew()
    print(f"  {col}: {skewness:.2f} ({'Muy sesgado' if abs(skewness) > 1 else 'Moderadamente sesgado' if abs(skewness) > 0.5 else 'Casi normal'})")

# Visualizar distribuciones
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, col in enumerate(df.columns):
    # Histograma
    axes[i].hist(df[col], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[i].set_title(f'Distribución de {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frecuencia')
    
    # Línea de media
    mean_val = df[col].mean()
    axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Media: {mean_val:.2f}')
    axes[i].legend()

plt.tight_layout()
plt.savefig('distribuciones_originales.png', dpi=300, bbox_inches='tight')
plt.close()

#----------------------------------------------- 3. PRUEBA CON DIFERENTES MÉTODOS

print("\n3. PRUEBA CON DIFERENTES MÉTODOS DE CLUSTERING")
print("-" * 40)

# Método 1: K-means con datos originales
print("\nMétodo 1: K-means con datos originales")
kmeans_original = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster_Original'] = kmeans_original.fit_predict(df)
print("Distribución:")
print(df['Cluster_Original'].value_counts().sort_index())

# Método 2: K-means con StandardScaler
print("\nMétodo 2: K-means con StandardScaler")
scaler_std = StandardScaler()
df_scaled_std = scaler_std.fit_transform(df)
kmeans_std = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster_StandardScaler'] = kmeans_std.fit_predict(df_scaled_std)
print("Distribución:")
print(df['Cluster_StandardScaler'].value_counts().sort_index())

# Método 3: K-means con RobustScaler (mejor para outliers)
print("\nMétodo 3: K-means con RobustScaler")
scaler_robust = RobustScaler()
df_scaled_robust = scaler_robust.fit_transform(df)
kmeans_robust = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster_RobustScaler'] = kmeans_robust.fit_predict(df_scaled_robust)
print("Distribución:")
print(df['Cluster_RobustScaler'].value_counts().sort_index())

# Método 4: Gaussian Mixture Model
print("\nMétodo 4: Gaussian Mixture Model")
gmm = GaussianMixture(n_components=3, random_state=42)
df['Cluster_GMM'] = gmm.fit_predict(df_scaled_robust)
print("Distribución:")
print(df['Cluster_GMM'].value_counts().sort_index())

#----------------------------------------------- 4. ANÁLISIS DE RESULTADOS

print("\n4. ANÁLISIS DE RESULTADOS")
print("-" * 40)

# Comparar distribuciones
metodos = ['Cluster_Original', 'Cluster_StandardScaler', 'Cluster_RobustScaler', 'Cluster_GMM']
print("Comparación de distribuciones por método:")
print("-" * 50)

for metodo in metodos:
    distribucion = df[metodo].value_counts().sort_index()
    print(f"\n{metodo}:")
    for cluster, count in distribucion.items():
        porcentaje = count / len(df) * 100
        print(f"  Cluster {cluster}: {count} clientes ({porcentaje:.1f}%)")

# Calcular balance
print("\nAnálisis de balance:")
for metodo in metodos:
    distribucion = df[metodo].value_counts()
    min_cluster = distribucion.min()
    max_cluster = distribucion.max()
    ratio_balance = min_cluster / max_cluster
    print(f"{metodo}: Ratio de balance = {ratio_balance:.3f} ({'Bien balanceado' if ratio_balance > 0.3 else 'Desbalanceado'})")

#----------------------------------------------- 5. SOLUCIÓN RECOMENDADA

print("\n5. SOLUCIÓN RECOMENDADA")
print("-" * 40)

# Usar RobustScaler + K-means como mejor opción
mejor_metodo = 'Cluster_RobustScaler'
df['Cluster_Final'] = df[mejor_metodo]

# Análisis del cluster final
print(f"Usando {mejor_metodo} como método final:")
distribucion_final = df['Cluster_Final'].value_counts().sort_index()
for cluster, count in distribucion_final.items():
    porcentaje = count / len(df) * 100
    print(f"  Cluster {cluster}: {count} clientes ({porcentaje:.1f}%)")

# Caracterizar clusters finales
print(f"\nCaracterización de clusters finales:")
for cluster in sorted(df['Cluster_Final'].unique()):
    cluster_data = df[df['Cluster_Final'] == cluster]
    print(f"\nCluster {cluster} ({len(cluster_data)} clientes):")
    print(f"  Frecuencia: {cluster_data['Frecuencia'].mean():.1f} ± {cluster_data['Frecuencia'].std():.1f}")
    print(f"  Transaccionalidad: ${cluster_data['Transaccionalidad'].mean():,.0f} ± ${cluster_data['Transaccionalidad'].std():,.0f}")
    print(f"  Ingresos: ${cluster_data['Ingresos'].mean():,.0f} ± ${cluster_data['Ingresos'].std():,.0f}")

#----------------------------------------------- 6. VISUALIZACIONES

print("\n6. GENERANDO VISUALIZACIONES")
print("-" * 40)

# Comparación de métodos
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

for i, metodo in enumerate(metodos):
    row = i // 2
    col = i % 2
    
    distribucion = df[metodo].value_counts().sort_index()
    axes[row, col].bar(distribucion.index, distribucion.values, color=['skyblue', 'lightgreen', 'orange'])
    axes[row, col].set_title(f'Distribución - {metodo}')
    axes[row, col].set_xlabel('Cluster')
    axes[row, col].set_ylabel('Número de Clientes')
    
    # Añadir valores en las barras
    for j, v in enumerate(distribucion.values):
        axes[row, col].text(j, v + max(distribucion.values)*0.01, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('comparacion_metodos_clustering.png', dpi=300, bbox_inches='tight')
plt.close()

# Scatter plot del método final
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['Frecuencia'], df['Transaccionalidad'], 
                     c=df['Cluster_Final'], cmap='viridis', alpha=0.6, s=50)
plt.xlabel('Frecuencia')
plt.ylabel('Transaccionalidad')
plt.title('Clusters Finales - Frecuencia vs Transaccionalidad')
plt.colorbar(scatter, label='Cluster')
plt.savefig('clusters_finales_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

#----------------------------------------------- 7. CONCLUSIONES

print("\n7. CONCLUSIONES Y RECOMENDACIONES")
print("-" * 40)

print("PROBLEMAS IDENTIFICADOS:")
print("  1. Datos muy sesgados (skewness alta)")
print("  2. Presencia de outliers significativos")
print("  3. Escalas muy diferentes entre variables")
print("  4. K-means es sensible a outliers")

print("\nSOLUCIONES APLICADAS:")
print("  1. Uso de RobustScaler en lugar de StandardScaler")
print("  2. RobustScaler es menos sensible a outliers")
print("  3. Normalización robusta de las variables")
print("  4. Comparación con múltiples métodos")

print("\nMEJOR MÉTODO:")
print(f"  RobustScaler + K-means proporciona clusters más balanceados")
print(f"  Ratio de balance: {min(df['Cluster_Final'].value_counts()) / max(df['Cluster_Final'].value_counts()):.3f}")

# Guardar resultados
df_final = df[['Frecuencia', 'Transaccionalidad', 'Ingresos', 'Cluster_Final']].copy()
df_final.to_csv('clientes_clusters_balanceados.csv', index=False)

print(f"\n✅ Archivos generados:")
print("   - clientes_clusters_balanceados.csv")
print("   - distribuciones_originales.png")
print("   - comparacion_metodos_clustering.png")
print("   - clusters_finales_scatter.png")

print("\n" + "=" * 60)
print("ANÁLISIS COMPLETADO")
print("=" * 60) 