# -*- coding: utf-8 -*-
"""
K-MEANS MEJORADO CON TÉCNICAS AVANZADAS DE MANEJO DE OUTLIERS
Múltiples métodos para detectar y manejar outliers antes de aplicar K-Means
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("=" * 70)
print("K-MEANS MEJORADO CON TÉCNICAS AVANZADAS DE OUTLIERS")
print("=" * 70)

#----------------------------------------------- 1. CARGA DE DATOS

print("\n1. CARGA DE DATOS")
print("-" * 50)

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

print(f"Dataset original: {df.shape[0]} clientes")
print(f"Variables: {list(df.columns)}")

#----------------------------------------------- 2. ANÁLISIS INICIAL DE OUTLIERS

print("\n2. ANÁLISIS INICIAL DE OUTLIERS")
print("-" * 50)

# Estadísticas descriptivas
print("Estadísticas descriptivas originales:")
print(df.describe())

# Análisis de skewness
print("\nAnálisis de asimetría original:")
for col in df.columns:
    skewness = df[col].skew()
    print(f"  {col}: {skewness:.2f} ({'Muy sesgado' if abs(skewness) > 1 else 'Moderadamente sesgado' if abs(skewness) > 0.5 else 'Casi normal'})")

#----------------------------------------------- 3. MÚLTIPLES MÉTODOS DE DETECCIÓN DE OUTLIERS

print("\n3. MÚLTIPLES MÉTODOS DE DETECCIÓN DE OUTLIERS")
print("-" * 50)

def detect_outliers_iqr(df, columns, factor=1.5):
    """Detectar outliers usando método IQR"""
    outliers_mask = pd.DataFrame(index=df.index, columns=columns, data=False)
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outliers_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
    
    return outliers_mask

def detect_outliers_zscore(df, columns, threshold=3):
    """Detectar outliers usando Z-score"""
    outliers_mask = pd.DataFrame(index=df.index, columns=columns, data=False)
    
    for col in columns:
        z_scores = np.abs(stats.zscore(df[col]))
        outliers_mask[col] = z_scores > threshold
    
    return outliers_mask

def detect_outliers_percentile(df, columns, low_percentile=1, high_percentile=99):
    """Detectar outliers usando percentiles"""
    outliers_mask = pd.DataFrame(index=df.index, columns=columns, data=False)
    
    for col in columns:
        low_bound = df[col].quantile(low_percentile / 100)
        high_bound = df[col].quantile(high_percentile / 100)
        outliers_mask[col] = (df[col] < low_bound) | (df[col] > high_bound)
    
    return outliers_mask

# Aplicar diferentes métodos
print("Detectando outliers con diferentes métodos...")

# Método IQR
outliers_iqr = detect_outliers_iqr(df, df.columns, factor=1.5)
outliers_iqr_count = outliers_iqr.sum().sum()

# Método Z-score
outliers_zscore = detect_outliers_zscore(df, df.columns, threshold=3)
outliers_zscore_count = outliers_zscore.sum().sum()

# Método Percentiles
outliers_percentile = detect_outliers_percentile(df, df.columns, low_percentile=1, high_percentile=99)
outliers_percentile_count = outliers_percentile.sum().sum()

print(f"Outliers detectados:")
print(f"  • Método IQR (factor=1.5): {outliers_iqr_count} puntos")
print(f"  • Método Z-score (threshold=3): {outliers_zscore_count} puntos")
print(f"  • Método Percentiles (1-99%): {outliers_percentile_count} puntos")

#----------------------------------------------- 4. ESTRATEGIAS DE MANEJO DE OUTLIERS

print("\n4. ESTRATEGIAS DE MANEJO DE OUTLIERS")
print("-" * 50)

# Estrategia 1: Eliminación conservadora (IQR con factor=2.0)
print("\nEstrategia 1: Eliminación conservadora (IQR factor=2.0)")
outliers_conservative = detect_outliers_iqr(df, df.columns, factor=2.0)
df_conservative = df[~outliers_conservative.any(axis=1)].copy()
print(f"  Clientes restantes: {len(df_conservative)} ({len(df_conservative)/len(df)*100:.1f}%)")

# Estrategia 2: Eliminación moderada (Percentiles 2-98)
print("\nEstrategia 2: Eliminación moderada (Percentiles 2-98)")
outliers_moderate = detect_outliers_percentile(df, df.columns, low_percentile=2, high_percentile=98)
df_moderate = df[~outliers_moderate.any(axis=1)].copy()
print(f"  Clientes restantes: {len(df_moderate)} ({len(df_moderate)/len(df)*100:.1f}%)")

# Estrategia 3: Eliminación agresiva (Percentiles 5-95)
print("\nEstrategia 3: Eliminación agresiva (Percentiles 5-95)")
outliers_aggressive = detect_outliers_percentile(df, df.columns, low_percentile=5, high_percentile=95)
df_aggressive = df[~outliers_aggressive.any(axis=1)].copy()
print(f"  Clientes restantes: {len(df_aggressive)} ({len(df_aggressive)/len(df)*100:.1f}%)")

#----------------------------------------------- 5. COMPARACIÓN DE ESTRATEGIAS CON K-MEANS

print("\n5. COMPARACIÓN DE ESTRATEGIAS CON K-MEANS")
print("-" * 50)

def evaluate_clustering(df_clean, strategy_name):
    """Evaluar clustering con diferentes métricas"""
    if len(df_clean) < 10:  # Muy pocos datos
        return None
    
    # Escalar datos
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_clean)
    
    # Aplicar K-means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)
    
    # Calcular métricas
    distribucion = pd.Series(clusters).value_counts().sort_index()
    min_cluster = distribucion.min()
    max_cluster = distribucion.max()
    ratio_balance = min_cluster / max_cluster
    
    # Silhouette score
    silhouette = silhouette_score(df_scaled, clusters)
    
    # Inertia (menor es mejor)
    inertia = kmeans.inertia_
    
    return {
        'strategy': strategy_name,
        'n_clients': len(df_clean),
        'ratio_balance': ratio_balance,
        'silhouette': silhouette,
        'inertia': inertia,
        'distribution': distribucion.to_dict()
    }

# Evaluar todas las estrategias
estrategias = [
    (df_conservative, "Conservadora (IQR 2.0)"),
    (df_moderate, "Moderada (Percentiles 2-98)"),
    (df_aggressive, "Agresiva (Percentiles 5-95)")
]

resultados = []
for df_clean, nombre in estrategias:
    resultado = evaluate_clustering(df_clean, nombre)
    if resultado:
        resultados.append(resultado)

# Mostrar resultados
print("Resultados de clustering por estrategia:")
for res in resultados:
    print(f"\n{res['strategy']}:")
    print(f"  • Clientes: {res['n_clients']}")
    print(f"  • Ratio de balance: {res['ratio_balance']:.3f}")
    print(f"  • Silhouette score: {res['silhouette']:.3f}")
    print(f"  • Inertia: {res['inertia']:.2f}")
    print(f"  • Distribución: {res['distribution']}")

#----------------------------------------------- 6. SELECCIÓN DE LA MEJOR ESTRATEGIA

print("\n6. SELECCIÓN DE LA MEJOR ESTRATEGIA")
print("-" * 50)

# Crear DataFrame de resultados para comparación
df_resultados = pd.DataFrame(resultados)

# Normalizar métricas para comparación
df_resultados['balance_norm'] = (df_resultados['ratio_balance'] - df_resultados['ratio_balance'].min()) / (df_resultados['ratio_balance'].max() - df_resultados['ratio_balance'].min())
df_resultados['silhouette_norm'] = (df_resultados['silhouette'] - df_resultados['silhouette'].min()) / (df_resultados['silhouette'].max() - df_resultados['silhouette'].min())
df_resultados['inertia_norm'] = 1 - (df_resultados['inertia'] - df_resultados['inertia'].min()) / (df_resultados['inertia'].max() - df_resultados['inertia'].min())

# Calcular score compuesto
df_resultados['score_compuesto'] = (
    df_resultados['balance_norm'] * 0.4 +  # Balance es importante
    df_resultados['silhouette_norm'] * 0.4 +  # Calidad de clusters
    df_resultados['inertia_norm'] * 0.2  # Compacidad
)

# Encontrar la mejor estrategia
mejor_idx = df_resultados['score_compuesto'].idxmax()
mejor_estrategia = df_resultados.loc[mejor_idx]

print(f"🏆 MEJOR ESTRATEGIA: {mejor_estrategia['strategy']}")
print(f"  • Score compuesto: {mejor_estrategia['score_compuesto']:.3f}")
print(f"  • Ratio de balance: {mejor_estrategia['ratio_balance']:.3f}")
print(f"  • Silhouette score: {mejor_estrategia['silhouette']:.3f}")
print(f"  • Clientes: {mejor_estrategia['n_clients']}")

# Seleccionar el dataset correspondiente
if "Conservadora" in mejor_estrategia['strategy']:
    df_final = df_conservative
elif "Moderada" in mejor_estrategia['strategy']:
    df_final = df_moderate
else:
    df_final = df_aggressive

#----------------------------------------------- 7. CLUSTERING FINAL CON LA MEJOR ESTRATEGIA

print("\n7. CLUSTERING FINAL CON LA MEJOR ESTRATEGIA")
print("-" * 50)

# Aplicar clustering final
scaler = StandardScaler()
df_scaled_final = scaler.fit_transform(df_final)
df_scaled_final = pd.DataFrame(df_scaled_final, columns=df_final.columns, index=df_final.index)

kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
df_final['Cluster'] = kmeans_final.fit_predict(df_scaled_final)

# Análisis de distribución final
print("Distribución final de clusters:")
distribucion_final = df_final['Cluster'].value_counts().sort_index()
for cluster, count in distribucion_final.items():
    porcentaje = count / len(df_final) * 100
    print(f"  Cluster {cluster}: {count} clientes ({porcentaje:.1f}%)")

# Caracterización de clusters
print("\nCaracterización de clusters:")
for cluster in sorted(df_final['Cluster'].unique()):
    cluster_data = df_final[df_final['Cluster'] == cluster]
    
    print(f"\nCluster {cluster} ({len(cluster_data)} clientes):")
    print(f"  Frecuencia: {cluster_data['Frecuencia'].mean():.1f} ± {cluster_data['Frecuencia'].std():.1f}")
    print(f"  Transaccionalidad: ${cluster_data['Transaccionalidad'].mean():,.0f} ± ${cluster_data['Transaccionalidad'].std():,.0f}")
    print(f"  Ingresos: ${cluster_data['Ingresos'].mean():,.0f} ± ${cluster_data['Ingresos'].std():,.0f}")

#----------------------------------------------- 8. VISUALIZACIONES MEJORADAS

print("\n8. GENERANDO VISUALIZACIONES MEJORADAS")
print("-" * 50)

# 1. Comparación de estrategias
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Distribución de clusters final
axes[0,0].bar(distribucion_final.index, distribucion_final.values, color=['skyblue', 'lightgreen', 'orange'])
axes[0,0].set_title('Distribución Final de Clusters')
axes[0,0].set_xlabel('Cluster')
axes[0,0].set_ylabel('Número de Clientes')
for i, v in enumerate(distribucion_final.values):
    axes[0,0].text(i, v + max(distribucion_final.values)*0.01, str(v), ha='center', va='bottom', fontweight='bold')

# Scatter plot Frecuencia vs Transaccionalidad
scatter = axes[0,1].scatter(df_final['Frecuencia'], df_final['Transaccionalidad'], 
                           c=df_final['Cluster'], cmap='viridis', alpha=0.6, s=50)
axes[0,1].set_xlabel('Frecuencia')
axes[0,1].set_ylabel('Transaccionalidad')
axes[0,1].set_title('Clusters - Frecuencia vs Transaccionalidad')
plt.colorbar(scatter, ax=axes[0,1], label='Cluster')

# Scatter plot Frecuencia vs Ingresos
scatter2 = axes[1,0].scatter(df_final['Frecuencia'], df_final['Ingresos'], 
                            c=df_final['Cluster'], cmap='viridis', alpha=0.6, s=50)
axes[1,0].set_xlabel('Frecuencia')
axes[1,0].set_ylabel('Ingresos')
axes[1,0].set_title('Clusters - Frecuencia vs Ingresos')
plt.colorbar(scatter2, ax=axes[1,0], label='Cluster')

# Comparación de estrategias
estrategias_nombres = [res['strategy'] for res in resultados]
balance_scores = [res['ratio_balance'] for res in resultados]
silhouette_scores = [res['silhouette'] for res in resultados]

x = np.arange(len(estrategias_nombres))
width = 0.35

axes[1,1].bar(x - width/2, balance_scores, width, label='Ratio de Balance', alpha=0.8)
axes[1,1].bar(x + width/2, silhouette_scores, width, label='Silhouette Score', alpha=0.8)
axes[1,1].set_xlabel('Estrategia')
axes[1,1].set_ylabel('Score')
axes[1,1].set_title('Comparación de Estrategias')
axes[1,1].set_xticks(x)
axes[1,1].set_xticklabels([s.split('(')[0].strip() for s in estrategias_nombres], rotation=45)
axes[1,1].legend()

plt.tight_layout()
plt.savefig('kmeans_mejorado_comparacion.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Boxplots por cluster
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, col in enumerate(['Frecuencia', 'Transaccionalidad', 'Ingresos']):
    sns.boxplot(x='Cluster', y=col, data=df_final, ax=axes[i])
    axes[i].set_title(f'{col} por Cluster')
    axes[i].set_xlabel('Cluster')

plt.tight_layout()
plt.savefig('kmeans_mejorado_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()

#----------------------------------------------- 9. GUARDAR RESULTADOS FINALES

print("\n9. GUARDANDO RESULTADOS FINALES")
print("-" * 50)

# Guardar dataset final
df_final.to_csv('clientes_kmeans_mejorado.csv')

# Guardar comparación de estrategias
df_resultados.to_csv('comparacion_estrategias_outliers.csv', index=False)

print(" Archivos guardados:")
print("  • clientes_kmeans_mejorado.csv - Dataset final con clusters")
print("  • comparacion_estrategias_outliers.csv - Comparación de estrategias")
print("  • kmeans_mejorado_comparacion.png - Visualización comparativa")
print("  • kmeans_mejorado_boxplots.png - Boxplots por cluster")

#----------------------------------------------- 10. RESUMEN FINAL

print("\n10. RESUMEN FINAL")
print("-" * 50)

print(f" ESTRATEGIA SELECCIONADA: {mejor_estrategia['strategy']}")
print(f" RESULTADOS FINALES:")
print(f"  • Clientes analizados: {len(df_final)} de {len(df)} originales")
print(f"  • Ratio de balance: {mejor_estrategia['ratio_balance']:.3f}")
print(f"  • Silhouette score: {mejor_estrategia['silhouette']:.3f}")
print(f"  • Calidad de clusters: {'Excelente' if mejor_estrategia['silhouette'] > 0.5 else 'Buena' if mejor_estrategia['silhouette'] > 0.3 else 'Aceptable'}")

print(f"\n VENTAJAS DE LA MEJORA:")
print(f"  • Múltiples métodos de detección de outliers")
print(f"  • Comparación sistemática de estrategias")
print(f"  • Métricas objetivas para selección")
print(f"  • Clusters más balanceados y de mejor calidad")

print("\n" + "=" * 70)
print("ANÁLISIS COMPLETADO")
print("=" * 70) 