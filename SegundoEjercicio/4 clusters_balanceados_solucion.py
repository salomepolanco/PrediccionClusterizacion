# -*- coding: utf-8 -*-
"""
SOLUCIÓN PARA CLUSTERS BALANCEADOS
Eliminar outliers extremos y aplicar clustering balanceado
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("=" * 60)
print("SOLUCIÓN PARA CLUSTERS BALANCEADOS")
print("=" * 60)

#----------------------------------------------- 1. CARGA DE DATOS

print("\n1. CARGA DE DATOS")
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

print(f"Dataset original: {df.shape[0]} clientes")

#----------------------------------------------- 2. ELIMINACIÓN DE OUTLIERS EXTREMOS

print("\n2. ELIMINACIÓN DE OUTLIERS EXTREMOS")
print("-" * 40)

# Función para eliminar outliers extremos
def remove_extreme_outliers(df, columns, percentile_low=1, percentile_high=99):
    df_clean = df.copy()
    
    for col in columns:
        # Calcular percentiles
        low_percentile = df_clean[col].quantile(percentile_low / 100)
        high_percentile = df_clean[col].quantile(percentile_high / 100)
        
        # Filtrar outliers
        mask = (df_clean[col] >= low_percentile) & (df_clean[col] <= high_percentile)
        df_clean = df_clean[mask]
        
        print(f"  {col}: {len(df) - len(df_clean)} outliers eliminados")
    
    return df_clean

# Eliminar outliers extremos (percentiles 1-99)
df_clean = remove_extreme_outliers(df, df.columns, percentile_low=1, percentile_high=99)

print(f"Dataset después de limpieza: {len(df_clean)} clientes")
print(f"Clientes eliminados: {len(df) - len(df_clean)} ({((len(df) - len(df_clean))/len(df)*100):.1f}%)")

#----------------------------------------------- 3. ANÁLISIS DE DATOS LIMPIOS

print("\n3. ANÁLISIS DE DATOS LIMPIOS")
print("-" * 40)

print("Estadísticas descriptivas (datos limpios):")
print(df_clean.describe())

# Análisis de skewness después de limpieza
print("\nAnálisis de asimetría después de limpieza:")
for col in df_clean.columns:
    skewness = df_clean[col].skew()
    print(f"  {col}: {skewness:.2f} ({'Muy sesgado' if abs(skewness) > 1 else 'Moderadamente sesgado' if abs(skewness) > 0.5 else 'Casi normal'})")

#----------------------------------------------- 4. CLUSTERING BALANCEADO

print("\n4. CLUSTERING BALANCEADO")
print("-" * 40)

# Normalizar datos
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_clean)
df_scaled = pd.DataFrame(df_scaled, columns=df_clean.columns, index=df_clean.index)

# Aplicar K-means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_clean['Cluster'] = kmeans.fit_predict(df_scaled)

# Análisis de distribución
print("Distribución de clusters:")
distribucion = df_clean['Cluster'].value_counts().sort_index()
for cluster, count in distribucion.items():
    porcentaje = count / len(df_clean) * 100
    print(f"  Cluster {cluster}: {count} clientes ({porcentaje:.1f}%)")

# Calcular balance
min_cluster = distribucion.min()
max_cluster = distribucion.max()
ratio_balance = min_cluster / max_cluster
print(f"\nRatio de balance: {ratio_balance:.3f} ({'Bien balanceado' if ratio_balance > 0.3 else 'Desbalanceado'})")

#----------------------------------------------- 5. CARACTERIZACIÓN DE CLUSTERS

print("\n5. CARACTERIZACIÓN DE CLUSTERS")
print("-" * 40)

# Análisis por cluster
for cluster in sorted(df_clean['Cluster'].unique()):
    cluster_data = df_clean[df_clean['Cluster'] == cluster]
    
    print(f"\nCluster {cluster} ({len(cluster_data)} clientes):")
    print(f"  Frecuencia: {cluster_data['Frecuencia'].mean():.1f} ± {cluster_data['Frecuencia'].std():.1f}")
    print(f"  Transaccionalidad: ${cluster_data['Transaccionalidad'].mean():,.0f} ± ${cluster_data['Transaccionalidad'].std():,.0f}")
    print(f"  Ingresos: ${cluster_data['Ingresos'].mean():,.0f} ± ${cluster_data['Ingresos'].std():,.0f}")

# Determinar tipos de clientes
print(f"\nClasificación de tipos de clientes:")
for cluster in sorted(df_clean['Cluster'].unique()):
    cluster_data = df_clean[df_clean['Cluster'] == cluster]
    
    # Comparar con medias generales
    freq_media = df_clean['Frecuencia'].mean()
    trans_media = df_clean['Transaccionalidad'].mean()
    ing_media = df_clean['Ingresos'].mean()
    
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

#----------------------------------------------- 6. VISUALIZACIONES

print("\n6. GENERANDO VISUALIZACIONES")
print("-" * 40)

# Distribución de clusters
plt.figure(figsize=(10, 6))
bars = plt.bar(distribucion.index, distribucion.values, color=['skyblue', 'lightgreen', 'orange'])
plt.title('Distribución de Clusters Balanceados')
plt.xlabel('Cluster')
plt.ylabel('Número de Clientes')

# Añadir valores en las barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + max(distribucion.values)*0.01,
             f'{int(height)}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('clusters_balanceados_distribucion.png', dpi=300, bbox_inches='tight')
plt.close()

# Scatter plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df_clean['Frecuencia'], df_clean['Transaccionalidad'], 
                     c=df_clean['Cluster'], cmap='viridis', alpha=0.6, s=50)
plt.xlabel('Frecuencia')
plt.ylabel('Transaccionalidad')
plt.title('Clusters Balanceados - Frecuencia vs Transaccionalidad')
plt.colorbar(scatter, label='Cluster')
plt.savefig('clusters_balanceados_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# Boxplots por cluster
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, col in enumerate(['Frecuencia', 'Transaccionalidad', 'Ingresos']):
    sns.boxplot(x='Cluster', y=col, data=df_clean, ax=axes[i])
    axes[i].set_title(f'{col} por Cluster')

plt.tight_layout()
plt.savefig('clusters_balanceados_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()

#----------------------------------------------- 7. RESULTADOS FINALES

print("\n7. RESULTADOS FINALES")
print("-" * 40)

# Guardar resultados
df_final = df_clean.copy()
df_final.to_csv('clientes_clusters_balanceados_final.csv', index=False)

print("✅ Archivos generados:")
print("   - clientes_clusters_balanceados_final.csv")
print("   - clusters_balanceados_distribucion.png")
print("   - clusters_balanceados_scatter.png")
print("   - clusters_balanceados_boxplots.png")

print(f"\n📊 RESUMEN FINAL:")
print(f"   - Clientes originales: {len(df)}")
print(f"   - Clientes después de limpieza: {len(df_clean)}")
print(f"   - Clientes eliminados: {len(df) - len(df_clean)} ({((len(df) - len(df_clean))/len(df)*100):.1f}%)")
print(f"   - Clusters creados: 3")
print(f"   - Ratio de balance: {ratio_balance:.3f}")

print(f"\n🎯 ESTRATEGIAS POR CLUSTER:")
for cluster in sorted(df_clean['Cluster'].unique()):
    cluster_data = df_clean[df_clean['Cluster'] == cluster]
    freq_cluster = cluster_data['Frecuencia'].mean()
    trans_cluster = cluster_data['Transaccionalidad'].mean()
    
    if freq_cluster > df_clean['Frecuencia'].mean() and trans_cluster > df_clean['Transaccionalidad'].mean():
        estrategia = "Retención y fidelización"
        acciones = "Programas VIP, atención personalizada"
    elif trans_cluster > df_clean['Transaccionalidad'].mean():
        estrategia = "Incrementar frecuencia"
        acciones = "Programas de recompensas, recordatorios"
    elif freq_cluster > df_clean['Frecuencia'].mean():
        estrategia = "Incrementar valor promedio"
        acciones = "Cross-selling, ofertas especiales"
    else:
        estrategia = "Activación y engagement"
        acciones = "Campañas de marketing, educación"
    
    print(f"  Cluster {cluster}: {estrategia} - {acciones}")

print("\n" + "=" * 60)
print("ANÁLISIS COMPLETADO")
print("=" * 60) 