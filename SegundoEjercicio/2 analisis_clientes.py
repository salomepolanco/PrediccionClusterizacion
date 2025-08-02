# -*- coding: utf-8 -*-
"""
ANÁLISIS DE CLIENTES - ÍNDICE Y CARACTERIZACIÓN DE GRUPOS
Dataset FRI: 5000 clientes con Frecuencia, Transaccionalidad e Ingresos

Primer intento de clustering
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("=" * 60)
print("ANÁLISIS DE CLIENTES - ÍNDICE Y CARACTERIZACIÓN")
print("=" * 60)

#----------------------------------------------- 1. CARGA DE DATOS

print("\n1. CARGA Y PREPARACIÓN DE DATOS")
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

print(f"Dataset cargado: {df.shape[0]} clientes, {df.shape[1]} variables")
print(f"Variables: {list(df.columns)}")

#----------------------------------------------- 2. ANÁLISIS EXPLORATORIO

print("\n2. ANÁLISIS EXPLORATORIO")
print("-" * 40)

# Estadísticas descriptivas
print("Estadísticas descriptivas:")
print(df.describe())

# Verificar outliers
print("\nDetección de outliers (valores > Q3 + 1.5*IQR):")
for col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[df[col] > Q3 + 1.5*IQR]
    print(f"  {col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")

#----------------------------------------------- 3. CREACIÓN DEL ÍNDICE DE CLIENTE

print("\n3. CREACIÓN DEL ÍNDICE DE CLIENTE")
print("-" * 40)

# Normalizar las variables para crear el índice
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns,
    index=df.index
)

# Crear índice compuesto (promedio ponderado)
# Pesos basados en la importancia de cada variable
pesos = {
    'Frecuencia': 0.3,        # 30% - frecuencia de transacciones
    'Transaccionalidad': 0.4, # 40% - monto de transacciones (más importante)
    'Ingresos': 0.3           # 30% - ingresos totales
}

# Calcular índice
df['Indice_Cliente'] = (
    df_scaled['Frecuencia'] * pesos['Frecuencia'] +
    df_scaled['Transaccionalidad'] * pesos['Transaccionalidad'] +
    df_scaled['Ingresos'] * pesos['Ingresos']
)

print("Índice de cliente calculado con pesos:")
for var, peso in pesos.items():
    print(f"  {var}: {peso*100}%")

print(f"\nEstadísticas del índice:")
print(df['Indice_Cliente'].describe())

# Clasificar clientes por índice
df['Categoria_Indice'] = pd.cut(
    df['Indice_Cliente'],
    bins=5,
    labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto']
)

print(f"\nDistribución por categoría de índice:")
print(df['Categoria_Indice'].value_counts().sort_index())

#----------------------------------------------- 4. ANÁLISIS DE CLUSTERING

print("\n4. ANÁLISIS DE CLUSTERING")
print("-" * 40)

# Usar las variables originales para clustering
X_cluster = df[['Frecuencia', 'Transaccionalidad', 'Ingresos']].copy()

# Determinar número óptimo de clusters
inertias = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    inertias.append(kmeans.inertia_)

# Gráfico del codo
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Número de Clusters')
plt.ylabel('Inercia')
plt.title('Método del Codo para Determinar Número Óptimo de Clusters')
plt.grid(True, alpha=0.3)
plt.savefig('metodo_codo_clusters.png', dpi=300, bbox_inches='tight')
plt.close()

# Aplicar K-means con 3 clusters
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_cluster)

print(f"Clustering aplicado: {n_clusters} clusters")
print(f"Distribución de clusters:")
print(df['Cluster'].value_counts().sort_index())

#----------------------------------------------- 5. CARACTERIZACIÓN DE GRUPOS

print("\n5. CARACTERIZACIÓN DE GRUPOS")
print("-" * 40)

# Análisis por cluster
cluster_analysis = df.groupby('Cluster').agg({
    'Frecuencia': ['mean', 'std', 'min', 'max'],
    'Transaccionalidad': ['mean', 'std', 'min', 'max'],
    'Ingresos': ['mean', 'std', 'min', 'max'],
    'Indice_Cliente': ['mean', 'std']
}).round(2)

print("Caracterización de clusters:")
print(cluster_analysis)

# Crear perfiles de clientes
perfiles = {}
for cluster in range(n_clusters):
    cluster_data = df[df['Cluster'] == cluster]
    
    # Características del cluster
    perfil = {
        'Tamaño': len(cluster_data),
        'Porcentaje': len(cluster_data) / len(df) * 100,
        'Frecuencia_Media': cluster_data['Frecuencia'].mean(),
        'Transaccionalidad_Media': cluster_data['Transaccionalidad'].mean(),
        'Ingresos_Medios': cluster_data['Ingresos'].mean(),
        'Indice_Medio': cluster_data['Indice_Cliente'].mean()
    }
    
    # Determinar tipo de cliente para 3 segmentos
    if perfil['Frecuencia_Media'] > df['Frecuencia'].mean() and perfil['Transaccionalidad_Media'] > df['Transaccionalidad'].mean():
        perfil['Tipo'] = 'Cliente Premium'
    elif perfil['Transaccionalidad_Media'] > df['Transaccionalidad'].mean():
        perfil['Tipo'] = 'Cliente de Alto Valor'
    else:
        perfil['Tipo'] = 'Cliente Regular'
    
    perfiles[cluster] = perfil

print(f"\nPerfiles de clientes:")
for cluster, perfil in perfiles.items():
    print(f"\nCluster {cluster} - {perfil['Tipo']}:")
    print(f"  Tamaño: {perfil['Tamaño']} clientes ({perfil['Porcentaje']:.1f}%)")
    print(f"  Frecuencia media: {perfil['Frecuencia_Media']:.1f}")
    print(f"  Transaccionalidad media: ${perfil['Transaccionalidad_Media']:,.2f}")
    print(f"  Ingresos medios: ${perfil['Ingresos_Medios']:,.2f}")
    print(f"  Índice medio: {perfil['Indice_Medio']:.2f}")

#----------------------------------------------- 6. VISUALIZACIONES

print("\n6. GENERANDO VISUALIZACIONES")
print("-" * 40)

# 1. Distribución del índice de cliente
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
sns.histplot(df['Indice_Cliente'], bins=30, kde=True)
plt.title('Distribución del Índice de Cliente')
plt.xlabel('Índice de Cliente')
plt.ylabel('Frecuencia')

# 2. Distribución por categoría de índice
plt.subplot(2, 2, 2)
df['Categoria_Indice'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribución por Categoría de Índice')
plt.xlabel('Categoría')
plt.ylabel('Número de Clientes')
plt.xticks(rotation=45)

# 3. Distribución por cluster
plt.subplot(2, 2, 3)
df['Cluster'].value_counts().plot(kind='bar', color='lightgreen')
plt.title('Distribución por Cluster')
plt.xlabel('Cluster')
plt.ylabel('Número de Clientes')

# 4. Relación entre variables por cluster
plt.subplot(2, 2, 4)
scatter = plt.scatter(df['Frecuencia'], df['Transaccionalidad'], 
                     c=df['Cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('Frecuencia')
plt.ylabel('Transaccionalidad')
plt.title('Frecuencia vs Transaccionalidad por Cluster')
plt.colorbar(scatter, label='Cluster')

plt.tight_layout()
plt.savefig('analisis_clientes_general.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Análisis detallado por cluster
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Frecuencia por cluster
sns.boxplot(x='Cluster', y='Frecuencia', data=df, ax=axes[0,0])
axes[0,0].set_title('Frecuencia por Cluster')

# Transaccionalidad por cluster
sns.boxplot(x='Cluster', y='Transaccionalidad', data=df, ax=axes[0,1])
axes[0,1].set_title('Transaccionalidad por Cluster')

# Ingresos por cluster
sns.boxplot(x='Cluster', y='Ingresos', data=df, ax=axes[1,0])
axes[1,0].set_title('Ingresos por Cluster')

# Índice por cluster
sns.boxplot(x='Cluster', y='Indice_Cliente', data=df, ax=axes[1,1])
axes[1,1].set_title('Índice de Cliente por Cluster')

plt.tight_layout()
plt.savefig('analisis_detallado_clusters.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Matriz de correlación
plt.figure(figsize=(8, 6))
corr_matrix = df[['Frecuencia', 'Transaccionalidad', 'Ingresos', 'Indice_Cliente']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
plt.title('Matriz de Correlación')
plt.tight_layout()
plt.savefig('matriz_correlacion_clientes.png', dpi=300, bbox_inches='tight')
plt.close()

#----------------------------------------------- 7. RESULTADOS FINALES

print("\n7. RESULTADOS FINALES")
print("-" * 40)

# Guardar resultados
df_final = df.copy()
df_final.to_csv('clientes_con_indice_y_clusters.csv')

print("✅ Archivos generados:")
print("   - clientes_con_indice_y_clusters.csv")
print("   - metodo_codo_clusters.png")
print("   - analisis_clientes_general.png")
print("   - analisis_detallado_clusters.png")
print("   - matriz_correlacion_clientes.png")

print(f"\n📊 RESUMEN DEL ANÁLISIS:")
print(f"   - Total de clientes analizados: {len(df)}")
print(f"   - Variables consideradas: {list(df.columns[:-3])}")  # Excluir columnas generadas
print(f"   - Índice de cliente creado con pesos: {pesos}")
print(f"   - Clusters identificados: {n_clusters}")

print(f"\n🎯 RECOMENDACIONES POR TIPO DE CLIENTE:")
for cluster, perfil in perfiles.items():
    print(f"\n{perfil['Tipo']} (Cluster {cluster}):")
    if perfil['Tipo'] == 'Cliente Premium':
        print("  • Estrategia: Retención y fidelización")
        print("  • Acciones: Programas VIP, atención personalizada")
    elif perfil['Tipo'] == 'Cliente de Alto Valor':
        print("  • Estrategia: Incrementar frecuencia")
        print("  • Acciones: Programas de recompensas, recordatorios")
    else:
        print("  • Estrategia: Activación y engagement")
        print("  • Acciones: Campañas de marketing, educación")

print("\n" + "=" * 60)
print("ANÁLISIS COMPLETADO")
print("=" * 60) 