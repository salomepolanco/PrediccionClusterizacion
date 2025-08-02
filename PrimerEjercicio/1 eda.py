# -*- coding: utf-8 -*-
"""
ANÁLISIS EXPLORATORIO DE DATOS Y SELECCIÓN DE VARIABLES
Bank Churn Modelling Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency, pearsonr
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 9

#----------------------------------------------- 1. CARGA Y LIMPIEZA DE DATOS

print("=" * 60)
print("ANÁLISIS EXPLORATORIO DE DATOS - BANK CHURN MODELLING")
print("=" * 60)

# Cargar datos
df = pd.read_csv("Bank_churn_modelling.csv", sep=";", encoding="utf-8")

# Limpiar nombres de columnas
df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")

# Renombrar variable objetivo
df.rename(columns={'Exited': 'Desercion'}, inplace=True)

# Eliminar columnas irrelevantes
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"Variables: {list(df.columns)}")

#----------------------------------------------- 2. ANÁLISIS EXPLORATORIO BÁSICO

print("\n" + "=" * 40)
print("ANÁLISIS EXPLORATORIO BÁSICO")
print("=" * 40)

# Información general del dataset
print("\n1. INFORMACIÓN GENERAL:")
print(df.info())

# Estadísticas descriptivas
print("\n2. ESTADÍSTICAS DESCRIPTIVAS:")
print(df.describe())

# Verificar valores faltantes
print("\n3. VALORES FALTANTES:")
missing_data = df.isnull().sum()
if missing_data.sum() == 0:
    print("✓ No hay valores faltantes en el dataset")
else:
    print(missing_data[missing_data > 0])

# Distribución de la variable objetivo
print("\n4. DISTRIBUCIÓN DE LA VARIABLE OBJETIVO:")
target_dist = df['Desercion'].value_counts()
print(f"Deserción: {target_dist[1]} ({target_dist[1]/len(df)*100:.1f}%)")
print(f"No Deserción: {target_dist[0]} ({target_dist[0]/len(df)*100:.1f}%)")

#----------------------------------------------- 3. ANÁLISIS DE VARIABLES NUMÉRICAS

print("\n" + "=" * 40)
print("ANÁLISIS DE VARIABLES NUMÉRICAS")
print("=" * 40)

# Definir variables numéricas
numeric_vars = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

# Crear figura para múltiples subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, var in enumerate(numeric_vars):
    # Boxplot
    sns.boxplot(x='Desercion', y=var, data=df, ax=axes[i])
    axes[i].set_title(f'{var} vs Deserción')
    axes[i].set_xlabel('Deserción (0=No, 1=Sí)')
    
    # Estadísticas
    churned = df[df['Desercion'] == 1][var]
    not_churned = df[df['Desercion'] == 0][var]
    
    # Prueba t
    t_stat, p_val = ttest_ind(churned, not_churned)
    
    # Efecto tamaño (Cohen's d)
    pooled_std = np.sqrt(((len(churned) - 1) * churned.var() + (len(not_churned) - 1) * not_churned.var()) / (len(churned) + len(not_churned) - 2))
    cohens_d = (churned.mean() - not_churned.mean()) / pooled_std
    
    print(f"\n{var}:")
    print(f"  - p-valor: {p_val:.4f}")
    print(f"  - Cohen's d: {cohens_d:.3f}")
    print(f"  - Media (Desertores): {churned.mean():.2f}")
    print(f"  - Media (No Desertores): {not_churned.mean():.2f}")

plt.tight_layout()
plt.savefig('analisis_variables_numericas.png', dpi=300, bbox_inches='tight')
plt.close()

#----------------------------------------------- 4. ANÁLISIS DE VARIABLES CATEGÓRICAS

print("\n" + "=" * 40)
print("ANÁLISIS DE VARIABLES CATEGÓRICAS")
print("=" * 40)

# Definir variables categóricas
categorical_vars = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']

# Crear figura para múltiples subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, var in enumerate(categorical_vars):
    # Gráfico de barras
    contingency_table = pd.crosstab(df[var], df['Desercion'], normalize='index') * 100
    contingency_table.plot(kind='bar', ax=axes[i])
    axes[i].set_title(f'Tasa de Deserción por {var}')
    axes[i].set_ylabel('Porcentaje de Deserción (%)')
    axes[i].legend(['No Desertó', 'Desertó'])
    axes[i].tick_params(axis='x', rotation=45)
    
    # Prueba chi-cuadrado
    chi2, p_val, _, _ = chi2_contingency(pd.crosstab(df[var], df['Desercion']))
    
    # Cramer's V (medida de asociación)
    n = len(df)
    min_dim = min(len(df[var].unique()), 2) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    
    print(f"\n{var}:")
    print(f"  - p-valor: {p_val:.4f}")
    print(f"  - Cramer's V: {cramers_v:.3f}")
    print(f"  - Tasa de deserción por categoría:")
    for category in df[var].unique():
        rate = df[(df[var] == category) & (df['Desercion'] == 1)].shape[0] / df[df[var] == category].shape[0] * 100
        print(f"    {category}: {rate:.1f}%")

plt.tight_layout()
plt.savefig('analisis_variables_categoricas.png', dpi=300, bbox_inches='tight')
plt.close()

#----------------------------------------------- 5. ANÁLISIS DE CORRELACIONES

print("\n" + "=" * 40)
print("ANÁLISIS DE CORRELACIONES")
print("=" * 40)

# Preparar datos para correlación
df_corr = df.copy()
le = LabelEncoder()

# Codificar variables categóricas para correlación
for var in categorical_vars:
    df_corr[var] = le.fit_transform(df_corr[var])

# Matriz de correlación
correlation_matrix = df_corr.corr()

# Visualizar matriz de correlación
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Matriz de Correlación de Variables')
plt.tight_layout()
plt.savefig('matriz_correlacion.png', dpi=300, bbox_inches='tight')
plt.close()

# Correlaciones con la variable objetivo
target_correlations = correlation_matrix['Desercion'].abs().sort_values(ascending=False)
print("\nCorrelaciones con Deserción (ordenadas por importancia):")
for var, corr in target_correlations.items():
    if var != 'Desercion':
        print(f"  {var}: {corr:.3f}")

#----------------------------------------------- 6. SELECCIÓN DE VARIABLES

print("\n" + "=" * 40)
print("SELECCIÓN DE VARIABLES")
print("=" * 40)

# Preparar datos para selección de variables
X = df_corr.drop('Desercion', axis=1)
y = df_corr['Desercion']

# 6.1 Selección basada en ANOVA F-test
print("\n1. SELECCIÓN BASADA EN ANOVA F-TEST:")
f_selector = SelectKBest(score_func=f_classif, k='all')
f_scores = f_selector.fit(X, y)
f_scores_df = pd.DataFrame({
    'Variable': X.columns,
    'F-Score': f_scores.scores_,
    'P-Value': f_scores.pvalues_
}).sort_values('F-Score', ascending=False)

print(f_scores_df)

# 6.2 Selección basada en Información Mutua
print("\n2. SELECCIÓN BASADA EN INFORMACIÓN MUTUA:")
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_scores_df = pd.DataFrame({
    'Variable': X.columns,
    'MI-Score': mi_scores
}).sort_values('MI-Score', ascending=False)

print(mi_scores_df)

# 6.3 Selección basada en Random Forest
print("\n3. SELECCIÓN BASADA EN RANDOM FOREST:")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
rf_importance = pd.DataFrame({
    'Variable': X.columns,
    'RF-Importance': rf.feature_importances_
}).sort_values('RF-Importance', ascending=False)

print(rf_importance)

#----------------------------------------------- 7. PUNTUACIÓN COMPUESTA DE IMPORTANCIA

print("\n" + "=" * 40)
print("PUNTUACIÓN COMPUESTA DE IMPORTANCIA")
print("=" * 40)

# Normalizar puntuaciones
f_scores_norm = (f_scores_df['F-Score'] - f_scores_df['F-Score'].min()) / (f_scores_df['F-Score'].max() - f_scores_df['F-Score'].min())
mi_scores_norm = (mi_scores_df['MI-Score'] - mi_scores_df['MI-Score'].min()) / (mi_scores_df['MI-Score'].max() - mi_scores_df['MI-Score'].min())
rf_importance_norm = (rf_importance['RF-Importance'] - rf_importance['RF-Importance'].min()) / (rf_importance['RF-Importance'].max() - rf_importance['RF-Importance'].min())

# Crear DataFrame con puntuaciones compuestas
importance_df = pd.DataFrame({
    'Variable': X.columns,
    'F_Score_Norm': f_scores_norm.values,
    'MI_Score_Norm': mi_scores_norm.values,
    'RF_Importance_Norm': rf_importance_norm.values
})

# Calcular puntuación compuesta (promedio ponderado)
importance_df['Composite_Score'] = (
    importance_df['F_Score_Norm'] * 0.3 +
    importance_df['MI_Score_Norm'] * 0.3 +
    importance_df['RF_Importance_Norm'] * 0.4
)

importance_df = importance_df.sort_values('Composite_Score', ascending=False)

print("Puntuación compuesta de importancia de variables:")
print(importance_df)

# Visualizar importancia de variables
plt.figure(figsize=(10, 6))
bars = plt.barh(importance_df['Variable'], importance_df['Composite_Score'])
plt.xlabel('Puntuación Compuesta de Importancia')
plt.title('Importancia de Variables para Predecir Deserción')
plt.gca().invert_yaxis()

# Añadir valores en las barras
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left', va='center')

plt.tight_layout()
plt.savefig('importancia_variables.png', dpi=300, bbox_inches='tight')
plt.close()

#----------------------------------------------- 8. RECOMENDACIÓN FINAL

print("\n" + "=" * 40)
print("RECOMENDACIÓN FINAL")
print("=" * 40)

# Seleccionar variables con puntuación compuesta > 0.5
selected_vars = importance_df[importance_df['Composite_Score'] > 0.5]['Variable'].tolist()

print(f"Variables recomendadas para el modelo (puntuación > 0.5):")
for i, var in enumerate(selected_vars, 1):
    score = importance_df[importance_df['Variable'] == var]['Composite_Score'].iloc[0]
    print(f"{i}. {var} (Puntuación: {score:.3f})")

print(f"\nTotal de variables seleccionadas: {len(selected_vars)}")

# Guardar resultados
importance_df.to_csv('importancia_variables.csv', index=False)
print("\nResultados guardados en 'importancia_variables.csv'")

# Crear gráfico final de importancia
plt.figure(figsize=(8, 5))
top_vars = importance_df.head(10)
bars = plt.barh(top_vars['Variable'], top_vars['Composite_Score'])
plt.xlabel('Puntuación Compuesta de Importancia')
plt.title('Top 10 Variables Más Importantes para Predecir Deserción')
plt.gca().invert_yaxis()

for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left', va='center')

plt.tight_layout()
plt.savefig('top10_variables.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print("ANÁLISIS COMPLETADO")
print("=" * 60)