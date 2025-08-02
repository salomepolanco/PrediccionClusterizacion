# -*- coding: utf-8 -*-
"""
RESUMEN DE SELECCIÓN DE VARIABLES
Basado en el análisis exploratorio de datos
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 9

# Cargar resultados del análisis
importance_df = pd.read_csv('importancia_variables.csv')

print("=" * 60)
print("RESUMEN DE SELECCIÓN DE VARIABLES")
print("=" * 60)

# Mostrar todas las variables ordenadas por importancia
print("\nTODAS LAS VARIABLES ORDENADAS POR IMPORTANCIA:")
print("-" * 50)
for i, (_, row) in enumerate(importance_df.iterrows(), 1):
    print(f"{i:2d}. {row['Variable']:15s} - Puntuación: {row['Composite_Score']:.3f}")

# Seleccionar variables con puntuación > 0.5 (alta importancia)
high_importance = importance_df[importance_df['Composite_Score'] > 0.5]
medium_importance = importance_df[(importance_df['Composite_Score'] > 0.2) & (importance_df['Composite_Score'] <= 0.5)]
low_importance = importance_df[importance_df['Composite_Score'] <= 0.2]

print(f"\n" + "=" * 40)
print("RECOMENDACIÓN FINAL DE VARIABLES")
print("=" * 40)

print(f"\n🔴 VARIABLES DE ALTA IMPORTANCIA (Puntuación > 0.5):")
print("-" * 50)
if len(high_importance) > 0:
    for i, (_, row) in enumerate(high_importance.iterrows(), 1):
        print(f"{i}. {row['Variable']} (Puntuación: {row['Composite_Score']:.3f})")
else:
    print("No hay variables con puntuación > 0.5")

print(f"\n🟡 VARIABLES DE MEDIA IMPORTANCIA (0.2 < Puntuación ≤ 0.5):")
print("-" * 50)
if len(medium_importance) > 0:
    for i, (_, row) in enumerate(medium_importance.iterrows(), 1):
        print(f"{i}. {row['Variable']} (Puntuación: {row['Composite_Score']:.3f})")
else:
    print("No hay variables con puntuación entre 0.2 y 0.5")

print(f"\n🟢 VARIABLES DE BAJA IMPORTANCIA (Puntuación ≤ 0.2):")
print("-" * 50)
if len(low_importance) > 0:
    for i, (_, row) in enumerate(low_importance.iterrows(), 1):
        print(f"{i}. {row['Variable']} (Puntuación: {row['Composite_Score']:.3f})")
else:
    print("No hay variables con puntuación ≤ 0.2")

# Crear visualización de las mejores variables
plt.figure(figsize=(10, 6))

# Crear barras con colores según importancia
colors = []
for score in importance_df['Composite_Score']:
    if score > 0.5:
        colors.append('red')
    elif score > 0.2:
        colors.append('orange')
    else:
        colors.append('lightblue')

bars = plt.barh(importance_df['Variable'], importance_df['Composite_Score'], color=colors)

# Añadir valores en las barras
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left', va='center', fontweight='bold')

plt.xlabel('Puntuación Compuesta de Importancia')
plt.title('Importancia de Variables para Predecir Deserción\n(Clasificadas por nivel de importancia)', fontsize=14)
plt.gca().invert_yaxis()

# Añadir líneas de referencia
plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Alta Importancia (>0.5)')
plt.axvline(x=0.2, color='orange', linestyle='--', alpha=0.7, label='Media Importancia (>0.2)')
plt.legend()

plt.tight_layout()
plt.savefig('resumen_importancia_variables.png', dpi=300, bbox_inches='tight')
plt.show()

# Recomendación específica para el modelo
print(f"\n" + "=" * 40)
print("RECOMENDACIÓN PARA EL MODELO")
print("=" * 40)

# Variables recomendadas (alta + media importancia)
recommended_vars = importance_df[importance_df['Composite_Score'] > 0.2]['Variable'].tolist()

print(f"\n✅ VARIABLES RECOMENDADAS PARA INCLUIR EN EL MODELO:")
print("-" * 50)
for i, var in enumerate(recommended_vars, 1):
    score = importance_df[importance_df['Variable'] == var]['Composite_Score'].iloc[0]
    print(f"{i}. {var} (Puntuación: {score:.3f})")

print(f"\n📊 RESUMEN:")
print(f"   - Total de variables disponibles: {len(importance_df)}")
print(f"   - Variables recomendadas: {len(recommended_vars)}")
print(f"   - Variables excluidas: {len(importance_df) - len(recommended_vars)}")

# Variables a excluir
excluded_vars = importance_df[importance_df['Composite_Score'] <= 0.2]['Variable'].tolist()
if excluded_vars:
    print(f"\n❌ VARIABLES RECOMENDADAS PARA EXCLUIR:")
    print("-" * 50)
    for var in excluded_vars:
        score = importance_df[importance_df['Variable'] == var]['Composite_Score'].iloc[0]
        print(f"   - {var} (Puntuación: {score:.3f})")

print(f"\n" + "=" * 60)
print("ANÁLISIS COMPLETADO")
print("=" * 60) 