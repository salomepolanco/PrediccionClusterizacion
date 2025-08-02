# -*- coding: utf-8 -*-
"""
RESUMEN DE MEJORAS CON BALANCEO
Comparación entre modelos originales y mejorados
"""

import pandas as pd

print("=" * 70)
print("RESUMEN DE MEJORAS CON MANEJO DE DESBALANCE")
print("=" * 70)

# Cargar resultados
resultados_originales = pd.read_csv('resultados_comparacion.csv')
resultados_mejorados = pd.read_csv('resultados_mejorados.csv')

print("\n COMPARACIÓN DE MODELOS")
print("-" * 50)

print("\n MODELOS ORIGINALES (Sin manejo de desbalance):")
for _, row in resultados_originales.iterrows():
    print(f"\n{row['Modelo']}:")
    print(f"  • Accuracy: {row['Accuracy']:.1%}")
    print(f"  • Precision: {row['Precision']:.1%}")
    print(f"  • Recall: {row['Recall']:.1%}")
    print(f"  • F1-Score: {row['F1_Score']:.1%}")
    print(f"  • AUC-ROC: {row['AUC_ROC']:.1%}")

print("\n MODELOS MEJORADOS (Con Class Weights):")
for _, row in resultados_mejorados.iterrows():
    print(f"\n{row['Modelo']}:")
    print(f"  • Accuracy: {row['Accuracy']:.1%}")
    print(f"  • Precision: {row['Precision']:.1%}")
    print(f"  • Recall: {row['Recall']:.1%}")
    print(f"  • F1-Score: {row['F1_Score']:.1%}")
    print(f"  • AUC-ROC: {row['AUC_ROC']:.1%}")

print("\n" + "=" * 50)
print("ANÁLISIS DE MEJORAS")
print("=" * 50)

# Comparar Random Forest
rf_original = resultados_originales[resultados_originales['Modelo'] == 'Random Forest'].iloc[0]
rf_mejorado = resultados_mejorados[resultados_mejorados['Modelo'] == 'Random Forest (Class Weights)'].iloc[0]

print(f"\n RANDOM FOREST:")
print(f"  • F1-Score: {rf_original['F1_Score']:.1%} → {rf_mejorado['F1_Score']:.1%}")
print(f"  • Recall: {rf_original['Recall']:.1%} → {rf_mejorado['Recall']:.1%}")
print(f"  • Precision: {rf_original['Precision']:.1%} → {rf_mejorado['Precision']:.1%}")

# Comparar Logístico
logit_original = resultados_originales[resultados_originales['Modelo'] == 'Logístico'].iloc[0]
logit_mejorado = resultados_mejorados[resultados_mejorados['Modelo'] == 'Logístico (Class Weights)'].iloc[0]

print(f"\n MODELO LOGÍSTICO:")
print(f"  • F1-Score: {logit_original['F1_Score']:.1%} → {logit_mejorado['F1_Score']:.1%} (+{((logit_mejorado['F1_Score']-logit_original['F1_Score'])/logit_original['F1_Score']*100):.0f}%)")
print(f"  • Recall: {logit_original['Recall']:.1%} → {logit_mejorado['Recall']:.1%} (+{((logit_mejorado['Recall']-logit_original['Recall'])/logit_original['Recall']*100):.0f}%)")
print(f"  • Precision: {logit_original['Precision']:.1%} → {logit_mejorado['Precision']:.1%}")

print("\n" + "=" * 50)
print("CONCLUSIONES PRINCIPALES")
print("=" * 50)

print(f"\n MEJOR MODELO MEJORADO:")
mejor_modelo_idx = resultados_mejorados['F1_Score'].idxmax()
mejor_modelo = resultados_mejorados.loc[mejor_modelo_idx, 'Modelo']
mejor_f1 = resultados_mejorados.loc[mejor_modelo_idx, 'F1_Score']
mejor_recall = resultados_mejorados.loc[mejor_modelo_idx, 'Recall']

print(f"  • {mejor_modelo}")
print(f"  • F1-Score: {mejor_f1:.1%}")
print(f"  • Recall: {mejor_recall:.1%}")

print(f"\n MEJORAS OBTENIDAS:")
print(f"  • El modelo logístico mejoró dramáticamente con class weights")
print(f"  • Recall del logístico: +{((logit_mejorado['Recall']-logit_original['Recall'])/logit_original['Recall']*100):.0f}%")
print(f"  • F1-Score del logístico: +{((logit_mejorado['F1_Score']-logit_original['F1_Score'])/logit_original['F1_Score']*100):.0f}%")
print(f"  • El Random Forest mantuvo rendimiento similar")

print(f"\n IMPACTO EN EL NEGOCIO:")
print(f"  • Mejor identificación de clientes en riesgo de deserción")
print(f"  • Recall de {mejor_recall:.1%} significa identificar al {mejor_recall:.0%} de los desertores reales")
print(f"  • Más efectivo para estrategias de retención de clientes")

print(f"\n TÉCNICA APLICADA:")
print(f"  • Class Weights: Asignar mayor peso a la clase minoritaria (deserción)")
print(f"  • Peso para No Deserción: 0.63")
print(f"  • Peso para Deserción: 2.45")
print(f"  • Esto compensa el desbalance del dataset (20.4% vs 79.6%)")

print("\n" + "=" * 70) 