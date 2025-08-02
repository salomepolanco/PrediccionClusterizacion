# -*- coding: utf-8 -*-
"""
RESUMEN SIMPLE DE LOS MODELOS DE PREDICCIÓN
"""

import pandas as pd

print("=" * 60)
print("RESUMEN DE MODELOS DE PREDICCIÓN DE DESERCIÓN")
print("=" * 60)

# Cargar resultados
resultados = pd.read_csv('resultados_comparacion.csv')

print("\n📊 RESULTADOS PRINCIPALES:")
print("-" * 40)

for _, row in resultados.iterrows():
    print(f"\n{row['Modelo']}:")
    print(f"  • Precisión general: {row['Accuracy']:.1%}")
    print(f"  • Capacidad de predicción (AUC-ROC): {row['AUC_ROC']:.1%}")
    print(f"  • Precisión en desertores: {row['Precision']:.1%}")
    print(f"  • Detección de desertores: {row['Recall']:.1%}")

print("\n" + "=" * 40)
print("INTERPRETACIÓN SIMPLE")
print("=" * 40)

print("\n🎯 ¿QUÉ SIGNIFICAN ESTOS NÚMEROS?")
print("• Precisión general: De cada 100 predicciones, cuántas son correctas")
print("• AUC-ROC: Qué tan bien el modelo distingue entre desertores y no desertores")
print("• Precisión en desertores: De los que predijimos como desertores, cuántos realmente desertaron")
print("• Detección de desertores: De los que realmente desertaron, cuántos logramos identificar")

print("\n🏆 MEJOR MODELO:")
mejor_modelo = resultados.loc[resultados['AUC_ROC'].idxmax(), 'Modelo']
mejor_auc = resultados['AUC_ROC'].max()
print(f"• {mejor_modelo} con {mejor_auc:.1%} de capacidad de predicción")

print("\n💡 CONCLUSIONES:")
print("• Ambos modelos tienen rendimiento similar")
print("• El modelo logístico es ligeramente mejor en capacidad de predicción")
print("• Las 5 variables seleccionadas son efectivas para predecir deserción")
print("• El modelo puede identificar patrones importantes de deserción")

print("\n🔍 VARIABLES MÁS IMPORTANTES:")
print("• CreditScore: Puntaje crediticio del cliente")
print("• Age: Edad del cliente") 
print("• Geography: Ubicación geográfica")
print("• Tenure: Tiempo con el banco")
print("• Gender: Género del cliente")

print("\n" + "=" * 60) 