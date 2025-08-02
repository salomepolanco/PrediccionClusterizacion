# -*- coding: utf-8 -*-
"""
MODELOS DE PREDICCIÓN DE DESERCIÓN
Usando las 5 variables más importantes identificadas en el EDA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 9

print("=" * 60)
print("MODELOS DE PREDICCIÓN DE DESERCIÓN")
print("=" * 60)

#----------------------------------------------- 1. CARGA Y PREPARACIÓN DE DATOS

print("\n1. CARGA Y PREPARACIÓN DE DATOS")
print("-" * 40)

# Cargar datos
df = pd.read_csv("Bank_churn_modelling.csv", sep=";", encoding="utf-8")

# Limpiar nombres de columnas
df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")

# Renombrar variable objetivo
df.rename(columns={'Exited': 'Desercion'}, inplace=True)

# Eliminar columnas irrelevantes
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Variables seleccionadas del EDA
variables_seleccionadas = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure']

print(f"Variables seleccionadas: {variables_seleccionadas}")
print(f"Dataset original: {df.shape[0]} filas, {df.shape[1]} columnas")

# Preparar datos con variables seleccionadas
X = df[variables_seleccionadas].copy()
y = df['Desercion']

print(f"Dataset de modelado: {X.shape[0]} filas, {X.shape[1]} columnas")
print(f"Distribución de la variable objetivo:")
print(f"  - Deserción: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
print(f"  - No Deserción: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")

#----------------------------------------------- 2. PREPROCESAMIENTO

print("\n2. PREPROCESAMIENTO")
print("-" * 40)

# Codificar variables categóricas
le_geography = LabelEncoder()
le_gender = LabelEncoder()

X['Geography'] = le_geography.fit_transform(X['Geography'])
X['Gender'] = le_gender.fit_transform(X['Gender'])

print("Variables categóricas codificadas:")
print(f"  - Geography: {list(le_geography.classes_)} → {list(range(len(le_geography.classes_)))}")
print(f"  - Gender: {list(le_gender.classes_)} → {list(range(len(le_gender.classes_)))}")

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nDivisión train/test:")
print(f"  - Train: {X_train.shape[0]} muestras")
print(f"  - Test: {X_test.shape[0]} muestras")

#----------------------------------------------- 3. MODELO RANDOM FOREST

print("\n3. MODELO RANDOM FOREST")
print("-" * 40)

# Entrenar modelo
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predicciones
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Métricas
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

print("Métricas del modelo Random Forest:")
print(f"  - Accuracy: {accuracy_rf:.3f}")
print(f"  - Precision: {precision_rf:.3f}")
print(f"  - Recall: {recall_rf:.3f}")
print(f"  - F1-Score: {f1_rf:.3f}")
print(f"  - AUC-ROC: {auc_rf:.3f}")

# Matriz de confusión
cm_rf = confusion_matrix(y_test, y_pred_rf)
print(f"\nMatriz de confusión:")
print(cm_rf)

# Importancia de variables
importance_rf = pd.DataFrame({
    'Variable': variables_seleccionadas,
    'Importancia': rf_model.feature_importances_
}).sort_values('Importancia', ascending=False)

print(f"\nImportancia de variables (Random Forest):")
for i, row in importance_rf.iterrows():
    print(f"  - {row['Variable']}: {row['Importancia']:.3f}")

#----------------------------------------------- 4. MODELO LOGÍSTICO

print("\n4. MODELO LOGÍSTICO")
print("-" * 40)

# Entrenar modelo
logit_model = LogisticRegression(random_state=42, max_iter=1000)
logit_model.fit(X_train, y_train)

# Predicciones
y_pred_logit = logit_model.predict(X_test)
y_pred_proba_logit = logit_model.predict_proba(X_test)[:, 1]

# Métricas
accuracy_logit = accuracy_score(y_test, y_pred_logit)
precision_logit = precision_score(y_test, y_pred_logit)
recall_logit = recall_score(y_test, y_pred_logit)
f1_logit = f1_score(y_test, y_pred_logit)
auc_logit = roc_auc_score(y_test, y_pred_proba_logit)

print("Métricas del modelo Logístico:")
print(f"  - Accuracy: {accuracy_logit:.3f}")
print(f"  - Precision: {precision_logit:.3f}")
print(f"  - Recall: {recall_logit:.3f}")
print(f"  - F1-Score: {f1_logit:.3f}")
print(f"  - AUC-ROC: {auc_logit:.3f}")

# Matriz de confusión
cm_logit = confusion_matrix(y_test, y_pred_logit)
print(f"\nMatriz de confusión:")
print(cm_logit)

# Coeficientes del modelo
coef_logit = pd.DataFrame({
    'Variable': variables_seleccionadas,
    'Coeficiente': logit_model.coef_[0]
}).sort_values('Coeficiente', ascending=False)

print(f"\nCoeficientes del modelo logístico:")
for i, row in coef_logit.iterrows():
    print(f"  - {row['Variable']}: {row['Coeficiente']:.3f}")

#----------------------------------------------- 5. COMPARACIÓN DE MODELOS

print("\n5. COMPARACIÓN DE MODELOS")
print("-" * 40)

# Crear DataFrame de comparación
comparison_df = pd.DataFrame({
    'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
    'Random Forest': [accuracy_rf, precision_rf, recall_rf, f1_rf, auc_rf],
    'Logístico': [accuracy_logit, precision_logit, recall_logit, f1_logit, auc_logit]
})

print("Comparación de métricas:")
print(comparison_df.to_string(index=False, float_format='%.3f'))

# Determinar el mejor modelo
if auc_rf > auc_logit:
    mejor_modelo = "Random Forest"
    mejor_auc = auc_rf
else:
    mejor_modelo = "Logístico"
    mejor_auc = auc_logit

print(f"\n🎯 MEJOR MODELO: {mejor_modelo} (AUC-ROC: {mejor_auc:.3f})")

#----------------------------------------------- 6. VISUALIZACIONES

print("\n6. GENERANDO VISUALIZACIONES")
print("-" * 40)

# 1. Curvas ROC
plt.figure(figsize=(10, 6))

# Random Forest ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.3f})', linewidth=2)

# Logístico ROC
fpr_logit, tpr_logit, _ = roc_curve(y_test, y_pred_proba_logit)
plt.plot(fpr_logit, tpr_logit, label=f'Logístico (AUC = {auc_logit:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC - Comparación de Modelos')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('curvas_roc.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Matrices de confusión
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Random Forest
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Matriz de Confusión - Random Forest')
axes[0].set_xlabel('Predicción')
axes[0].set_ylabel('Real')

# Logístico
sns.heatmap(cm_logit, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('Matriz de Confusión - Logístico')
axes[1].set_xlabel('Predicción')
axes[1].set_ylabel('Real')

plt.tight_layout()
plt.savefig('matrices_confusion.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Importancia de variables
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Random Forest
bars1 = axes[0].barh(importance_rf['Variable'], importance_rf['Importancia'])
axes[0].set_title('Importancia de Variables - Random Forest')
axes[0].set_xlabel('Importancia')
for bar in bars1:
    width = bar.get_width()
    axes[0].text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{width:.3f}', ha='left', va='center')

# Logístico (valor absoluto de coeficientes)
coef_abs = coef_logit.copy()
coef_abs['Coeficiente'] = abs(coef_abs['Coeficiente'])
coef_abs = coef_abs.sort_values('Coeficiente', ascending=True)

bars2 = axes[1].barh(coef_abs['Variable'], coef_abs['Coeficiente'])
axes[1].set_title('Importancia de Variables - Logístico (|Coeficientes|)')
axes[1].set_xlabel('|Coeficiente|')
for bar in bars2:
    width = bar.get_width()
    axes[1].text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{width:.3f}', ha='left', va='center')

plt.tight_layout()
plt.savefig('importancia_variables_modelos.png', dpi=300, bbox_inches='tight')
plt.close()

#----------------------------------------------- 7. INTERPRETACIÓN DE RESULTADOS

print("\n7. INTERPRETACIÓN DE RESULTADOS")
print("-" * 40)

print(" ANÁLISIS DE RENDIMIENTO:")
print(f"  • Ambos modelos muestran un rendimiento similar")
print(f"  • Random Forest: AUC-ROC = {auc_rf:.3f}")
print(f"  • Modelo Logístico: AUC-ROC = {auc_logit:.3f}")
print(f"  • El {mejor_modelo} es ligeramente superior")

print("\n INTERPRETACIÓN DE MÉTRICAS:")
print(f"  • Accuracy: Porcentaje de predicciones correctas")
print(f"  • Precision: De los que predijimos como desertores, cuántos realmente desertaron")
print(f"  • Recall: De los que realmente desertaron, cuántos logramos identificar")
print(f"  • F1-Score: Balance entre precision y recall")
print(f"  • AUC-ROC: Capacidad del modelo para distinguir entre clases")

print("\n ANÁLISIS DE VARIABLES:")
print("  Variables más importantes según Random Forest:")
for i, row in importance_rf.iterrows():
    print(f"    - {row['Variable']}: {row['Importancia']:.3f}")

print("\n  Variables más importantes según modelo logístico:")
for i, row in coef_logit.iterrows():
    efecto = "positivo" if row['Coeficiente'] > 0 else "negativo"
    print(f"    - {row['Variable']}: {row['Coeficiente']:.3f} (efecto {efecto})")

print("\n RECOMENDACIONES:")
print(f"  • Usar el modelo {mejor_modelo} para predicciones")
print(f"  • Las 5 variables seleccionadas son efectivas para predecir deserción")
print(f"  • El modelo puede identificar patrones de deserción con buena precisión")

# Guardar resultados
resultados = {
    'Modelo': ['Random Forest', 'Logístico'],
    'Accuracy': [accuracy_rf, accuracy_logit],
    'Precision': [precision_rf, precision_logit],
    'Recall': [recall_rf, recall_logit],
    'F1_Score': [f1_rf, f1_logit],
    'AUC_ROC': [auc_rf, auc_logit]
}

resultados_df = pd.DataFrame(resultados)
resultados_df.to_csv('resultados_comparacion.csv', index=False)

print(f"\n Resultados guardados en 'resultados_comparacion.csv'")
print(f" Gráficos guardados:")
print(f"   - curvas_roc.png")
print(f"   - matrices_confusion.png") 
print(f"   - importancia_variables_modelos.png")

print("\n" + "=" * 60)
print("ANÁLISIS COMPLETADO")
print("=" * 60)
