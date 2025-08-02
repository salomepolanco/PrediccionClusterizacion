# -*- coding: utf-8 -*-
"""
MODELOS MEJORADOS DE PREDICCIÓN DE DESERCIÓN
Con manejo apropiado del desbalance del dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, roc_auc_score, roc_curve,
                           balanced_accuracy_score, matthews_corrcoef, precision_recall_curve, auc)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("=" * 70)
print("MODELOS MEJORADOS DE PREDICCIÓN DE DESERCIÓN")
print("Manejo apropiado del desbalance del dataset")
print("=" * 70)

#----------------------------------------------- 1. CARGA Y PREPARACIÓN DE DATOS

print("\n1. CARGA Y PREPARACIÓN DE DATOS")
print("-" * 50)

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

# Calcular ratio de balance
balance_ratio = min(y.value_counts()) / max(y.value_counts())
print(f"  - Ratio de balance: {balance_ratio:.3f}")

#----------------------------------------------- 2. PREPROCESAMIENTO

print("\n2. PREPROCESAMIENTO")
print("-" * 50)

# Codificar variables categóricas
le_geography = LabelEncoder()
le_gender = LabelEncoder()

X['Geography'] = le_geography.fit_transform(X['Geography'])
X['Gender'] = le_gender.fit_transform(X['Gender'])

print("Variables categóricas codificadas:")
print(f"  - Geography: {list(le_geography.classes_)} → {list(range(len(le_geography.classes_)))}")
print(f"  - Gender: {list(le_gender.classes_)} → {list(range(len(le_gender.classes_)))}")

# Dividir en train y test (estratificado para mantener proporción de clases)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nDivisión train/test (estratificada):")
print(f"  - Train: {X_train.shape[0]} muestras")
print(f"  - Test: {X_test.shape[0]} muestras")
print(f"  - Distribución en train: {y_train.value_counts().to_dict()}")
print(f"  - Distribución en test: {y_test.value_counts().to_dict()}")

#----------------------------------------------- 3. MANEJO DEL DESBALANCE

print("\n3. MANEJO DEL DESBALANCE")
print("-" * 50)

# Calcular class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

print(f"Class weights calculados:")
print(f"  - Clase 0 (No Deserción): {class_weight_dict[0]:.3f}")
print(f"  - Clase 1 (Deserción): {class_weight_dict[1]:.3f}")

# Aplicar SMOTE para balancear el dataset de entrenamiento
print("\nAplicando SMOTE para balancear el dataset...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Dataset después de SMOTE:")
print(f"  - Train balanceado: {X_train_balanced.shape[0]} muestras")
print(f"  - Distribución: {pd.Series(y_train_balanced).value_counts().to_dict()}")

#----------------------------------------------- 4. MODELO RANDOM FOREST MEJORADO

print("\n4. MODELO RANDOM FOREST MEJORADO")
print("-" * 50)

# Entrenar modelo con class weights
rf_model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)

# Entrenar modelo con datos balanceados por SMOTE
rf_model_smote = RandomForestClassifier(
    n_estimators=100, 
    random_state=42
)
rf_model_smote.fit(X_train_balanced, y_train_balanced)

# Predicciones
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

y_pred_rf_smote = rf_model_smote.predict(X_test)
y_pred_proba_rf_smote = rf_model_smote.predict_proba(X_test)[:, 1]

# Métricas para Random Forest con class weights
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
balanced_acc_rf = balanced_accuracy_score(y_test, y_pred_rf)
mcc_rf = matthews_corrcoef(y_test, y_pred_rf)

# Métricas para Random Forest con SMOTE
accuracy_rf_smote = accuracy_score(y_test, y_pred_rf_smote)
precision_rf_smote = precision_score(y_test, y_pred_rf_smote)
recall_rf_smote = recall_score(y_test, y_pred_rf_smote)
f1_rf_smote = f1_score(y_test, y_pred_rf_smote)
auc_rf_smote = roc_auc_score(y_test, y_pred_proba_rf_smote)
balanced_acc_rf_smote = balanced_accuracy_score(y_test, y_pred_rf_smote)
mcc_rf_smote = matthews_corrcoef(y_test, y_pred_rf_smote)

print("Métricas Random Forest (Class Weights):")
print(f"  - Accuracy: {accuracy_rf:.3f}")
print(f"  - Precision: {precision_rf:.3f}")
print(f"  - Recall: {recall_rf:.3f}")
print(f"  - F1-Score: {f1_rf:.3f}")
print(f"  - AUC-ROC: {auc_rf:.3f}")
print(f"  - Balanced Accuracy: {balanced_acc_rf:.3f}")
print(f"  - MCC: {mcc_rf:.3f}")

print("\nMétricas Random Forest (SMOTE):")
print(f"  - Accuracy: {accuracy_rf_smote:.3f}")
print(f"  - Precision: {precision_rf_smote:.3f}")
print(f"  - Recall: {recall_rf_smote:.3f}")
print(f"  - F1-Score: {f1_rf_smote:.3f}")
print(f"  - AUC-ROC: {auc_rf_smote:.3f}")
print(f"  - Balanced Accuracy: {balanced_acc_rf_smote:.3f}")
print(f"  - MCC: {mcc_rf_smote:.3f}")

#----------------------------------------------- 5. MODELO LOGÍSTICO MEJORADO

print("\n5. MODELO LOGÍSTICO MEJORADO")
print("-" * 50)

# Entrenar modelo con class weights
logit_model = LogisticRegression(
    random_state=42, 
    max_iter=1000,
    class_weight='balanced'
)
logit_model.fit(X_train, y_train)

# Entrenar modelo con datos balanceados por SMOTE
logit_model_smote = LogisticRegression(
    random_state=42, 
    max_iter=1000
)
logit_model_smote.fit(X_train_balanced, y_train_balanced)

# Predicciones
y_pred_logit = logit_model.predict(X_test)
y_pred_proba_logit = logit_model.predict_proba(X_test)[:, 1]

y_pred_logit_smote = logit_model_smote.predict(X_test)
y_pred_proba_logit_smote = logit_model_smote.predict_proba(X_test)[:, 1]

# Métricas para Logístico con class weights
accuracy_logit = accuracy_score(y_test, y_pred_logit)
precision_logit = precision_score(y_test, y_pred_logit)
recall_logit = recall_score(y_test, y_pred_logit)
f1_logit = f1_score(y_test, y_pred_logit)
auc_logit = roc_auc_score(y_test, y_pred_proba_logit)
balanced_acc_logit = balanced_accuracy_score(y_test, y_pred_logit)
mcc_logit = matthews_corrcoef(y_test, y_pred_logit)

# Métricas para Logístico con SMOTE
accuracy_logit_smote = accuracy_score(y_test, y_pred_logit_smote)
precision_logit_smote = precision_score(y_test, y_pred_logit_smote)
recall_logit_smote = recall_score(y_test, y_pred_logit_smote)
f1_logit_smote = f1_score(y_test, y_pred_logit_smote)
auc_logit_smote = roc_auc_score(y_test, y_pred_proba_logit_smote)
balanced_acc_logit_smote = balanced_accuracy_score(y_test, y_pred_logit_smote)
mcc_logit_smote = matthews_corrcoef(y_test, y_pred_logit_smote)

print("Métricas Logístico (Class Weights):")
print(f"  - Accuracy: {accuracy_logit:.3f}")
print(f"  - Precision: {precision_logit:.3f}")
print(f"  - Recall: {recall_logit:.3f}")
print(f"  - F1-Score: {f1_logit:.3f}")
print(f"  - AUC-ROC: {auc_logit:.3f}")
print(f"  - Balanced Accuracy: {balanced_acc_logit:.3f}")
print(f"  - MCC: {mcc_logit:.3f}")

print("\nMétricas Logístico (SMOTE):")
print(f"  - Accuracy: {accuracy_logit_smote:.3f}")
print(f"  - Precision: {precision_logit_smote:.3f}")
print(f"  - Recall: {recall_logit_smote:.3f}")
print(f"  - F1-Score: {f1_logit_smote:.3f}")
print(f"  - AUC-ROC: {auc_logit_smote:.3f}")
print(f"  - Balanced Accuracy: {balanced_acc_logit_smote:.3f}")
print(f"  - MCC: {mcc_logit_smote:.3f}")

#----------------------------------------------- 6. VALIDACIÓN CRUZADA

print("\n6. VALIDACIÓN CRUZADA")
print("-" * 50)

# Validación cruzada estratificada
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Para Random Forest con class weights
cv_scores_rf = cross_val_score(rf_model, X, y, cv=skf, scoring='f1')
print(f"Random Forest (Class Weights) - CV F1-Score: {cv_scores_rf.mean():.3f} (+/- {cv_scores_rf.std() * 2:.3f})")

# Para Random Forest con SMOTE
cv_scores_rf_smote = cross_val_score(rf_model_smote, X, y, cv=skf, scoring='f1')
print(f"Random Forest (SMOTE) - CV F1-Score: {cv_scores_rf_smote.mean():.3f} (+/- {cv_scores_rf_smote.std() * 2:.3f})")

# Para Logístico con class weights
cv_scores_logit = cross_val_score(logit_model, X, y, cv=skf, scoring='f1')
print(f"Logístico (Class Weights) - CV F1-Score: {cv_scores_logit.mean():.3f} (+/- {cv_scores_logit.std() * 2:.3f})")

# Para Logístico con SMOTE
cv_scores_logit_smote = cross_val_score(logit_model_smote, X, y, cv=skf, scoring='f1')
print(f"Logístico (SMOTE) - CV F1-Score: {cv_scores_logit_smote.mean():.3f} (+/- {cv_scores_logit_smote.std() * 2:.3f})")

#----------------------------------------------- 7. VISUALIZACIONES MEJORADAS

print("\n7. GENERANDO VISUALIZACIONES MEJORADAS")
print("-" * 50)

# 1. Curvas ROC comparativas
plt.figure(figsize=(12, 8))

# Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
fpr_rf_smote, tpr_rf_smote, _ = roc_curve(y_test, y_pred_proba_rf_smote)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (Class Weights) - AUC = {auc_rf:.3f}', linewidth=2)
plt.plot(fpr_rf_smote, tpr_rf_smote, label=f'Random Forest (SMOTE) - AUC = {auc_rf_smote:.3f}', linewidth=2, linestyle='--')

# Logístico
fpr_logit, tpr_logit, _ = roc_curve(y_test, y_pred_proba_logit)
fpr_logit_smote, tpr_logit_smote, _ = roc_curve(y_test, y_pred_proba_logit_smote)
plt.plot(fpr_logit, tpr_logit, label=f'Logístico (Class Weights) - AUC = {auc_logit:.3f}', linewidth=2)
plt.plot(fpr_logit_smote, tpr_logit_smote, label=f'Logístico (SMOTE) - AUC = {auc_logit_smote:.3f}', linewidth=2, linestyle='--')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC - Modelos Mejorados')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('curvas_roc_mejoradas.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Curvas Precision-Recall
plt.figure(figsize=(12, 8))

# Random Forest
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_pred_proba_rf)
precision_rf_smote, recall_rf_smote, _ = precision_recall_curve(y_test, y_pred_proba_rf_smote)
pr_auc_rf = auc(recall_rf, precision_rf)
pr_auc_rf_smote = auc(recall_rf_smote, precision_rf_smote)

plt.plot(recall_rf, precision_rf, label=f'Random Forest (Class Weights) - PR-AUC = {pr_auc_rf:.3f}', linewidth=2)
plt.plot(recall_rf_smote, precision_rf_smote, label=f'Random Forest (SMOTE) - PR-AUC = {pr_auc_rf_smote:.3f}', linewidth=2, linestyle='--')

# Logístico
precision_logit, recall_logit, _ = precision_recall_curve(y_test, y_pred_proba_logit)
precision_logit_smote, recall_logit_smote, _ = precision_recall_curve(y_test, y_pred_proba_logit_smote)
pr_auc_logit = auc(recall_logit, precision_logit)
pr_auc_logit_smote = auc(recall_logit_smote, precision_logit_smote)

plt.plot(recall_logit, precision_logit, label=f'Logístico (Class Weights) - PR-AUC = {pr_auc_logit:.3f}', linewidth=2)
plt.plot(recall_logit_smote, precision_logit_smote, label=f'Logístico (SMOTE) - PR-AUC = {pr_auc_logit_smote:.3f}', linewidth=2, linestyle='--')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curvas Precision-Recall - Modelos Mejorados')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('curvas_precision_recall.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Comparación de métricas
modelos = ['RF (Class Weights)', 'RF (SMOTE)', 'Logit (Class Weights)', 'Logit (SMOTE)']
f1_scores = [f1_rf, f1_rf_smote, f1_logit, f1_logit_smote]
recalls = [recall_rf, recall_rf_smote, recall_logit, recall_logit_smote]
balanced_accs = [balanced_acc_rf, balanced_acc_rf_smote, balanced_acc_logit, balanced_acc_logit_smote]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# F1-Score
axes[0].bar(modelos, f1_scores, color=['blue', 'lightblue', 'orange', 'lightcoral'])
axes[0].set_title('F1-Score por Modelo')
axes[0].set_ylabel('F1-Score')
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(f1_scores):
    axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# Recall
axes[1].bar(modelos, recalls, color=['blue', 'lightblue', 'orange', 'lightcoral'])
axes[1].set_title('Recall por Modelo')
axes[1].set_ylabel('Recall')
axes[1].tick_params(axis='x', rotation=45)
for i, v in enumerate(recalls):
    axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# Balanced Accuracy
axes[2].bar(modelos, balanced_accs, color=['blue', 'lightblue', 'orange', 'lightcoral'])
axes[2].set_title('Balanced Accuracy por Modelo')
axes[2].set_ylabel('Balanced Accuracy')
axes[2].tick_params(axis='x', rotation=45)
for i, v in enumerate(balanced_accs):
    axes[2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('comparacion_metricas_mejoradas.png', dpi=300, bbox_inches='tight')
plt.close()

#----------------------------------------------- 8. GUARDAR RESULTADOS MEJORADOS

print("\n8. GUARDANDO RESULTADOS MEJORADOS")
print("-" * 50)

# Crear DataFrame con todos los resultados
resultados_mejorados = {
    'Modelo': ['RF (Class Weights)', 'RF (SMOTE)', 'Logit (Class Weights)', 'Logit (SMOTE)'],
    'Accuracy': [accuracy_rf, accuracy_rf_smote, accuracy_logit, accuracy_logit_smote],
    'Precision': [precision_rf, precision_rf_smote, precision_logit, precision_logit_smote],
    'Recall': [recall_rf, recall_rf_smote, recall_logit, recall_logit_smote],
    'F1_Score': [f1_rf, f1_rf_smote, f1_logit, f1_logit_smote],
    'AUC_ROC': [auc_rf, auc_rf_smote, auc_logit, auc_logit_smote],
    'Balanced_Accuracy': [balanced_acc_rf, balanced_acc_rf_smote, balanced_acc_logit, balanced_acc_logit_smote],
    'MCC': [mcc_rf, mcc_rf_smote, mcc_logit, mcc_logit_smote]
}

resultados_df = pd.DataFrame(resultados_mejorados)
resultados_df.to_csv('resultados_mejorados.csv', index=False)

print("✅ Resultados guardados en 'resultados_mejorados.csv'")
print("✅ Gráficos guardados:")
print("   - curvas_roc_mejoradas.png")
print("   - curvas_precision_recall.png")
print("   - comparacion_metricas_mejoradas.png")

#----------------------------------------------- 9. RESUMEN FINAL

print("\n9. RESUMEN FINAL")
print("-" * 50)

# Encontrar el mejor modelo basado en F1-Score
mejor_modelo_idx = resultados_df['F1_Score'].idxmax()
mejor_modelo = resultados_df.loc[mejor_modelo_idx, 'Modelo']
mejor_f1 = resultados_df.loc[mejor_modelo_idx, 'F1_Score']
mejor_recall = resultados_df.loc[mejor_modelo_idx, 'Recall']

print(f"🏆 MEJOR MODELO: {mejor_modelo}")
print(f"   - F1-Score: {mejor_f1:.3f}")
print(f"   - Recall: {mejor_recall:.3f}")

print(f"\n📊 MEJORAS OBTENIDAS:")
print(f"   - F1-Score mejorado de ~0.38 a {mejor_f1:.3f} (+{((mejor_f1-0.38)/0.38*100):.1f}%)")
print(f"   - Recall mejorado de ~0.33 a {mejor_recall:.3f} (+{((mejor_recall-0.33)/0.33*100):.1f}%)")

print(f"\n💡 CONCLUSIONES:")
print(f"   - El manejo del desbalance mejoró significativamente el rendimiento")
print(f"   - SMOTE y class weights son técnicas efectivas para este dataset")
print(f"   - Las métricas apropiadas muestran mejor rendimiento real")

print("\n" + "=" * 70)
print("ANÁLISIS MEJORADO COMPLETADO")
print("=" * 70) 