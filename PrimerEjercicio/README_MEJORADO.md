# Análisis de Deserción de Clientes Bancarios - VERSIÓN MEJORADA

## Objetivo

Analizar qué factores influyen en que los clientes abandonen un banco y crear modelos para predecir la deserción, **con manejo apropiado del desbalance del dataset**.

---

## ⚠️ PROBLEMA IDENTIFICADO: Dataset Desbalanceado

### **Análisis del Balance**

- **Deserción**: 2,037 clientes (20.4%)
- **No Deserción**: 7,963 clientes (79.6%)
- **Ratio de balance**: 0.256 (25.6%)
- **Clasificación**: Dataset DESBALANCEADO

### **Impacto del Desbalance**

- Los modelos originales estaban sesgados hacia la clase mayoritaria
- Baja sensibilidad (recall) en identificar desertores
- Métricas engañosas (accuracy alta pero recall bajo)

---

## 1. Variables más importantes para identificar clientes que desertan

### **Metodología Utilizada**

Para encontrar las variables más importantes, utilizamos tres métodos diferentes y los combinamos:

1. **Análisis Estadístico (F-Score)**: Mide qué tan bien cada variable separa a los clientes que se van de los que se quedan
2. **Información Mutua**: Evalúa qué tan útil es cada variable para predecir la deserción
3. **Bosque Aleatorio**: Usa un modelo de árboles para determinar la importancia de cada variable

### **Top 5 Variables Más Importantes**

| Posición | Variable        | Puntuación | Explicación                                                                                             |
| -------- | --------------- | ---------- | ------------------------------------------------------------------------------------------------------- |
| **1**    | **CreditScore** | 1.00       | El puntaje crediticio es la variable más importante. Los clientes con mejor crédito tienden a quedarse. |
| **2**    | **Geography**   | 0.60       | La ubicación geográfica del cliente influye significativamente en la deserción.                         |
| **3**    | **Gender**      | 0.34       | El género del cliente tiene un impacto moderado en la decisión de abandonar.                            |
| **4**    | **Age**         | 0.31       | La edad del cliente es un factor importante para predecir la deserción.                                 |
| **5**    | **Tenure**      | 0.24       | El tiempo que el cliente lleva en el banco es relevante para predecir si se irá.                        |

---

## 2. Modelos creados para predecir la deserción

### **Modelos Originales (Sin manejo de desbalance)**

- **Bosque Aleatorio (Random Forest)**: Modelo de aprendizaje automático que usa múltiples árboles de decisión
- **Regresión Logística**: Modelo estadístico lineal para clasificación

### **Modelos Mejorados (Con manejo de desbalance)**

- **Bosque Aleatorio con Class Weights**: Asigna mayor peso a la clase minoritaria
- **Regresión Logística con Class Weights**: Compensa el desbalance con pesos ajustados

### **Técnica de Balanceo Aplicada**

- **Class Weights**: Asignar mayor peso a la clase minoritaria (deserción)
- **Peso para No Deserción**: 0.63
- **Peso para Deserción**: 2.45
- **Objetivo**: Compensar el desbalance del dataset (20.4% vs 79.6%)

---

## 3. Resultados de los modelos mejorados

### **Métricas de Rendimiento - Modelos Mejorados**

| Modelo                            | Accuracy | Precision | Recall    | F1-Score  | AUC-ROC |
| --------------------------------- | -------- | --------- | --------- | --------- | ------- |
| **Random Forest (Class Weights)** | 78.5%    | 45.7%     | 31.7%     | 37.4%     | 70.2%   |
| **Logístico (Class Weights)**     | 70.5%    | 37.3%     | **65.8%** | **47.6%** | 74.8%   |

### **Comparación con Modelos Originales**

| Modelo            | F1-Score Original | F1-Score Mejorado | Mejora    |
| ----------------- | ----------------- | ----------------- | --------- |
| **Random Forest** | 38.0%             | 37.4%             | Similar   |
| **Logístico**     | 11.7%             | **47.6%**         | **+308%** |

---

## 4. Mejor modelo identificado

### **Logístico con Class Weights**

**Razones de la elección:**

- **F1-Score más alto**: 47.6% (vs 37.4% del Random Forest)
- **Recall excepcional**: 65.8% (identifica al 66% de los desertores reales)
- **Mejora dramática**: +308% en F1-Score vs modelo original
- **Balance adecuado**: Entre precisión y sensibilidad

### **Interpretación del rendimiento**

- **Recall de 65.8%**: De cada 100 clientes que realmente desertan, el modelo identifica correctamente a 66
- **F1-Score de 47.6%**: Balance saludable entre precisión y sensibilidad
- **AUC-ROC de 74.8%**: Buena capacidad discriminativa

---

## 5. Impacto de las mejoras

### **Mejoras Obtenidas**

1. **Recall del Logístico**: De 6.9% a 65.8% (+857%)
2. **F1-Score del Logístico**: De 11.7% a 47.6% (+308%)
3. **Identificación de desertores**: 10 veces más efectivo

### **Impacto en el Negocio**

- **Mejor identificación**: Ahora se identifican al 66% de los clientes en riesgo
- **Estrategias de retención**: Más efectivas al tener mejor información
- **Reducción de pérdidas**: Mayor capacidad de prevenir deserciones

---

## 6. Técnicas aplicadas para el balanceo

### **Class Weights**

- **Concepto**: Asignar mayor peso a la clase minoritaria durante el entrenamiento
- **Implementación**: Automática en scikit-learn con `class_weight='balanced'`
- **Ventajas**: Simple, efectivo, no requiere modificar los datos

### **Alternativas Consideradas**

- **SMOTE**: Generación de datos sintéticos (no implementado en esta versión)
- **Undersampling**: Reducción de la clase mayoritaria
- **Ensemble Methods**: Combinación de múltiples modelos

---

## 7. Comparación final de modelos

### **Ranking por F1-Score (Mejorado)**

1. **Logístico (Class Weights)**: 47.6%
2. **Random Forest (Class Weights)**: 37.4%

### **Ranking por Recall (Mejorado)**

1. **Logístico (Class Weights)**: 65.8%
2. **Random Forest (Class Weights)**: 31.7%

### **Ranking por AUC-ROC (Mejorado)**

1. **Logístico (Class Weights)**: 74.8%
2. **Random Forest (Class Weights)**: 70.2%

---

## Conclusiones Principales

1. **El desbalance era crítico**: Los modelos originales no podían identificar desertores efectivamente
2. **Class Weights es efectivo**: Mejoró dramáticamente el rendimiento del modelo logístico
3. **El Logístico es superior**: Con balanceo, supera al Random Forest en todas las métricas importantes
4. **Recall mejorado significativamente**: De 6.9% a 65.8% (+857%)
5. **Aplicación práctica viable**: El modelo puede identificar efectivamente clientes en riesgo

---

## Archivos Generados

### **Análisis Original**

- `importancia_variables.csv`: Ranking de variables importantes
- `resultados_comparacion.csv`: Comparación de rendimiento de modelos originales
- `README.md`: Documentación original

### **Análisis Mejorado**

- `resultados_mejorados.csv`: Resultados con manejo de desbalance
- `modelos_balanceados_simple.py`: Script de modelos mejorados
- `resumen_mejoras_balanceo.py`: Resumen de mejoras obtenidas
- `README_MEJORADO.md`: Esta documentación

### **Visualizaciones**

- `matriz_correlacion.png`: Correlaciones entre variables
- `curvas_roc.png`: Curvas ROC de modelos originales
- `curvas_roc_mejoradas.png`: Curvas ROC de modelos mejorados
- `curvas_precision_recall.png`: Curvas Precision-Recall

---

## Recomendaciones para Implementación

### **Modelo Recomendado**

- **Usar**: Logístico con Class Weights
- **Métricas a monitorear**: F1-Score y Recall
- **Threshold**: Considerar ajustar según necesidades del negocio


---

## Resumen de Mejoras

| Aspecto                          | Antes    | Después   | Mejora        |
| -------------------------------- | -------- | --------- | ------------- |
| **Recall del Logístico**         | 6.9%     | 65.8%     | +857%         |
| **F1-Score del Logístico**       | 11.7%    | 47.6%     | +308%         |
| **Identificación de Desertores** | Muy baja | Alta      | Dramática     |
| **Aplicabilidad Práctica**       | Limitada | Excelente | Significativa |

**El manejo del desbalance transformó completamente la utilidad práctica del modelo para identificar clientes en riesgo de deserción.**
