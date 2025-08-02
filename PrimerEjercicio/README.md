# 📊 Análisis de Deserción de Clientes Bancarios

## 🎯 Objetivo

Analizar qué factores influyen en que los clientes abandonen un banco y crear modelos para predecir la deserción.

---

## 1. 🔍 ¿Cuáles son las variables más importantes para identificar clientes que desertan?

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

### **¿Por qué estas 5 variables?**

- **CreditScore**: Es el indicador más fuerte de la salud financiera del cliente
- **Geography**: Diferentes regiones tienen diferentes comportamientos bancarios
- **Gender**: Los patrones de deserción varían entre géneros
- **Age**: Los clientes de diferentes edades tienen diferentes necesidades bancarias
- **Tenure**: Los clientes más antiguos suelen ser más leales

---

## 2. 🤖 ¿Qué modelo se creó para predecir la deserción?

### **Modelo: Bosque Aleatorio (Random Forest)**

- **Tipo**: Modelo de aprendizaje automático que usa múltiples árboles de decisión
- **Variables utilizadas**: Las 5 variables más importantes identificadas
- **Objetivo**: Predecir si un cliente abandonará el banco (Sí/No)

### **¿Cómo funciona?**

El modelo analiza las características del cliente (puntaje crediticio, ubicación, género, edad, tiempo en el banco) y determina la probabilidad de que abandone el banco.

---

## 3. 📈 ¿Cómo interpretar los resultados del Bosque Aleatorio?

### **Métricas de Rendimiento**

| Métrica          | Valor | ¿Qué significa?                                                                |
| ---------------- | ----- | ------------------------------------------------------------------------------ |
| **Precisión**    | 78.1% | De cada 100 predicciones, 78 son correctas                                     |
| **Exactitud**    | 44.8% | Cuando el modelo dice que un cliente se va, tiene razón el 45% de las veces    |
| **Sensibilidad** | 32.9% | El modelo identifica correctamente al 33% de los clientes que realmente se van |
| **F1-Score**     | 38.0% | Puntuación balanceada entre precisión y sensibilidad                           |
| **AUC-ROC**      | 70.9% | El modelo es 71% mejor que adivinar al azar                                    |

### **Interpretación Simple**

- **El modelo es moderadamente bueno**: Predice correctamente en el 78% de los casos
- **Tiene dificultad para identificar desertores**: Solo detecta al 33% de los clientes que realmente se van
- **Es mejor que adivinar**: Tiene un rendimiento 71% mejor que la casualidad

---

## 4. 📊 ¿Qué modelo logístico se creó?

### **Modelo: Regresión Logística**

- **Tipo**: Modelo estadístico lineal para clasificación
- **Variables utilizadas**: Las mismas 5 variables importantes
- **Objetivo**: Igual que el anterior, predecir deserción

### **¿Cómo funciona?**

La regresión logística calcula la probabilidad de deserción basándose en una combinación lineal de las variables del cliente.

---

## 5. 📊 ¿Cómo interpretar los resultados del modelo logístico?

### **Métricas de Rendimiento**

| Métrica          | Valor | ¿Qué significa?                                                                    |
| ---------------- | ----- | ---------------------------------------------------------------------------------- |
| **Precisión**    | 78.8% | De cada 100 predicciones, 79 son correctas                                         |
| **Exactitud**    | 38.4% | Cuando el modelo dice que un cliente se va, tiene razón el 38% de las veces        |
| **Sensibilidad** | 6.9%  | El modelo identifica correctamente solo al 7% de los clientes que realmente se van |
| **F1-Score**     | 11.7% | Puntuación balanceada muy baja                                                     |
| **AUC-ROC**      | 74.4% | El modelo es 74% mejor que adivinar al azar                                        |

### **Interpretación Simple**

- **Precisión similar**: Ambos modelos tienen rendimiento similar en general
- **Muy baja sensibilidad**: El modelo logístico es muy malo para identificar clientes que realmente se van
- **Mejor AUC-ROC**: Tiene mejor capacidad discriminativa que el bosque aleatorio

---

## 6. ⚖️ ¿Cuál modelo es mejor? ¿Por qué?

### **Comparación Directa**

| Métrica          | Bosque Aleatorio | Regresión Logística | Ganador    |
| ---------------- | ---------------- | ------------------- | ---------- |
| **Precisión**    | 78.1%            | 78.8%               | Logístico  |
| **Exactitud**    | 44.8%            | 38.4%               | Bosque     |
| **Sensibilidad** | 32.9%            | 6.9%                | **Bosque** |
| **F1-Score**     | 38.0%            | 11.7%               | **Bosque** |
| **AUC-ROC**      | 70.9%            | 74.4%               | Logístico  |

### **¿Cuál es mejor?**

**🏆 El Bosque Aleatorio es mejor para este caso específico**

### **¿Por qué?**

1. **Mejor sensibilidad**: Identifica correctamente al 33% de los desertores vs solo 7% del logístico
2. **Mejor F1-Score**: Tiene un balance más saludable entre precisión y sensibilidad
3. **Más útil en la práctica**: Para un banco, es más importante identificar clientes que se van que tener alta precisión general

### **¿Cuándo usar cada uno?**

- **Bosque Aleatorio**: Cuando necesitas identificar la mayor cantidad posible de clientes en riesgo
- **Regresión Logística**: Cuando necesitas un modelo más simple y interpretable

---

## 🎯 Conclusiones Principales

1. **Las 5 variables más importantes** para predecir deserción son: puntaje crediticio, ubicación, género, edad y tiempo en el banco.

2. **Ambos modelos tienen rendimiento similar** en términos generales (precisión ~78%).

3. **El Bosque Aleatorio es más útil** porque identifica mejor a los clientes que realmente abandonan.

4. **La deserción es difícil de predecir** - incluso el mejor modelo solo identifica al 33% de los desertores reales.

5. **El puntaje crediticio es el factor más importante** - los clientes con mejor crédito tienden a ser más leales.

---

## 📁 Archivos Generados

- `importancia_variables.csv`: Ranking de variables importantes
- `resultados_comparacion.csv`: Comparación de rendimiento de modelos
- `matriz_correlacion.png`: Correlaciones entre variables
- `curvas_roc.png`: Curvas ROC de ambos modelos
- `matrices_confusion.png`: Matrices de confusión de ambos modelos
