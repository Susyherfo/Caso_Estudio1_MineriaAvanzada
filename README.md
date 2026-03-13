# Caso de Estudio – Minería de Datos Avanzada

## Modelado Predictivo y Análisis de Series Temporales en Datos de Consumo Energético

Proyecto desarrollado para el curso **BCD-7213 – Minería de Datos Avanzada** en **LEAD University**.

Autores:

- Susana Herrera Fonseca
- Kendra Gutiérrez

Profesor:
Juan Murillo Morera

San José, Costa Rica – Marzo 2026

---

# Descripción del Proyecto

Este proyecto desarrolla un análisis de consumo energético utilizando técnicas de **minería de datos, aprendizaje supervisado y modelado de series temporales**.

El objetivo principal es analizar el comportamiento del consumo eléctrico y evaluar distintos modelos predictivos que permitan:

- Identificar **eventos de alto consumo energético**
- Analizar el **comportamiento temporal del consumo**
- Comparar el desempeño de distintos modelos estadísticos y de machine learning

Para ello se utilizan tanto **modelos de clasificación supervisada** como **modelos de series temporales**.

---

# Dataset

Se utiliza el dataset **Individual Household Electric Power Consumption**, el cual contiene mediciones del consumo eléctrico doméstico registradas a intervalos de tiempo.

El conjunto de datos incluye variables como:

- Global Active Power
- Global Reactive Power
- Voltage
- Global Intensity
- Sub Metering 1
- Sub Metering 2
- Sub Metering 3

A partir de estas variables se construyó una variable objetivo binaria que identifica **eventos de alto consumo energético**.

---

# Modelos Utilizados

Se implementaron distintos modelos para abordar el problema desde diferentes perspectivas.

## Modelos de Clasificación

Se entrenaron modelos supervisados para identificar eventos de alto consumo:

### Logistic Regression

Modelo probabilístico utilizado como **baseline** para clasificación binaria.

Ventajas:

- Interpretabilidad
- Estabilidad estadística
- Bajo costo computacional

### Random Forest

Modelo de **ensamble basado en árboles de decisión** que permite capturar relaciones no lineales entre variables.

Ventajas:

- Alta capacidad predictiva
- Reducción de varianza
- Manejo de interacciones entre variables

---

## Modelos de Series Temporales

Para analizar la evolución del consumo energético se utilizaron modelos clásicos de series temporales.

### ARIMA

Modelo autoregresivo integrado de media móvil que captura dependencias temporales y tendencias.

### Holt-Winters

Modelo de suavizamiento exponencial que incorpora componentes de tendencia.

---

# Evaluación de Modelos

## Clasificación

Los modelos se evaluaron mediante:

- Área bajo la curva ROC (AUC)
- Curva ROC

Resultados obtenidos:

| Modelo | AUC |
|------|------|
| Logistic Regression | 0.7993 |
| Random Forest | 0.7847 |

---

## Series Temporales

Para la evaluación de los modelos de predicción temporal se utilizaron:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

Resultados:

| Modelo | RMSE | MAE |
|------|------|------|
| ARIMA | 0.4232 | 0.3440 |
| Holt-Winters | 0.4701 | 0.3274 |

---

# Dashboard Interactivo

Se desarrolló una aplicación interactiva utilizando **Streamlit** que permite:

- Análisis exploratorio del consumo energético
- Visualización de series temporales
- Evaluación de modelos
- Comparación de predicciones
- Exploración dinámica del dataset

El dashboard incluye:

- métricas resumen
- gráficos de consumo
- curva ROC
- comparación de predicciones
- análisis temporal

---

# ⚙️ Tecnologías Utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- Plotly
- Streamlit

---

# Cómo ejecutar el proyecto

### 1 Instalar dependencias
pip install -r requirements.txt
streamlit run app.py

---
Esto abrirá el **dashboard interactivo** en el navegador.

---

#  Conclusiones

Los resultados muestran que distintos enfoques ofrecen ventajas complementarias:

- Logistic Regression mostró buen desempeño discriminativo.
- Random Forest capturó relaciones no lineales entre variables.
- ARIMA presentó mejor desempeño en predicción temporal según RMSE.
- Holt-Winters mostró menor error absoluto medio.

La integración de técnicas de clasificación y series temporales permite obtener una visión más completa del comportamiento del consumo energético.

---

# Curso

BCD-7213 – Minería de Datos Avanzada  
LEAD University


# 📂 Estructura del Proyecto
