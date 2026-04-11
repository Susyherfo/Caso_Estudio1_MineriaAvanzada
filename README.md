# Caso de Estudio – Minería de Datos Avanzada

# Caso de Estudio 1

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


# Caso de Estudio #2 – Minería de Datos Avanzada

## Web Mining, Redes Neuronales y Reglas de Asociación aplicadas al Consumo Energético


## Descripción

Caso de estudio integrado que aplica tres técnicas de minería de datos avanzada
sobre el dataset **Individual Household Electric Power Consumption**:

| Módulo | Técnica | Aplicación |
|--------|---------|------------|
| Web Mining | Content / Structure / Usage | Análisis de páginas IEA/EIA + API demanda eléctrica |
| Redes Neuronales | MLP, LSTM, CNN1D, Autoencoder, Regresión | Clasificación, predicción y detección de anomalías |
| Reglas de Asociación | FP-Growth (mlxtend) | Patrones de co-ocurrencia en consumo discretizado |

---

## Estructura del proyecto

```
caso_estudio_energia/
├── data/
│   └── energy.csv
├── src/
│   ├── preprocesamiento.py       # Carga y limpieza del dataset
│   ├── clasificacion.py          # Modelos de clasificación clásica
│   ├── k_fold.py                 # Validación cruzada estratificada
│   ├── series_temporales.py      # ARIMA y Holt-Winters
│   ├── hiperparametrizacion.py   # Búsqueda genética y Grid Search
│   ├── redes_neuronales.py       # 5 arquitecturas de redes neuronales  ← NUEVO
│   ├── reglas_asociacion.py      # FP-Growth sobre consumo energético   ← NUEVO
│   ├── web_mining.py             # Content, Structure y Usage Mining    ← NUEVO
│   └── data_api.py               # Conector API EIA
├── app.py                        # Dashboard Streamlit integrado
├── main.py                       # Pipeline CLI
├── requirements.txt
└── README.md
```

---

## Módulos nuevos (Caso de Estudio #2)

### 1. Web Mining (`src/web_mining.py`)

Tres subáreas implementadas:

**Content Mining**
- Extrae texto de páginas de organismos energéticos (IEA, EIA)
- Mide frecuencia de keywords energéticos
- Calcula score de relevancia (menciones / 1000 palabras)

**Structure Mining**
- Analiza jerarquía de headings, tipos de enlaces y datos estructurados
- Mide densidad de contenido informacional
- Detecta presencia de JSON-LD y Open Graph

**Usage Mining**
- Consume la API pública de EIA (U.S. Energy Information Administration)
- Obtiene datos de demanda eléctrica diaria del operador NYISO
- Permite contrastar consumo doméstico vs demanda de red

### 2. Redes Neuronales (`src/redes_neuronales.py`)

Cinco arquitecturas distintas aplicadas al mismo dataset:

| # | Modelo | Objetivo | Arquitectura clave |
|---|--------|----------|--------------------|
| 1 | MLP | Clasificación binaria | Dense(128→64→1) + Dropout |
| 2 | LSTM | Predicción series temporales | LSTM(64) + Dense(32→1) |
| 3 | CNN 1D | Clasificación en ventanas | Conv1D(32→64) + GlobalAvgPool |
| 4 | Autoencoder | Detección de anomalías | Encoder(64→32→16) + Decoder |
| 5 | Red Regresión | Predicción valor continuo | Dense(256→128→64→1) + BatchNorm |

### 3. Reglas de Asociación (`src/reglas_asociacion.py`)

- Discretiza variables continuas (bajo / medio / alto)
- Construye matriz de transacciones one-hot
- Aplica FP-Growth con umbrales configurables
- Genera reglas con soporte, confianza y lift
- Ejemplo: *"fin_de_semana & sub3_activo → consumo_alto (lift=2.1)"*

---

## Instalación y ejecución

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar dashboard
streamlit run app.py
```

El dashboard se abrirá en `http://localhost:8501`.

---

## Dataset

**Individual Household Electric Power Consumption**  
Fuente: UCI Machine Learning Repository  
Variables: Global Active Power, Voltage, Global Intensity, Sub Metering 1/2/3

---

## Evaluación del Caso #2

| Rubro | Peso | Implementado |
|-------|------|-------------|
| Conceptos: Web Mining, Redes Neuronales, Reglas de Asociación | 30% | ✓ |
| Implementación integrada en un solo caso práctico | 70% | ✓ |
