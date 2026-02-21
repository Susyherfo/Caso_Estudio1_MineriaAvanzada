# ==========================
# imports principales
# ==========================

import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

from src.preprocesamiento import load_and_clean_data
from src.clasificacion import create_binary_target, split_data, train_logistic_regression, train_random_forest
from src.evaluacion import compute_auc
from src.series_temporales import prepare_time_series, train_test_split_time_series, run_arima, run_holt_winters, compare_models


# ==========================
# configuracion de la pagina
# ==========================

st.set_page_config(
    page_title="Minería de Datos - Energía",
    page_icon="⚡",
    layout="wide"
)


# ==========================
# carga de datos con cache
# ==========================

# st.cache_data evita recargar los datos cada vez que el usuario interactua con la app
@st.cache_data
def cargar_datos():
    df = load_and_clean_data("data/energy.csv", sample_size=50000, preserve_time_order=True)
    df = create_binary_target(df)
    return df


# ==========================
# menu lateral
# ==========================

with st.sidebar:
    selected = option_menu(
        menu_title="Menú Principal",
        options=["Inicio", "EDA", "Clasificación", "Series de Tiempo"],
        icons=["house", "bar-chart", "cpu", "clock"],
        menu_icon="cast",
        default_index=0,
    )


# ==========================
# paginas
# ==========================

df = cargar_datos()

# --- pagina de inicio ---
if selected == "Inicio":
    st.title("Análisis de Consumo Eléctrico")
    st.markdown("""
    Esta aplicación realiza un benchmarking de modelos de minería de datos
    sobre el dataset de consumo eléctrico del hogar (UCI).
    
    **Navegá por el menú lateral para explorar:**
    - 📊 **EDA**: análisis exploratorio de los datos
    - 🤖 **Clasificación**: comparación de modelos con AUC y curva ROC
    - 📈 **Series de Tiempo**: benchmarking de ARIMA y Holt-Winters
    """)

    st.subheader("Vista previa del dataset")
    st.dataframe(df.head(10))

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de registros", f"{len(df):,}")
    col2.metric("Variables", df.shape[1])
    col3.metric("Valores nulos", df.isnull().sum().sum())


# --- pagina de eda ---
elif selected == "EDA":
    st.title("Análisis Exploratorio (EDA)")
    
    st.subheader("Descripción estadística")
    st.write(df.describe())

    st.subheader("Histograma por variable")
    columna = st.selectbox("Seleccioná una variable:", df.select_dtypes(include='number').columns)
    fig = px.histogram(df, x=columna, marginal="box", title=f"Distribución de {columna}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mapa de correlación")
    corr = df.select_dtypes(include='number').corr()
    fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Heatmap de Correlación")
    st.plotly_chart(fig_corr, use_container_width=True)


# --- pagina de clasificacion ---
elif selected == "Clasificación":
    st.title("Clasificación y Benchmarking")

    st.info("Entrenando modelos... esto puede tardar un momento.")

    # entrenamiento con cache para no repetirlo
    @st.cache_data
    def entrenar_modelos(df):
        X_train, X_test, y_train, y_test = split_data(df)
        log_model = train_logistic_regression(X_train, y_train)
        rf_model = train_random_forest(X_train, y_train)
        log_auc, log_prob = compute_auc(log_model, X_test, y_test)
        rf_auc, rf_prob = compute_auc(rf_model, X_test, y_test)
        return X_test, y_test, log_auc, log_prob, rf_auc, rf_prob

    X_test, y_test, log_auc, log_prob, rf_auc, rf_prob = entrenar_modelos(df)

    # metricas en columnas
    col1, col2 = st.columns(2)
    col1.metric("AUC - Logistic Regression", f"{log_auc:.4f}")
    col2.metric("AUC - Random Forest", f"{rf_auc:.4f}")

    # curva roc con plotly
    st.subheader("Curva ROC")
    fpr_log, tpr_log, _ = roc_curve(y_test, log_prob)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr_log, y=tpr_log, name=f"Logistic (AUC={log_auc:.4f})"))
    fig_roc.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, name=f"Random Forest (AUC={rf_auc:.4f})"))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name="Baseline"))
    fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", title="Curva ROC")
    st.plotly_chart(fig_roc, use_container_width=True)


# --- pagina de series de tiempo ---
elif selected == "Series de Tiempo":
    st.title("Series de Tiempo")

    st.info("Entrenando modelos de series de tiempo...")

    @st.cache_data
    def entrenar_series(df):
        series = prepare_time_series(df)
        train_ts, test_ts = train_test_split_time_series(series)
        _, arima_forecast, arima_rmse, arima_mae = run_arima(train_ts, test_ts)
        _, hw_forecast, hw_rmse, hw_mae = run_holt_winters(train_ts, test_ts)
        return train_ts, test_ts, arima_forecast, arima_rmse, arima_mae, hw_forecast, hw_rmse, hw_mae

    train_ts, test_ts, arima_forecast, arima_rmse, arima_mae, hw_forecast, hw_rmse, hw_mae = entrenar_series(df)

    # comparacion de metricas
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ARIMA")
        st.metric("RMSE", f"{arima_rmse:.4f}")
        st.metric("MAE", f"{arima_mae:.4f}")
    with col2:
        st.subheader("Holt-Winters")
        st.metric("RMSE", f"{hw_rmse:.4f}")
        st.metric("MAE", f"{hw_mae:.4f}")

    decision = compare_models((arima_rmse, arima_mae), (hw_rmse, hw_mae))
    st.success(decision)

    # grafico de forecasts
    st.subheader("Comparación de Pronósticos")
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(y=test_ts.values, name="Real"))
    fig_ts.add_trace(go.Scatter(y=arima_forecast, name="ARIMA"))
    fig_ts.add_trace(go.Scatter(y=hw_forecast, name="Holt-Winters"))
    fig_ts.update_layout(title="Forecast vs Real", xaxis_title="Tiempo", yaxis_title="Consumo")
    st.plotly_chart(fig_ts, use_container_width=True)