import streamlit as st
import pandas as pd
import plotly.express as px

from main import (
    load_data,
    run_models,
    make_predictions,
    cross_validation_models,
    benchmark_timeseries
)

# ----------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------
st.set_page_config(
    page_title="Análisis Energético",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Dashboard de Análisis Energético")

# ----------------------------------------------------
# CARGAR DATOS
# ----------------------------------------------------
df = load_data()

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
st.sidebar.header("⚙️ Panel de Control")

date_range = st.sidebar.date_input(
    "Seleccionar rango de fechas",
    [df["datetime"].min(), df["datetime"].max()]
)

model_selected = st.sidebar.selectbox(
    "Modelo a visualizar",
    ["ARIMA", "Holt-Winters", "Random Forest", "Logistic Regression"]
)

retrain = st.sidebar.button("Re-entrenar modelo")

# ----------------------------------------------------
# FILTRAR DATOS
# ----------------------------------------------------
df_filtered = df[
    (df["datetime"] >= pd.to_datetime(date_range[0])) &
    (df["datetime"] <= pd.to_datetime(date_range[1]))
]

# ----------------------------------------------------
# TABS
# ----------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Análisis Exploratorio",
    "🤖 Modelos y Evaluación",
    "📈 Predicciones",
    "📌 Conclusiones"
])

# ----------------------------------------------------
# TAB 1 - EDA
# ----------------------------------------------------
with tab1:

    st.subheader("Resumen del dataset")

    if df_filtered.empty:
        st.warning("No hay datos en el rango seleccionado")

    else:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Consumo promedio",
            f"{df_filtered['consumption'].mean():.2f}"
        )

        col2.metric(
            "Pico máximo",
            f"{df_filtered['consumption'].max():.2f}"
        )

        col3.metric(
            "Total registros",
            len(df_filtered)
        )

        col4.metric(
            "Rango fechas",
            f"{df_filtered['datetime'].min().date()} - {df_filtered['datetime'].max().date()}"
        )

        st.divider()

        st.subheader("Serie temporal de consumo")

        fig = px.line(
            df_filtered,
            x="datetime",
            y="consumption",
            title="Consumo energético en el tiempo"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Distribución del consumo")

        fig2 = px.histogram(
            df_filtered,
            x="consumption",
            nbins=50
        )

        st.plotly_chart(fig2, use_container_width=True)

# ----------------------------------------------------
# TAB 2 - MODELOS
# ----------------------------------------------------
with tab2:

    st.subheader("Evaluación de Modelos")

    if retrain:

        results = run_models(df_filtered)

        st.success("Modelos re-entrenados")

        st.dataframe(results)

    st.divider()

    st.subheader("Cross Validation (K-Fold)")

    cv_results = cross_validation_models(df_filtered)

    st.dataframe(cv_results)

    st.divider()

    st.subheader("Benchmark Series Temporales")

    ts_results = benchmark_timeseries(df_filtered)

    st.dataframe(ts_results)

    with st.expander("ℹ️ Interpretación de métricas"):

        st.write("""
        **RMSE** mide el error cuadrático medio.

        **MAE** representa el error absoluto medio.

        **AUC** evalúa la capacidad del modelo para distinguir clases.
        """)

# ----------------------------------------------------
# TAB 3 - PREDICCIONES
# ----------------------------------------------------
with tab3:

    st.subheader("Predicciones")

    predictions = make_predictions(model_selected, df_filtered)

    fig = px.line(
        predictions,
        x="datetime",
        y=["real", "predicted"],
        title=f"Predicciones usando {model_selected}"
    )

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------
# TAB 4 - CONCLUSIONES
# ----------------------------------------------------
with tab4:

    st.subheader("Conclusiones del análisis")

    st.write("""
    - Los modelos permiten identificar patrones de consumo energético.
    - ARIMA mostró buen desempeño en series temporales.
    - Random Forest captura relaciones no lineales en los datos.
    """)

    with st.expander("Cómo usar la aplicación"):

        st.write("""
        1. Seleccione un rango de fechas en el panel lateral.
        2. Elija el modelo a visualizar.
        3. Explore resultados y predicciones.
        """)