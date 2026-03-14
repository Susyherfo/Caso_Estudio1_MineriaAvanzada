import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.metrics import roc_curve, auc

from src.preprocesamiento import load_and_clean_data
from src.clasificacion import split_data, train_logistic_regression, train_random_forest
from src.k_fold import aplicar_kfold
from src.series_temporales import (
    prepare_time_series,
    train_test_split_time_series,
    run_arima,
    run_holt_winters
)

# ------------------------------------------------
# CONFIGURACION STREAMLIT
# ------------------------------------------------

st.set_page_config(
    page_title="Energy Consumption Analysis",
    layout="wide"
)

st.title("⚡ Household Energy Consumption Analysis")

st.sidebar.markdown("---")
st.sidebar.write("Minería de Datos Avanzada")
st.sidebar.write("Caso de Estudio 1")
st.sidebar.write("Susana Herrera Fonseca & Kendra Gutierrez")

# ------------------------------------------------
# CARGA DE DATOS
# ------------------------------------------------

df = load_and_clean_data("data/energy.csv")

# ------------------------------------------------
# MENU
# ------------------------------------------------

menu = st.sidebar.selectbox(
    "Navigation",
    [
        "Dataset Exploration",
        "Data Visualization",
        "Classification Models",
        "K-Fold Validation",
        "Time Series Forecasting"
    ],
)

# ------------------------------------------------
# DATASET
# ------------------------------------------------

if menu == "Dataset Exploration":

    st.header("Dataset Overview")

    rows = st.slider("Rows to display", 5, 50, 10)

    st.dataframe(df.head(rows))

    col1, col2 = st.columns(2)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.subheader("Summary statistics")

    st.dataframe(df.describe())


# ------------------------------------------------
# VISUALIZACION
# ------------------------------------------------

elif menu == "Data Visualization":

    st.header("Exploratory Data Analysis")

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    variable = st.selectbox(
        "Select variable",
        numeric_columns
    )

    chart_type = st.radio(
        "Chart type",
        ["Histogram", "Boxplot"]
    )

    fig, ax = plt.subplots()

    if chart_type == "Histogram":
        sns.histplot(df[variable], kde=True, ax=ax)

    else:
        sns.boxplot(x=df[variable], ax=ax)

    ax.set_title(variable)

    st.pyplot(fig)

    # ------------------------------------------------
    # HEATMAP CORRELACION
    # ------------------------------------------------

    st.subheader("Correlation Matrix")

    corr = df[numeric_columns].corr()

    fig_corr, ax_corr = plt.subplots(figsize=(10,6))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        ax=ax_corr
    )

    st.pyplot(fig_corr)

    # ------------------------------------------------
    # GRAFICO INTERACTIVO
    # ------------------------------------------------

    st.subheader("Interactive Time Series")

    fig_interactive = px.line(
        df,
        y="global_active_power",
        title="Energy Consumption Over Time"
    )

    st.plotly_chart(fig_interactive, use_container_width=True)


# ------------------------------------------------
# CLASIFICACION
# ------------------------------------------------

elif menu == "Classification Models":

    st.header("Classification Models")

    X_train, X_test, y_train, y_test = split_data(df)

    log_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    log_score = log_model.score(X_test, y_test)
    rf_score = rf_model.score(X_test, y_test)

    col1, col2 = st.columns(2)

    col1.metric(
        "Logistic Regression Accuracy",
        f"{log_score:.3f}"
    )

    col2.metric(
        "Random Forest Accuracy",
        f"{rf_score:.3f}"
    )

    # ------------------------------------------------
    # ROC CURVE
    # ------------------------------------------------

    st.subheader("ROC Curve")

    y_prob_log = log_model.predict_proba(X_test)[:,1]
    y_prob_rf = rf_model.predict_proba(X_test)[:,1]

    fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

    auc_log = auc(fpr_log, tpr_log)
    auc_rf = auc(fpr_rf, tpr_rf)

    fig_roc, ax_roc = plt.subplots()

    ax_roc.plot(
        fpr_log,
        tpr_log,
        label=f"Logistic Regression AUC = {auc_log:.3f}"
    )

    ax_roc.plot(
        fpr_rf,
        tpr_rf,
        label=f"Random Forest AUC = {auc_rf:.3f}"
    )

    ax_roc.plot([0,1],[0,1],'--')

    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")

    ax_roc.legend()

    st.pyplot(fig_roc)


# ------------------------------------------------
# K-FOLD
# ------------------------------------------------

elif menu == "K-Fold Validation":

    st.header("Stratified K-Fold Validation")

    X_train, X_test, y_train, y_test = split_data(df)

    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])

    log_auc_mean, log_auc_std, rf_auc_mean, rf_auc_std = aplicar_kfold(X, y)

    col1, col2 = st.columns(2)

    col1.metric(
        "Logistic Regression AUC",
        f"{log_auc_mean:.4f}",
        f"± {log_auc_std:.4f}"
    )

    col2.metric(
        "Random Forest AUC",
        f"{rf_auc_mean:.4f}",
        f"± {rf_auc_std:.4f}"
    )

    st.info(
        "Stratified K-Fold validation provides a more robust estimation of model performance."
    )


# ------------------------------------------------
# SERIES TEMPORALES
# ------------------------------------------------

elif menu == "Time Series Forecasting":

    st.header("Energy Consumption Forecasting")

    series = prepare_time_series(df)

    train, test = train_test_split_time_series(series)

    model_choice = st.selectbox(
        "Select forecasting model",
        ["ARIMA", "Holt-Winters"]
    )

    if model_choice == "ARIMA":

        model, forecast, rmse, mae = run_arima(train, test)

    else:

        model, forecast, rmse, mae = run_holt_winters(train, test)

    col1, col2 = st.columns(2)

    col1.metric("RMSE", f"{rmse:.3f}")
    col2.metric("MAE", f"{mae:.3f}")

    fig_forecast, ax_forecast = plt.subplots()

    ax_forecast.plot(train, label="Train")
    ax_forecast.plot(test, label="Test")
    ax_forecast.plot(test.index, forecast, label="Forecast")

    ax_forecast.legend()

    st.pyplot(fig_forecast)