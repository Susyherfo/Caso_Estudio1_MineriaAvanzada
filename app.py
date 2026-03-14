import streamlit as st
import matplotlib.pyplot as plt

from src.preprocesamiento import load_and_clean_data
from src.clasificacion import split_data, train_logistic_regression, train_random_forest
from src.k_fold import aplicar_kfold
from src.series_temporales import (
    prepare_time_series,
    train_test_split_time_series,
    run_arima,
    run_holt_winters,
)

st.set_page_config(page_title="Energy Consumption Analysis", layout="wide")

st.title("Household Energy Consumption Analysis")

df = load_and_clean_data()

menu = st.sidebar.selectbox(
    "Navigation",
    [
        "Dataset Overview",
        "Classification",
        "K-Fold Validation",
        "Time Series Forecasting",
    ],
)

# ------------------------------------------------
# DATASET
# ------------------------------------------------

if menu == "Dataset Overview":

    st.header("Dataset Overview")

    st.write(df.head())
    st.write("Shape:", df.shape)

# ------------------------------------------------
# CLASIFICACION
# ------------------------------------------------

elif menu == "Classification":

    st.header("Classification Models")

    X_train, X_test, y_train, y_test = split_data(df)

    log_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    log_auc = log_model.score(X_test, y_test)
    rf_auc = rf_model.score(X_test, y_test)

    col1, col2 = st.columns(2)

    col1.metric("Logistic Regression Accuracy", f"{log_auc:.3f}")
    col2.metric("Random Forest Accuracy", f"{rf_auc:.3f}")

# ------------------------------------------------
# K-FOLD
# ------------------------------------------------

elif menu == "K-Fold Validation":

    st.header("K-Fold Cross Validation")

    X_train, X_test, y_train, y_test = split_data(df)

    X = X_train.append(X_test)
    y = y_train.append(y_test)

    log_auc_mean, log_auc_std, rf_auc_mean, rf_auc_std = aplicar_kfold(X, y)

    col1, col2 = st.columns(2)

    col1.metric(
        "Logistic Regression AUC (mean)",
        f"{log_auc_mean:.3f}",
        f"± {log_auc_std:.3f}",
    )

    col2.metric(
        "Random Forest AUC (mean)",
        f"{rf_auc_mean:.3f}",
        f"± {rf_auc_std:.3f}",
    )

# ------------------------------------------------
# SERIES TEMPORALES
# ------------------------------------------------

elif menu == "Time Series Forecasting":

    st.header("Time Series Forecasting")

    series = prepare_time_series(df)

    train, test = train_test_split_time_series(series)

    arima_model, arima_forecast, arima_rmse, arima_mae = run_arima(train, test)

    hw_model, hw_forecast, hw_rmse, hw_mae = run_holt_winters(train, test)

    col1, col2 = st.columns(2)

    col1.metric("ARIMA RMSE", f"{arima_rmse:.3f}")
    col2.metric("Holt-Winters RMSE", f"{hw_rmse:.3f}")

    fig, ax = plt.subplots()

    ax.plot(train, label="Train")
    ax.plot(test, label="Test")
    ax.plot(test.index, arima_forecast, label="ARIMA Forecast")

    ax.legend()

    st.pyplot(fig)