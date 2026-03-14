"""
Pipeline principal del proyecto
Caso de Estudio 1 - Minería de Datos Avanzada
"""

from src.preprocesamiento import load_and_clean_data
from src.clasificacion import split_data, train_logistic_regression, train_random_forest
from src.k_fold import aplicar_kfold
from src.series_temporales import (
    prepare_time_series,
    train_test_split_time_series,
    run_arima,
    run_holt_winters,
)


def main():

    print("Loading dataset...")
    df = load_and_clean_data()

    print("Preparing classification data...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Training Logistic Regression...")
    log_model = train_logistic_regression(X_train, y_train)

    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)

    print("Running K-Fold validation...")
    X, y = X_train.append(X_test), y_train.append(y_test)

    log_auc_mean, log_auc_std, rf_auc_mean, rf_auc_std = aplicar_kfold(X, y)

    print("Logistic AUC:", log_auc_mean)
    print("Random Forest AUC:", rf_auc_mean)

    print("Preparing time series...")

    series = prepare_time_series(df)

    train, test = train_test_split_time_series(series)

    print("Running ARIMA...")
    arima_model, arima_forecast, arima_rmse, arima_mae = run_arima(train, test)

    print("Running Holt-Winters...")
    hw_model, hw_forecast, hw_rmse, hw_mae = run_holt_winters(train, test)

    print("ARIMA RMSE:", arima_rmse)
    print("Holt-Winters RMSE:", hw_rmse)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()