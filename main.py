# ==========================
# IMPORTS
# ==========================

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from src.preprocesamiento import load_and_clean_data

from src.clasificacion import (
    create_binary_target,
    split_data,
    train_logistic_regression,
    train_random_forest
)

from src.evaluacion import compute_auc

from src.series_temporales import (
    prepare_time_series,
    train_test_split_time_series,
    run_arima,
    run_holt_winters,
    compare_models
)


# ==========================
# MAIN
# ==========================

def main():

    # -----------------------------
    # CARGA Y PREPROCESAMIENTO
    # -----------------------------

    df = load_and_clean_data(
        "data/energy.csv",
        sample_size=50000,
        preserve_time_order=True
    )

    df = create_binary_target(df)

    # -----------------------------
    # CLASIFICACION
    # -----------------------------

    X_train, X_test, y_train, y_test = split_data(df)

    log_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    log_auc, log_prob = compute_auc(log_model, X_test, y_test)
    rf_auc, rf_prob = compute_auc(rf_model, X_test, y_test)

    print(f"\nLogistic Regression AUC: {log_auc:.4f}")
    print(f"Random Forest AUC: {rf_auc:.4f}")

    # -----------------------------
    # CURVA ROC
    # -----------------------------

    fpr_log, tpr_log, _ = roc_curve(y_test, log_prob)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)

    plt.figure()
    plt.plot(fpr_log, tpr_log, label=f"Logistic AUC = {log_auc:.4f}")
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest AUC = {rf_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Comparacion Curva ROC")
    plt.legend()
    plt.show()

    # -----------------------------
    # SERIES DE TIEMPO
    # -----------------------------

    print("\nIniciando modelos de series de tiempo...")

    series = prepare_time_series(df)
    train_ts, test_ts = train_test_split_time_series(series)

    # ARIMA
    arima_model, arima_forecast, arima_rmse, arima_mae = run_arima(
        train_ts,
        test_ts
    )

    # HOLT-WINTERS
    hw_model, hw_forecast, hw_rmse, hw_mae = run_holt_winters(
        train_ts,
        test_ts
    )

    print("\n--- RESULTADOS SERIES DE TIEMPO ---")

    print("\nARIMA")
    print(f"RMSE: {arima_rmse:.4f}")
    print(f"MAE: {arima_mae:.4f}")

    print("\nHOLT-WINTERS")
    print(f"RMSE: {hw_rmse:.4f}")
    print(f"MAE: {hw_mae:.4f}")

    # Comparacion
    decision = compare_models(
        (arima_rmse, arima_mae),
        (hw_rmse, hw_mae)
    )

    print(f"\n{decision}")


# ==========================
# ENTRY POINT
# ==========================

if __name__ == "__main__":
    main()
