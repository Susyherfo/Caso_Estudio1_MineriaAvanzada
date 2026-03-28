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

# Agregar importaciones para hiperparametrización 
from src.Hiperparametrizacion import ModelEvaluator
from sklearn.metrics import r2_score, mean_squared_error
import joblib


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

    # ===============================
    # HIPERPARAMETRIZACION (NUEVO)
    # ===============================
    print("\nRunning Hyperparameter Optimization...")

    evaluator = ModelEvaluator(X_train, X_test, y_train, y_test)

    print("\nGenetic Search...")
    genetic_results = evaluator.genetic_search()

    print("\nExhaustive Search...")
    grid_results = evaluator.exhaustive_search()

    print("\nBest parameters (Genetic):")
    for name, res in genetic_results.items():
        print(name, res['best_params'])

    print("\nEvaluating best models...")
    best_model_name = None
    best_score = -999

    for name, res in genetic_results.items():
        model = res['estimator']
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        print(f"\n{name}")
        print("R2:", r2)
        print("RMSE:", rmse)

        if r2 > best_score:
            best_score = r2
            best_model_name = name
            best_model = model

    print("\nBest overall model:", best_model_name)

    # Guardar mejor modelo
    joblib.dump(best_model, "best_model.pkl")
    print("Best model saved as best_model.pkl")

    print("\nComparison Genetic vs Grid")
    for model in genetic_results.keys():
        print(f"\n{model}")
        print("Genetic:", genetic_results[model]['best_params'])
        print("Grid:", grid_results[model]['best_params'])

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()