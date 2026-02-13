"""
Modulo de series temporales

Este modulo implementa modelos clasicos de forecasting:
- ARIMA
- Holt-Winters

Se realiza benchmarking comparativo utilizando metricas
de error como RMSE y MAE.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# --------------------------------------------------
# 1️⃣ Preparacion de la serie temporal
# --------------------------------------------------

def prepare_time_series(df: pd.DataFrame,
                        column: str = "global_active_power",
                        freq: str = "D") -> pd.Series:
    """
    Agrega la serie temporal a una frecuencia especifica.

    Se utiliza agregacion diaria para:
    - Reducir ruido de alta frecuencia
    - Capturar patrones estructurales
    - Facilitar modelado estacional
    """

    # agregacion por frecuencia
    series = df[column].resample(freq).mean()

    # eliminar valores faltantes posteriores a la agregacion
    series.dropna(inplace=True)

    return series


# --------------------------------------------------
# 2️⃣ Division temporal train/test
# --------------------------------------------------

def train_test_split_time_series(series: pd.Series,
                                 test_size: float = 0.2):
    """
    Divide la serie respetando el orden cronologico.

    No se utiliza division aleatoria porque:
    - Se violaria la estructura temporal
    - Se generaria leakage de informacion
    """

    split_index = int(len(series) * (1 - test_size))

    train = series.iloc[:split_index]
    test = series.iloc[split_index:]

    return train, test


# --------------------------------------------------
# 3️⃣ Modelo ARIMA
# --------------------------------------------------

def run_arima(train: pd.Series,
              test: pd.Series,
              order=(2, 1, 2)):
    """
    Implementacion de modelo ARIMA.

    p: orden autoregresivo
    d: diferenciacion
    q: media movil

    Se utiliza una configuracion inicial (2,1,2)
    como punto de partida basado en literatura
    para series energeticas con tendencia moderada.
    """

    model = ARIMA(train, order=order)

    model_fit = model.fit()

    # forecasting fuera de muestra
    forecast = model_fit.forecast(steps=len(test))

    # metricas de evaluacion
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)

    return model_fit, forecast, rmse, mae


# --------------------------------------------------
# 4️⃣ Modelo Holt-Winters
# --------------------------------------------------

def run_holt_winters(train: pd.Series,
                     test: pd.Series,
                     seasonal_periods=7):
    """
    Implementacion de modelo Holt-Winters aditivo.

    Se asume:
    - Tendencia aditiva
    - Estacionalidad semanal (7 dias)

    Este modelo es adecuado cuando:
    - Existen patrones repetitivos
    - La estacionalidad es relativamente estable
    """

    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=seasonal_periods
    )

    model_fit = model.fit()

    forecast = model_fit.forecast(len(test))

    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)

    return model_fit, forecast, rmse, mae


# --------------------------------------------------
# 5️⃣ Benchmarking comparativo
# --------------------------------------------------

def compare_models(arima_metrics, hw_metrics):
    """
    Compara desempeño de modelos mediante RMSE.
    Retorna un string con la conclusion.
    """

    arima_rmse, arima_mae = arima_metrics
    hw_rmse, hw_mae = hw_metrics

    if arima_rmse < hw_rmse:
        return "ARIMA presenta menor error cuadratico medio."
    else:
        return "Holt-Winters presenta menor error cuadratico medio."
