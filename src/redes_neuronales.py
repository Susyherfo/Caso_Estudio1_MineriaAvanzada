"""
Módulo de Redes Neuronales
BCD-7213 – Minería de Datos Avanzada – Caso de Estudio #2

Las 5 redes usan el mismo feature engineering de clasificacion.py:
  · unmetered_power, power_factor, reactive_ratio, sub_total
  · voltage, sub_metering_3, sub1_on, log_sub2
  · hour_sin/cos, dow_sin/cos, is_peak, is_night, is_weekend, n_subs_on

Cambios respecto a la versión anterior:
  · global_intensity eliminada (leakage)
  · hour/dow como enteros reemplazados por encoding cíclico
  · sub_metering_1 convertido a binario (sub1_on)
  · sub_metering_2 transformado con log1p
  · Regresión profunda ahora predice global_active_power con
    features ortogonales (sin apparent_power para evitar redundancia)
  · LSTM y CNN1D usan frecuencia 'h' (pandas >= 2.2)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, mean_squared_error,
    mean_absolute_error, classification_report
)
import warnings
warnings.filterwarnings("ignore")

from src.clasificacion import construir_features, FEATURES_CLASIFICACION


def _get_keras():
    try:
        import tensorflow as tf
        from tensorflow import keras
        return keras, tf
    except ImportError:
        raise ImportError("Instala TensorFlow: pip install tensorflow")


# ─────────────────────────────────────────────────────────────────
# Utilidades internas
# ─────────────────────────────────────────────────────────────────

def _preparar_X_y_clasificacion(df: pd.DataFrame):
    """Usa el feature engineering unificado de clasificacion.py."""
    d = construir_features(df)
    threshold = d["global_active_power"].mean()
    y = (d["global_active_power"] > threshold).astype(int).values
    X = d[FEATURES_CLASIFICACION].values
    return X, y


def _preparar_serie(df: pd.DataFrame,
                    col: str = "global_active_power",
                    freq: str = "h",
                    n_steps: int = 24):
    """
    Agrega la serie a frecuencia horaria y construye ventanas
    deslizantes de n_steps pasos para modelos secuenciales.
    """
    d = df.copy()
    d.columns = d.columns.str.lower()
    if "period" in d.columns:
        d["period"] = pd.to_datetime(d["period"])
        d = d.set_index("period")
    d.index = pd.to_datetime(d.index)

    serie = d[col].resample(freq).mean().dropna()
    scaler = MinMaxScaler()
    vals = scaler.fit_transform(serie.values.reshape(-1, 1))

    X, y = [], []
    for i in range(n_steps, len(vals)):
        X.append(vals[i - n_steps:i, 0])
        y.append(vals[i, 0])

    return np.array(X), np.array(y), scaler


# ─────────────────────────────────────────────────────────────────
# 1. MLP – Perceptrón Multicapa (clasificación)
# ─────────────────────────────────────────────────────────────────

def entrenar_mlp(df: pd.DataFrame, epochs: int = 15,
                 batch_size: int = 512):
    """
    MLP para clasificación binaria alto/bajo consumo.

    Input: 16 features (incluyendo unmetered_power, encoding cíclico, flags).
    Arquitectura:
        Dense(128, ReLU) → Dropout(0.3)
        Dense(64,  ReLU) → Dropout(0.2)
        Dense(1, Sigmoid)

    Mejora vs versión anterior:
    · Sin global_intensity (leakage eliminado)
    · hour/dow como sin/cos → gradiente más informativo
    · StandardScaler cubre rangos heterogéneos (Wh/min vs [0,1])
    """
    keras, _ = _get_keras()

    X, y = _preparar_X_y_clasificacion(df)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    split = int(len(X_sc) * 0.8)
    X_tr, X_te = X_sc[:split], X_sc[split:]
    y_tr, y_te = y[:split],    y[split:]

    model = keras.Sequential([
        keras.layers.Input(shape=(X_tr.shape[1],)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(X_tr, y_tr, epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.15, verbose=0)

    y_prob = model.predict(X_te, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    return {
        "model":    model,
        "history":  history.history,
        "accuracy": accuracy_score(y_te, y_pred),
        "report":   classification_report(y_te, y_pred, output_dict=True),
        "y_test":   y_te,
        "y_pred":   y_pred,
        "nombre":   "MLP (Clasificación)",
    }


# ─────────────────────────────────────────────────────────────────
# 2. LSTM – Series Temporales
# ─────────────────────────────────────────────────────────────────

def entrenar_lstm(df: pd.DataFrame, epochs: int = 10,
                  batch_size: int = 256, n_steps: int = 24):
    """
    LSTM para predicción del consumo horario siguiente.
    Usa ventanas de 24h sobre la serie agregada horariamente.

    Arquitectura:
        LSTM(64) → Dropout(0.2) → Dense(32, ReLU) → Dense(1)
    """
    keras, _ = _get_keras()

    X, y, scaler = _preparar_serie(df, n_steps=n_steps)
    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split].reshape(-1, n_steps, 1), X[split:].reshape(-1, n_steps, 1)
    y_tr, y_te = y[:split], y[split:]

    model = keras.Sequential([
        keras.layers.Input(shape=(n_steps, 1)),
        keras.layers.LSTM(64),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(X_tr, y_tr, epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.15, verbose=0)

    y_pred_sc   = model.predict(X_te, verbose=0).flatten()
    y_pred_real = scaler.inverse_transform(y_pred_sc.reshape(-1, 1)).flatten()
    y_test_real = scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()

    return {
        "model":   model,
        "history": history.history,
        "rmse":    float(np.sqrt(mean_squared_error(y_test_real, y_pred_real))),
        "mae":     float(mean_absolute_error(y_test_real, y_pred_real)),
        "y_test":  y_test_real,
        "y_pred":  y_pred_real,
        "nombre":  "LSTM (Series Temporales)",
    }


# ─────────────────────────────────────────────────────────────────
# 3. CNN 1D – Patrones en ventanas temporales
# ─────────────────────────────────────────────────────────────────

def entrenar_cnn1d(df: pd.DataFrame, epochs: int = 10,
                   batch_size: int = 512, n_steps: int = 24):
    """
    CNN 1D para clasificar ventanas de 24h como alto/bajo consumo.
    Los filtros convolucionales detectan patrones locales independientes
    de su posición en la ventana.

    Arquitectura:
        Conv1D(32, k=3) → MaxPool(2)
        Conv1D(64, k=3) → GlobalAvgPool
        Dense(32, ReLU) → Dense(1, Sigmoid)
    """
    keras, _ = _get_keras()

    X, y, _ = _preparar_serie(df, n_steps=n_steps)
    y_cls   = (y > np.median(y)).astype(int)

    split = int(len(X) * 0.8)
    X_tr = X[:split].reshape(-1, n_steps, 1)
    X_te = X[split:].reshape(-1, n_steps, 1)
    y_tr, y_te = y_cls[:split], y_cls[split:]

    model = keras.Sequential([
        keras.layers.Input(shape=(n_steps, 1)),
        keras.layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    history = model.fit(X_tr, y_tr, epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.15, verbose=0)

    y_prob = model.predict(X_te, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    return {
        "model":    model,
        "history":  history.history,
        "accuracy": accuracy_score(y_te, y_pred),
        "report":   classification_report(y_te, y_pred, output_dict=True),
        "y_test":   y_te,
        "y_pred":   y_pred,
        "nombre":   "CNN 1D (Ventanas Temporales)",
    }


# ─────────────────────────────────────────────────────────────────
# 4. Autoencoder – Detección de Anomalías
# ─────────────────────────────────────────────────────────────────

def entrenar_autoencoder(df: pd.DataFrame, epochs: int = 15,
                         batch_size: int = 512,
                         percentil_umbral: int = 95):
    """
    Autoencoder para detección no supervisada de anomalías.
    Aprende la distribución normal del consumo; registros con
    error de reconstrucción alto son marcados como anómalos.

    Input: mismas 16 features del módulo de clasificación.
    Arquitectura (simétrica):
        Encoder: Dense(64) → Dense(32) → Dense(16)
        Decoder: Dense(32) → Dense(64) → Dense(16)
    """
    keras, _ = _get_keras()

    X, _ = _preparar_X_y_clasificacion(df)
    scaler = MinMaxScaler()
    X_sc = scaler.fit_transform(X)

    split  = int(len(X_sc) * 0.8)
    X_tr, X_te = X_sc[:split], X_sc[split:]
    n = X_tr.shape[1]

    inp = keras.layers.Input(shape=(n,))
    enc = keras.layers.Dense(64, activation="relu")(inp)
    enc = keras.layers.Dense(32, activation="relu")(enc)
    enc = keras.layers.Dense(16, activation="relu")(enc)
    dec = keras.layers.Dense(32, activation="relu")(enc)
    dec = keras.layers.Dense(64, activation="relu")(dec)
    out = keras.layers.Dense(n)(dec)

    ae = keras.Model(inputs=inp, outputs=out)
    ae.compile(optimizer="adam", loss="mse")
    history = ae.fit(X_tr, X_tr, epochs=epochs,
                     batch_size=batch_size,
                     validation_split=0.15, verbose=0)

    X_pred    = ae.predict(X_te, verbose=0)
    recon_err = np.mean(np.power(X_te - X_pred, 2), axis=1)
    umbral    = np.percentile(recon_err, percentil_umbral)
    anomalias = (recon_err > umbral).astype(int)
    n_anom    = int(anomalias.sum())

    return {
        "model":                ae,
        "history":              history.history,
        "reconstruction_error": recon_err,
        "umbral":               float(umbral),
        "anomalias":            anomalias,
        "n_anomalias":          n_anom,
        "pct_anomalias":        n_anom / len(anomalias) * 100,
        "nombre":               "Autoencoder (Detección de Anomalías)",
    }


# ─────────────────────────────────────────────────────────────────
# 5. Red de Regresión Profunda
# ─────────────────────────────────────────────────────────────────

def entrenar_red_regresion(df: pd.DataFrame, epochs: int = 15,
                           batch_size: int = 512):
    """
    Red profunda para predecir el valor continuo de
    global_active_power.

    Cambio clave: se excluye apparent_power (redundante con
    power_factor + voltage) y global_intensity (leakage).
    Se usan solo las 16 features ortogonales del módulo unificado.

    Arquitectura:
        Dense(256, ReLU) → BatchNorm → Dropout(0.3)
        Dense(128, ReLU) → BatchNorm → Dropout(0.2)
        Dense(64,  ReLU) → Dense(1)
    """
    keras, _ = _get_keras()

    d = construir_features(df)
    X = d[FEATURES_CLASIFICACION].values
    y = d["global_active_power"].values

    scaler_X = StandardScaler()
    scaler_y = MinMaxScaler()
    X_sc = scaler_X.fit_transform(X)
    y_sc = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    split = int(len(X_sc) * 0.8)
    X_tr, X_te = X_sc[:split], X_sc[split:]
    y_tr, y_te = y_sc[:split], y_sc[split:]

    model = keras.Sequential([
        keras.layers.Input(shape=(X_tr.shape[1],)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse", metrics=["mae"]
    )
    history = model.fit(X_tr, y_tr, epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.15, verbose=0)

    y_pred_sc   = model.predict(X_te, verbose=0).flatten()
    y_pred_real = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).flatten()
    y_test_real = scaler_y.inverse_transform(y_te.reshape(-1, 1)).flatten()

    ss_res = np.sum((y_test_real - y_pred_real) ** 2)
    ss_tot = np.sum((y_test_real - y_test_real.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "model":   model,
        "history": history.history,
        "rmse":    float(np.sqrt(mean_squared_error(y_test_real, y_pred_real))),
        "mae":     float(mean_absolute_error(y_test_real, y_pred_real)),
        "r2":      float(r2),
        "y_test":  y_test_real,
        "y_pred":  y_pred_real,
        "nombre":  "Red Profunda (Regresión)",
    }