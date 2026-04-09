"""
Módulo de Redes Neuronales
BCD-7213 – Minería de Datos Avanzada – Caso de Estudio #2

Implementa 5 tipos de redes neuronales sobre el dataset de consumo energético:

1. MLP  – Clasificación binaria (alto/bajo consumo)
2. LSTM – Predicción de series temporales
3. CNN1D – Clasificación sobre ventanas temporales
4. Autoencoder – Detección de anomalías en consumo
5. Red de Regresión Profunda – Predicción de valor continuo
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    accuracy_score, mean_squared_error,
    mean_absolute_error, classification_report
)
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Utilidades compartidas
# ─────────────────────────────────────────────

def _get_keras():
    """Importa keras de forma lazy para no fallar en ambientes sin TF."""
    try:
        import tensorflow as tf
        from tensorflow import keras
        return keras, tf
    except ImportError:
        raise ImportError(
            "TensorFlow no está instalado. "
            "Ejecuta: pip install tensorflow"
        )


def _preparar_features_clasificacion(df: pd.DataFrame):
    """Prepara X, y para clasificación binaria."""
    df = df.copy()
    df.columns = df.columns.str.lower()

    if "period" in df.columns:
        df["period"] = pd.to_datetime(df["period"])
        df = df.set_index("period")

    threshold = df["global_active_power"].mean()
    df["high_consumption"] = (
        df["global_active_power"] > threshold
    ).astype(int)

    features = [
        "voltage", "global_intensity",
        "sub_metering_1", "sub_metering_2", "sub_metering_3"
    ]
    df["hour"]        = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"]       = df.index.month
    features += ["hour", "day_of_week", "month"]

    X = df[features].values
    y = df["high_consumption"].values
    return X, y


def _preparar_serie(df: pd.DataFrame, col="global_active_power",
                    freq="H", n_steps=24):
    """Agrega a frecuencia horaria y construye ventanas de n_steps."""
    df = df.copy()
    df.columns = df.columns.str.lower()

    if "period" in df.columns:
        df["period"] = pd.to_datetime(df["period"])
        df = df.set_index("period")

    serie = df[col].resample(freq).mean().dropna()

    scaler = MinMaxScaler()
    valores = scaler.fit_transform(serie.values.reshape(-1, 1))

    X, y = [], []
    for i in range(n_steps, len(valores)):
        X.append(valores[i - n_steps:i, 0])
        y.append(valores[i, 0])

    return np.array(X), np.array(y), scaler


# ─────────────────────────────────────────────
# 1. MLP – Perceptrón Multicapa
# ─────────────────────────────────────────────

def entrenar_mlp(df: pd.DataFrame, epochs=15, batch_size=512):
    """
    Red neuronal densa (MLP) para clasificación binaria.

    Arquitectura:
        Input → Dense(128, ReLU) → Dropout(0.3)
              → Dense(64, ReLU)  → Dropout(0.2)
              → Dense(1, Sigmoid)

    Objetivo: predecir si el consumo en un instante es alto o bajo.
    """
    keras, tf = _get_keras()

    X, y = _preparar_features_clasificacion(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    split = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]

    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        verbose=0
    )

    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred      = (y_pred_prob > 0.5).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "model": model,
        "history": history.history,
        "accuracy": acc,
        "report": report,
        "y_test": y_test,
        "y_pred": y_pred,
        "nombre": "MLP (Clasificación)"
    }


# ─────────────────────────────────────────────
# 2. LSTM – Series Temporales
# ─────────────────────────────────────────────

def entrenar_lstm(df: pd.DataFrame, epochs=10,
                  batch_size=256, n_steps=24):
    """
    Red LSTM para predicción de consumo energético futuro.

    Arquitectura:
        Input(24 pasos) → LSTM(64) → Dropout(0.2)
                        → Dense(32, ReLU)
                        → Dense(1, Linear)

    Predice el consumo horario siguiente dadas las 24h anteriores.
    """
    keras, tf = _get_keras()

    X, y, scaler = _preparar_serie(df, n_steps=n_steps)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # LSTM necesita shape (samples, timesteps, features)
    X_train = X_train.reshape(-1, n_steps, 1)
    X_test  = X_test.reshape(-1, n_steps, 1)

    model = keras.Sequential([
        keras.layers.Input(shape=(n_steps, 1)),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])

    model.compile(optimizer="adam", loss="mse")

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        verbose=0
    )

    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # Desnormalizar
    y_pred_real = scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).flatten()
    y_test_real = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mae  = mean_absolute_error(y_test_real, y_pred_real)

    return {
        "model": model,
        "history": history.history,
        "rmse": rmse,
        "mae": mae,
        "y_test": y_test_real,
        "y_pred": y_pred_real,
        "nombre": "LSTM (Series Temporales)"
    }


# ─────────────────────────────────────────────
# 3. CNN 1D – Clasificación sobre ventanas
# ─────────────────────────────────────────────

def entrenar_cnn1d(df: pd.DataFrame, epochs=10,
                   batch_size=512, n_steps=24):
    """
    Red CNN 1D para clasificación sobre ventanas temporales.

    Arquitectura:
        Input(24 pasos) → Conv1D(32, k=3, ReLU) → MaxPool(2)
                        → Conv1D(64, k=3, ReLU) → GlobalAvgPool
                        → Dense(32, ReLU) → Dense(1, Sigmoid)

    Detecta patrones locales en secuencias de consumo para
    clasificar si la ventana corresponde a un período de
    alto consumo.
    """
    keras, tf = _get_keras()

    X, y, scaler = _preparar_serie(df, n_steps=n_steps)

    # Etiqueta: si el valor objetivo supera la mediana → alto consumo
    y_cls = (y > np.median(y)).astype(int)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_cls[:split], y_cls[split:]

    X_train = X_train.reshape(-1, n_steps, 1)
    X_test  = X_test.reshape(-1, n_steps, 1)

    model = keras.Sequential([
        keras.layers.Input(shape=(n_steps, 1)),
        keras.layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        verbose=0
    )

    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred      = (y_pred_prob > 0.5).astype(int)

    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "model": model,
        "history": history.history,
        "accuracy": acc,
        "report": report,
        "y_test": y_test,
        "y_pred": y_pred,
        "nombre": "CNN 1D (Ventanas Temporales)"
    }


# ─────────────────────────────────────────────
# 4. Autoencoder – Detección de Anomalías
# ─────────────────────────────────────────────

def entrenar_autoencoder(df: pd.DataFrame, epochs=15,
                         batch_size=512, percentil_umbral=95):
    """
    Autoencoder para detección de anomalías de consumo.

    Arquitectura (simétrica):
        Encoder: Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU)
        Decoder: Dense(32, ReLU) → Dense(64, ReLU) → Dense(n, Linear)

    Los registros con error de reconstrucción alto se marcan
    como anomalías (consumo inusual).
    """
    keras, tf = _get_keras()

    X, _ = _preparar_features_clasificacion(df)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    split = int(len(X_scaled) * 0.8)
    X_train = X_scaled[:split]
    X_test  = X_scaled[split:]

    n_features = X_train.shape[1]

    inp = keras.layers.Input(shape=(n_features,))
    # Encoder
    enc = keras.layers.Dense(64, activation="relu")(inp)
    enc = keras.layers.Dense(32, activation="relu")(enc)
    enc = keras.layers.Dense(16, activation="relu")(enc)
    # Decoder
    dec = keras.layers.Dense(32, activation="relu")(enc)
    dec = keras.layers.Dense(64, activation="relu")(dec)
    out = keras.layers.Dense(n_features)(dec)

    autoencoder = keras.Model(inputs=inp, outputs=out)
    autoencoder.compile(optimizer="adam", loss="mse")

    history = autoencoder.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        verbose=0
    )

    # Error de reconstrucción en test
    X_pred     = autoencoder.predict(X_test, verbose=0)
    recon_err  = np.mean(np.power(X_test - X_pred, 2), axis=1)

    umbral     = np.percentile(recon_err, percentil_umbral)
    anomalias  = (recon_err > umbral).astype(int)
    n_anomalias = int(anomalias.sum())

    return {
        "model": autoencoder,
        "history": history.history,
        "reconstruction_error": recon_err,
        "umbral": umbral,
        "anomalias": anomalias,
        "n_anomalias": n_anomalias,
        "pct_anomalias": n_anomalias / len(anomalias) * 100,
        "nombre": "Autoencoder (Detección de Anomalías)"
    }


# ─────────────────────────────────────────────
# 5. Red de Regresión Profunda
# ─────────────────────────────────────────────

def entrenar_red_regresion(df: pd.DataFrame, epochs=15,
                           batch_size=512):
    """
    Red neuronal profunda para regresión continua.

    Objetivo: predecir el valor exacto de global_active_power
    a partir de las otras variables del dataset.

    Arquitectura:
        Input → Dense(256, ReLU) → BatchNorm → Dropout(0.3)
              → Dense(128, ReLU) → BatchNorm → Dropout(0.2)
              → Dense(64, ReLU)
              → Dense(1, Linear)
    """
    keras, tf = _get_keras()

    df = df.copy()
    df.columns = df.columns.str.lower()

    if "period" in df.columns:
        df["period"] = pd.to_datetime(df["period"])
        df = df.set_index("period")

    features = [
        "voltage", "global_intensity",
        "sub_metering_1", "sub_metering_2", "sub_metering_3"
    ]
    df["hour"]        = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"]       = df.index.month
    features += ["hour", "day_of_week", "month"]

    X = df[features].values
    y = df["global_active_power"].values

    scaler_X = StandardScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    split = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y_scaled[:split], y_scaled[split:]

    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
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
        loss="mse",
        metrics=["mae"]
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        verbose=0
    )

    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred_real   = scaler_y.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).flatten()
    y_test_real   = scaler_y.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mae  = mean_absolute_error(y_test_real, y_pred_real)
    r2   = 1 - (
        np.sum((y_test_real - y_pred_real) ** 2) /
        np.sum((y_test_real - y_test_real.mean()) ** 2)
    )

    return {
        "model": model,
        "history": history.history,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "y_test": y_test_real,
        "y_pred": y_pred_real,
        "nombre": "Red Profunda (Regresión)"
    }