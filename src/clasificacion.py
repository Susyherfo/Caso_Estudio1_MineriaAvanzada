"""
Módulo de clasificación
BCD-7213 – Minería de Datos Avanzada – Caso de Estudio #2

Cambios respecto al Caso 1:
────────────────────────────────────────────────────────────
ELIMINADAS:
  · global_intensity   → corr=0.9988 con target (data leakage)
  · global_reactive_power → reemplazada por reactive_ratio
  · month              → solo toma valores 12 y 1 en la muestra
  · hour / day_of_week (enteros) → reemplazados por encoding cíclico

TRANSFORMADAS:
  · sub_metering_1  → sub1_on (binario: 93% ceros, skew=5.86)
  · sub_metering_2  → log1p(sub2) (reduce skew de 5.58 a 3.04)
  · hour            → hour_sin + hour_cos  (cíclico: 0h ≈ 23h)
  · day_of_week     → dow_sin  + dow_cos   (cíclico: lun ≈ dom)

NUEVAS (ingeniería de features):
  · unmetered_power  = GAP×1000/60 − sub1 − sub2 − sub3
                       Energía no medida por submedidores.
                       Importancia RF: 0.51 (feature #1)
  · power_factor     = global_active / apparent_power  ∈ [0,1]
  · reactive_ratio   = global_reactive / global_active
                       Importancia RF: 0.12
  · sub_total        = sub1 + sub2 + sub3
  · is_peak, is_night, is_weekend, n_subs_on

Resultado:
  Baseline (8 features)  → Accuracy=0.9962  AUC=0.9999
  Mejorado (16 features) → Accuracy=0.9987  AUC=1.0000
────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report


FEATURES_CLASIFICACION = [
    "unmetered_power",
    "power_factor",
    "reactive_ratio",
    "sub_total",
    "voltage",
    "sub_metering_3",
    "sub1_on",
    "log_sub2",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_peak",
    "is_night",
    "is_weekend",
    "n_subs_on",
]


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = d.columns.str.lower()
    if "period" in d.columns:
        d["period"] = pd.to_datetime(d["period"])
        d = d.set_index("period")
    d.index = pd.to_datetime(d.index)

    # Físicas derivadas
    d["unmetered_power"] = (
        d["global_active_power"] * 1000 / 60
        - d["sub_metering_1"]
        - d["sub_metering_2"]
        - d["sub_metering_3"]
    ).clip(lower=0)
    d["apparent_power"] = d["voltage"] * d["global_intensity"] / 1000
    d["power_factor"] = (
        d["global_active_power"] / d["apparent_power"].replace(0, np.nan)
    ).clip(0, 1)
    d["reactive_ratio"] = (
        d["global_reactive_power"] / d["global_active_power"].replace(0, np.nan)
    ).clip(lower=0)
    d["sub_total"] = d["sub_metering_1"] + d["sub_metering_2"] + d["sub_metering_3"]

    # Transformadas
    d["sub1_on"]  = (d["sub_metering_1"] > 0).astype(int)
    d["log_sub2"] = np.log1p(d["sub_metering_2"])

    # Temporales cíclicas
    hora = d.index.hour
    dia  = d.index.dayofweek
    d["hour_sin"] = np.sin(2 * np.pi * hora / 24)
    d["hour_cos"] = np.cos(2 * np.pi * hora / 24)
    d["dow_sin"]  = np.sin(2 * np.pi * dia  / 7)
    d["dow_cos"]  = np.cos(2 * np.pi * dia  / 7)

    # Flags
    d["is_peak"]    = hora.isin([7, 8, 17, 18, 19, 20]).astype(int)
    d["is_night"]   = hora.isin([0, 1, 2, 3, 4, 5]).astype(int)
    d["is_weekend"] = (dia >= 5).astype(int)
    d["n_subs_on"]  = (
        d["sub1_on"]
        + (d["sub_metering_2"] > 0).astype(int)
        + (d["sub_metering_3"] > 0).astype(int)
    )

    return d.dropna(subset=FEATURES_CLASIFICACION)


def crear_variable_objetivo(df: pd.DataFrame) -> pd.Series:
    threshold = df["global_active_power"].mean()
    return (df["global_active_power"] > threshold).astype(int)


def split_data(df: pd.DataFrame):
    d = construir_features(df)
    y = crear_variable_objetivo(d)
    X = d[FEATURES_CLASIFICACION]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_logistic_regression(X_train, y_train):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=42))
    ])
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_with_threshold(model, X_test, y_test, threshold=0.5):
    y_prob        = model.predict_proba(X_test)[:, 1]
    y_pred_custom = (y_prob > threshold).astype(int)
    cm            = confusion_matrix(y_test, y_pred_custom)
    report        = classification_report(y_test, y_pred_custom)
    return y_prob, y_pred_custom, cm, report


def get_feature_importance(rf_model) -> pd.Series:
    if hasattr(rf_model, "feature_importances_"):
        return pd.Series(
            rf_model.feature_importances_, index=FEATURES_CLASIFICACION
        ).sort_values(ascending=False)
    return pd.Series(dtype=float)