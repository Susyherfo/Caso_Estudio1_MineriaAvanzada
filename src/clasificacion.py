"""
modulo de clasificacion
contiene funciones para crear variable objetivo,
dividir datos y entrenar modelos
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    crea variable binaria:
    1 = alto consumo
    0 = bajo consumo

    el punto de corte se define como el promedio del consumo
    """

    # calcular promedio de consumo
    threshold = df['global_active_power'].mean()

    # crear nueva columna binaria
    df['high_consumption'] = (
        df['global_active_power'] > threshold
    ).astype(int)



# ---------------------------------------
# Feature Engineering Temporal
# ---------------------------------------

    df.index = pd.to_datetime(df.index)

    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    print(df.columns)

    return df

def preparar_features(df):

    df = df.copy()
    df.columns = df.columns.str.lower()

    # Crear variable objetivo
    threshold = df['global_active_power'].mean()
    df['high_consumption'] = (
        df['global_active_power'] > threshold
    ).astype(int)

    # Asegurar índice datetime
    df.index = pd.to_datetime(df.index)

    # Feature engineering
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    features = [
        'voltage',
        'global_intensity',
        'sub_metering_1',
        'sub_metering_2',
        'sub_metering_3',
        'hour',
        'day_of_week',
        'month'
    ]

    X = df[features]
    y = df['high_consumption']

    return X, y

def split_data(df):

    X, y = preparar_features(df)

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

def train_logistic_regression(X_train, y_train):
    """
    entrena modelo logistic regression
    """

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train):
    """
    entrena modelo random forest
    """

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def evaluate_with_threshold(model, X_test, y_test, threshold=0.5):
    """
    evalua el modelo usando un umbral personalizado
    """

    # obtener probabilidades
    y_prob = model.predict_proba(X_test)[:, 1]

    # aplicar umbral personalizado
    y_pred_custom = (y_prob > threshold).astype(int)

    # calcular matriz de confusion
    cm = confusion_matrix(y_test, y_pred_custom)

    # reporte de clasificacion
    report = classification_report(y_test, y_pred_custom)

    return y_prob, y_pred_custom, cm, report

