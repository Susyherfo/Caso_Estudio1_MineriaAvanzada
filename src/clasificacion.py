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

    return df


def split_data(df: pd.DataFrame):
    """
    separa variables predictoras y variable objetivo
    divide en entrenamiento y prueba
    """

    # seleccionar variables predictoras
    X = df[['voltage']]


    # variable objetivo
    y = df['high_consumption']

    # dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y  # mantiene proporción de clases
    )

    return X_train, X_test, y_train, y_test


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

