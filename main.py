import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.metrics import mean_squared_error, mean_absolute_error


# -----------------------------------------------------
# CARGA DE DATOS
# -----------------------------------------------------
def load_data():

    df = pd.read_csv(
        "data/energy.csv",
        sep=";",
        na_values="?"
    )

    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S"
    )

    numeric_cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df = df.dropna()

    df["consumption"] = df["Global_active_power"]

    threshold = df["consumption"].quantile(0.75)
    df["high_consumption"] = (df["consumption"] > threshold).astype(int)

    df = df.sort_values("datetime")

    return df


# -----------------------------------------------------
# ENTRENAMIENTO MODELOS CLASIFICACIÓN
# -----------------------------------------------------
def train_classification_models(df):

    features = [
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]

    X = df[features]
    y = df["high_consumption"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    results = []

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    lr_probs = lr.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_probs)

    results.append({
        "Model": "Logistic Regression",
        "AUC": lr_auc
    })

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    rf.fit(X_train, y_train)

    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_probs)

    results.append({
        "Model": "Random Forest",
        "AUC": rf_auc
    })

    results_df = pd.DataFrame(results)

    return results_df, rf, X_test, y_test


# -----------------------------------------------------
# BENCHMARK MODELOS
# -----------------------------------------------------
def run_models(df):

    results_df, _, _, _ = train_classification_models(df)

    return results_df


# -----------------------------------------------------
# CROSS VALIDATION
# -----------------------------------------------------
def cross_validation_models(df):

    features = [
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]

    X = df[features]
    y = df["high_consumption"]

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    lr_scores = cross_val_score(
        lr, X, y,
        cv=kfold,
        scoring="roc_auc"
    )

    rf_scores = cross_val_score(
        rf, X, y,
        cv=kfold,
        scoring="roc_auc"
    )

    results = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Mean AUC": [lr_scores.mean(), rf_scores.mean()],
        "Std AUC": [lr_scores.std(), rf_scores.std()]
    })

    return results


# -----------------------------------------------------
# ROC CURVE
# -----------------------------------------------------
def generate_roc_curve(df):

    _, model, X_test, y_test = train_classification_models(df)

    probs = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, probs)

    roc_data = pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr
    })

    return roc_data


# -----------------------------------------------------
# SERIES TEMPORALES
# -----------------------------------------------------
def prepare_time_series(df):

    ts = df.set_index("datetime")["consumption"]

    ts = ts.resample("D").mean()

    ts = ts.dropna()

    return ts


def train_arima(ts):

    model = ARIMA(ts, order=(2, 1, 2))

    fitted = model.fit()

    forecast = fitted.forecast(30)

    return forecast


def train_holt_winters(ts):

    model = ExponentialSmoothing(
        ts,
        trend="add"
    )

    fitted = model.fit()

    forecast = fitted.forecast(30)

    return forecast


# -----------------------------------------------------
# BENCHMARK SERIES TEMPORALES
# -----------------------------------------------------
def benchmark_timeseries(df):

    ts = prepare_time_series(df)

    train = ts[:-30]
    test = ts[-30:]

    arima_model = ARIMA(train, order=(2, 1, 2)).fit()
    arima_pred = arima_model.forecast(30)

    hw_model = ExponentialSmoothing(train, trend="add").fit()
    hw_pred = hw_model.forecast(30)

    arima_rmse = np.sqrt(mean_squared_error(test, arima_pred))
    hw_rmse = np.sqrt(mean_squared_error(test, hw_pred))

    arima_mae = mean_absolute_error(test, arima_pred)
    hw_mae = mean_absolute_error(test, hw_pred)

    results = pd.DataFrame({
        "Model": ["ARIMA", "Holt-Winters"],
        "RMSE": [arima_rmse, hw_rmse],
        "MAE": [arima_mae, hw_mae]
    })

    return results


# -----------------------------------------------------
# GENERAR PREDICCIONES
# -----------------------------------------------------
def make_predictions(model_name, df):

    ts = prepare_time_series(df)

    if model_name == "ARIMA":

        forecast = train_arima(ts)

    elif model_name == "Holt-Winters":

        forecast = train_holt_winters(ts)

    else:

        forecast = ts.rolling(5).mean().dropna()

    predictions = pd.DataFrame({
        "datetime": ts.index,
        "real": ts.values
    })

    predictions["predicted"] = predictions["real"].rolling(5).mean()

    return predictions