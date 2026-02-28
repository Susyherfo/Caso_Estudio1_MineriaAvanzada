# src/k_fold.py

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def aplicar_kfold(X, y, n_splits=5, random_state=42):
    """
    Aplica validación cruzada K-Fold (estratificada)
    a Logistic Regression y Random Forest.

    Retorna:
        - promedio AUC Logistic
        - std AUC Logistic
        - promedio AUC RF
        - std AUC RF
    """

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    auc_logistic = []
    auc_rf = []

    for train_index, test_index in skf.split(X, y):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # -------------------------
        # Logistic Regression
        # -------------------------
        log_model = LogisticRegression(max_iter=1000)
        log_model.fit(X_train, y_train)

        y_pred_prob_log = log_model.predict_proba(X_test)[:, 1]
        auc_logistic.append(roc_auc_score(y_test, y_pred_prob_log))

        # -------------------------
        # Random Forest
        # -------------------------
        rf_model = RandomForestClassifier(random_state=random_state)
        rf_model.fit(X_train, y_train)

        y_pred_prob_rf = rf_model.predict_proba(X_test)[:, 1]
        auc_rf.append(roc_auc_score(y_test, y_pred_prob_rf))

    return (
        np.mean(auc_logistic),
        np.std(auc_logistic),
        np.mean(auc_rf),
        np.std(auc_rf)
    )