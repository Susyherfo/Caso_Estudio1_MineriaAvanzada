"""
modulo de evaluacion
contiene funciones para auc, roc y validacion cruzada
"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score


def compute_auc(model, X_test, y_test):
    """
    calcula el area bajo la curva roc
    """

    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)

    return auc, y_prob


def plot_roc_curve(y_test, y_prob):
    """
    grafica curva roc
    """

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("curva roc")
    plt.show()


def cross_validate_auc(model, X, y):
    """
    realiza validacion cruzada con 5 folds usando auc
    """

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring='roc_auc'
    )

    return scores.mean()
