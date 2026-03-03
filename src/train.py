
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator


def train_l1_logistic(
    X_train,
    y_train,
    C_grid=None,
    cv: int = 10,
    random_state: int = 7
) -> tuple[BaseEstimator, float, float]:
    """
    L1-regularized logistic regression with CV-tuned C (inverse regularization strength).
    Returns: best_model, best_threshold, cv_auc
    """
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=2000,
            random_state=random_state
        ))
    ])

    # Default grid if none provided
    if C_grid is None:
        C_grid = np.logspace(-3, 3, 60)

    param_grid = {"model__C": list(C_grid)}

    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv_split,
        n_jobs=-1
    )
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    cv_auc = gs.best_score_

    # Threshold chosen on training predictions (Youden's J)
    p_train = best_model.predict_proba(X_train)[:, 1]
    best_threshold = _best_threshold_youden(y_train, p_train)

    return best_model, best_threshold, float(cv_auc)


def train_unregularized_logistic(
    X_train,
    y_train,
    random_state: int = 7
) -> tuple[BaseEstimator, float, float]:
    """
    Baseline logistic regression without regularization.
    Returns: model, best_threshold, train_auc
    """
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=5000,
            random_state=random_state
        ))
    ])

    pipe.fit(X_train, y_train)

    p_train = pipe.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, p_train)

    best_threshold = _best_threshold_youden(y_train, p_train)

    return pipe, best_threshold, float(train_auc)


def _best_threshold_youden(y_true, p_hat) -> float:
    """
    Choose threshold that maximizes Youden's J = TPR - FPR.
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_j = 0.5, -1e9

    # Ensure numpy arrays for safe vector ops
    y_true = np.asarray(y_true)

    for t in thresholds:
        y_pred = (p_hat >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tpr = tp / (tp + fn + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, t

    return float(best_t)

