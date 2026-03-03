

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
)


def evaluate_model(model, X_test, y_test, threshold: float, out_dir: str = "reports"):
    os.makedirs(out_dir, exist_ok=True)

    p = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p)

    yhat = (p >= threshold).astype(int)
    cm = confusion_matrix(y_test, yhat)

    metrics = {
        "test_roc_auc": float(auc),
        "accuracy": float(accuracy_score(y_test, yhat)),
        "precision": float(precision_score(y_test, yhat, zero_division=0)),
        "recall": float(recall_score(y_test, yhat, zero_division=0)),
        "f1": float(f1_score(y_test, yhat, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "threshold": float(threshold),
    }

    # ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC = {auc:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=160)
    plt.close()

    # Save metrics to JSON file
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def export_coefficients_and_plot(model, feature_names, out_dir="reports", top_k=20):
    """
    Export coefficients to CSV + plot top-k absolute coefficients.
    Assumes sklearn Pipeline with step name 'model' containing LogisticRegression.
    """
    os.makedirs(out_dir, exist_ok=True)

    lr = model.named_steps["model"]
    coefs = lr.coef_.ravel()

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs)
    }).sort_values("abs_coefficient", ascending=False)

    # Save full table
    coef_path = os.path.join(out_dir, "coefficients.csv")
    coef_df.to_csv(coef_path, index=False)

    # Plot top_k
    top = coef_df.head(top_k).iloc[::-1]  # reverse ordering for prettier barh
    plt.figure(figsize=(8, max(4, 0.28 * len(top))))
    plt.barh(top["feature"], top["abs_coefficient"])
    plt.xlabel("|coefficient| (standardized importance proxy)")
    plt.title(f"Top {top_k} Feature Importances (|Logistic Coef|)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=160)
    plt.close()

    return coef_path