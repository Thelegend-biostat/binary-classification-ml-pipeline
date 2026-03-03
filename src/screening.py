
import os
import json
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind


def screen_features_optional(
    X_train,
    y_train,
    X_test,
    do_screening: bool = False,
    alpha: float = 0.25,
    out_dir: str | None = None
):
    """
    Optional feature screening step.

    - If do_screening=False: returns X_train, X_test unchanged + metadata.
    - If do_screening=True:
        * numeric with >2 unique values: Welch two-sample t-test
        * binary/categorical: chi-square test of independence
      Keep features with p < alpha.

    Returns:
        X_train_screened, X_test_screened, info_dict
    """
    info = {
        "screening_enabled": bool(do_screening),
        "alpha": float(alpha),
        "kept_features": list(X_train.columns),
        "dropped_features": [],
    }

    if not do_screening:
        return X_train, X_test, info

    keep_cols = []
    dropped_cols = []

    for col in X_train.columns:
        x = X_train[col]

        # Numeric-like
        if pd.api.types.is_numeric_dtype(x) and x.nunique() > 2:
            g0 = x[y_train == 0]
            g1 = x[y_train == 1]
            _, p = ttest_ind(g0, g1, equal_var=False, nan_policy="omit")
        else:
            # Treat as categorical/binary
            tab = pd.crosstab(y_train, x)
            if tab.shape[0] < 2 or tab.shape[1] < 2:
                # Degenerate table -> keep (safe)
                p = 0.0
            else:
                _, p, _, _ = chi2_contingency(tab)

        if p < alpha:
            keep_cols.append(col)
        else:
            dropped_cols.append(col)

    # Safety: if screening removes everything, revert
    if len(keep_cols) == 0:
        info["kept_features"] = list(X_train.columns)
        info["dropped_features"] = []
        return X_train, X_test, info

    info["kept_features"] = keep_cols
    info["dropped_features"] = dropped_cols

    Xtr = X_train[keep_cols].copy()
    Xte = X_test[keep_cols].copy()

    # Optional: save screening info for reproducibility
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "screening.json"), "w") as f:
            json.dump(info, f, indent=4)

    return Xtr, Xte, info