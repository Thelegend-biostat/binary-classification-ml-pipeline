
import os

from src.config import Config
from src.data import load_data, split_data
from src.screening import screen_features_optional
from src.train import train_l1_logistic, train_unregularized_logistic
from src.evaluate import evaluate_model, export_coefficients_and_plot


def main():
    cfg = Config()

    # -----------------------------
    # Load + split
    # -----------------------------
    df = load_data(cfg.data_path, target=cfg.target_col)
    X_train, X_test, y_train, y_test = split_data(
        df,
        target=cfg.target_col,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=True,
    )

    # -----------------------------
    # Optional screening
    # Save screening info once (separate from model outputs)
    # -----------------------------
    screening_out = os.path.join(cfg.out_dir, "screening")
    X_train, X_test, screen_info = screen_features_optional(
        X_train, y_train, X_test,
        do_screening=cfg.do_screening,
        alpha=cfg.screen_alpha,
        out_dir=screening_out
    )

    # -----------------------------
    # Output dirs
    # -----------------------------
    lasso_out = os.path.join(cfg.out_dir, "lasso")
    base_out = os.path.join(cfg.out_dir, "baseline")

    # -----------------------------
    # Train + eval: LASSO (L1)
    # -----------------------------
    lasso_model, lasso_thr, lasso_cv_auc = train_l1_logistic(
        X_train, y_train,
        C_grid=cfg.C_grid,
        cv=cfg.cv_folds,
        random_state=cfg.random_state
    )
    lasso_metrics = evaluate_model(lasso_model, X_test, y_test, threshold=lasso_thr, out_dir=lasso_out)
    export_coefficients_and_plot(lasso_model, X_train.columns, out_dir=lasso_out, top_k=cfg.top_k_features)

    # -----------------------------
    # Train + eval: Baseline (no penalty)
    # -----------------------------
    base_metrics = None
    base_thr = None

    if cfg.run_baseline:
        base_model, base_thr, base_train_auc = train_unregularized_logistic(
            X_train, y_train,
            random_state=cfg.random_state
        )
        base_metrics = evaluate_model(base_model, X_test, y_test, threshold=base_thr, out_dir=base_out)
        export_coefficients_and_plot(base_model, X_train.columns, out_dir=base_out, top_k=cfg.top_k_features)

    # -----------------------------
    # Print summaries
    # -----------------------------
    print("\n=== Data / Screening ===")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Screening enabled: {screen_info['screening_enabled']} | alpha: {screen_info['alpha']}")
    print(f"Features kept: {len(screen_info['kept_features'])} | dropped: {len(screen_info['dropped_features'])}")

    print("\n=== LASSO (L1) Logistic ===")
    print(f"CV ROC-AUC: {lasso_cv_auc:.4f}")
    print(f"Chosen threshold: {lasso_thr:.3f}")
    print(f"Test ROC-AUC: {lasso_metrics['test_roc_auc']:.4f} | F1: {lasso_metrics['f1']:.4f}")

    if cfg.run_baseline and base_metrics is not None:
        print("\n=== Baseline (No Regularization) Logistic ===")
        print(f"Chosen threshold: {base_thr:.3f}")
        print(f"Test ROC-AUC: {base_metrics['test_roc_auc']:.4f} | F1: {base_metrics['f1']:.4f}")

        print("\n=== Model Comparison (Test Set) ===")
        print(f"LASSO  ROC-AUC: {lasso_metrics['test_roc_auc']:.4f} | F1: {lasso_metrics['f1']:.4f}")
        print(f"BASE   ROC-AUC: {base_metrics['test_roc_auc']:.4f} | F1: {base_metrics['f1']:.4f}")

    print("\nArtifacts saved under:")
    print(f" - {lasso_out}/")
    if cfg.run_baseline:
        print(f" - {base_out}/")
    if cfg.do_screening:
        print(f" - {screening_out}/ (screening.json)")


if __name__ == "__main__":
    main()