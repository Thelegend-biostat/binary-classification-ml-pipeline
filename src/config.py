from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Config:
    # -----------------------------
    # Data
    # -----------------------------
    data_path: str = "data/diabetes_data_upload.csv"
    target_col: str = "class"

    # -----------------------------
    # Split / Reproducibility
    # -----------------------------
    test_size: float = 1/3
    random_state: int = 7
    cv_folds: int = 10

    # -----------------------------
    # Feature Screening (optional)
    # -----------------------------
    do_screening: bool = False
    screen_alpha: float = 0.25

    # -----------------------------
    # Regularization Grid (L1)
    # -----------------------------
    C_grid: tuple = tuple(np.logspace(-3, 3, 60))

    # -----------------------------
    # Model Comparison
    # -----------------------------
    run_baseline: bool = True  # Toggle L1 vs baseline comparison

    # -----------------------------
    # Reporting / Artifacts
    # -----------------------------
    out_dir: str = "reports"
    top_k_features: int = 20