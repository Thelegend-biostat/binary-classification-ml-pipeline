
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize target to 0/1 (matches your R work)
    # R report uses: Negative -> 0, Positive -> 1 :contentReference[oaicite:1]{index=1}
    df[target] = df[target].map({"Negative": 0, "Positive": 1}).astype(int)

    # Convert Yes/No to 1/0 for all object cols except target
    for col in df.columns:
        if col == target:
            continue
        if df[col].dtype == "object":
            df[col] = df[col].map({"No": 0, "Yes": 1, "Male": 1, "Female": 0}).fillna(df[col])

    return df

def split_data(df: pd.DataFrame, target: str, test_size: float, random_state: int, stratify: bool = True):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None
    )
    