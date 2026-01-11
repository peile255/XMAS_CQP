"""
qt_model.py

Train a frozen JIT defect prediction model for the Qt project.
This model is used as a black-box predictor for RQ1 stability evaluation.

Output:
- models/qt_jit_model.joblib
"""

import joblib
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Config
# ============================================================

RANDOM_SEED = 42

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# Dataset loading
# ============================================================

def load_qt_dataset(csv_path: Path):
    """
    Load Qt JIT dataset.

    Label rule (JIT convention):
    - buggy = 1 if bugcount > 0
    - buggy = 0 otherwise
    """
    df = pd.read_csv(csv_path)

    if "bugcount" not in df.columns:
        raise ValueError("Qt dataset must contain 'bugcount' column")

    # ---------- label ----------
    y = (df["bugcount"] > 0).astype(int)

    # ---------- feature space ----------
    drop_cols = {
        "commit_id",
        "author_date",
        "bugcount",
        "fixcount",
    }

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # make sure everything is numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    return X, y


# ============================================================
# Model
# ============================================================

def build_model():
    """
    Deterministic JIT defect prediction model.

    NOTE:
    - We intentionally use a stable classifier
    - Explanation variance must NOT come from the model
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ))
    ])


# ============================================================
# Training
# ============================================================

def train_qt_model(csv_path: Path, overwrite: bool = False):
    model_path = MODEL_DIR / "qt_jit_model.joblib"

    if model_path.exists() and not overwrite:
        print(f"[qt_model] Model already exists: {model_path}")
        return

    print("[qt_model] Training JIT model for Qt")

    X, y = load_qt_dataset(csv_path)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    model = build_model()
    model.fit(X_train, y_train)

    acc = model.score(X_val, y_val)
    print(f"[qt_model] Validation accuracy: {acc:.4f}")

    bundle = {
        "model": model,
        "feature_columns": list(X.columns),
    }

    joblib.dump(bundle, model_path)
    print(f"[qt_model] Saved model to {model_path}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to qt.csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    train_qt_model(
        csv_path=Path(args.csv),
        overwrite=args.overwrite
    )
