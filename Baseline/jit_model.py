"""
jit_model.py

Unified JIT Defect Prediction Model (Black-box Core)
---------------------------------------------------

This module defines a *single*, reproducible JIT defect prediction model
shared by all explanation methods (LIME, PyExplainer, CfExplainer, XMAS-CQP).

Key design principles:
- model-centric (black-box)
- feature-visible only
- explanation-agnostic
- reproducible by construction
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Union

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# ============================================================
# Global config
# ============================================================

RANDOM_SEED = 42

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

SUPPORTED_PROJECTS = {"openstack", "qt", "activemq"}


# ============================================================
# Dataset loading
# ============================================================

def load_dataset(csv_path: Path):
    """
    Load JIT dataset and extract (X, y).

    Label priority:
    1. buggy
    2. bugcount > 0
    3. RealBug
    """
    df = pd.read_csv(csv_path)

    # ---------- label ----------
    if "buggy" in df.columns:
        y = df["buggy"].astype(int)
    elif "bugcount" in df.columns:
        y = (df["bugcount"] > 0).astype(int)
    elif "RealBug" in df.columns:
        y = df["RealBug"].astype(int)
    else:
        raise ValueError(
            "No valid defect label found "
            "(expected: buggy | bugcount | RealBug)"
        )

    # ---------- features ----------
    drop_cols = {
        "commit_id", "author_date",
        "buggy", "bugcount", "fixcount",
        "RealBug", "RealBugCount",
        "HeuBug", "HeuBugCount",
        "label", "File"
    }

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # enforce numeric-only, deterministic
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    return X, y


# ============================================================
# Model definition
# ============================================================

def build_model() -> RandomForestClassifier:
    """
    Build a stable JIT classifier.

    Note:
    - No scaling (tree-based)
    - Fixed random seed
    - Conservative hyperparameters to reduce variance
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )


# ============================================================
# Training
# ============================================================

def train_and_save_model(
    project: str,
    csv_path: Path,
    overwrite: bool = False
):
    assert project in SUPPORTED_PROJECTS, f"Unsupported project: {project}"

    model_path = MODEL_DIR / f"{project}_jit_model.joblib"
    if model_path.exists() and not overwrite:
        print(f"[jit_model] Model already exists: {model_path}")
        return model_path

    print(f"[jit_model] Training JIT model for project: {project}")

    X, y = load_dataset(csv_path)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_SEED
    )

    model = build_model()
    model.fit(X_train, y_train)

    acc = model.score(X_val, y_val)
    print(f"[jit_model] Validation accuracy: {acc:.4f}")

    joblib.dump(
        {
            "model": model,
            "feature_columns": list(X.columns),
            "random_seed": RANDOM_SEED,
        },
        model_path
    )

    print(f"[jit_model] Saved model to {model_path}")
    return model_path


# ============================================================
# Loading
# ============================================================

def load_model_bundle(project: str):
    model_path = MODEL_DIR / f"{project}_jit_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            f"Please train it first."
        )
    return joblib.load(model_path)


# ============================================================
# Black-box wrapper (for explainers)
# ============================================================

class JITModelWrapper:
    """
    Black-box interface exposed to explanation methods.

    Guarantees:
    - fixed feature order
    - accepts ndarray / dict / DataFrame
    - exposes predict / predict_proba only
    """

    def __init__(self, project: str):
        bundle = load_model_bundle(project)
        self.model = bundle["model"]
        self.feature_columns: List[str] = bundle["feature_columns"]

    def _to_dataframe(
        self,
        X: Union[pd.DataFrame, np.ndarray, dict, List[dict]]
    ) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=self.feature_columns)
        elif isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, list):
            df = pd.DataFrame(X)
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

        df = df[self.feature_columns]
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        return df

    def predict(self, X):
        df = self._to_dataframe(X)
        return self.model.predict(df)

    def predict_proba(self, X):
        df = self._to_dataframe(X)
        return self.model.predict_proba(df)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    train_and_save_model(
        project=args.project,
        csv_path=Path(args.csv),
        overwrite=args.overwrite
    )
