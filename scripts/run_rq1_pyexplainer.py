"""
run_rq1_pyexplainer.py

RQ1: Run-to-run stability evaluation for PyExplainer explanations
-----------------------------------------------------------------

This script generates PyExplainer explanations for the same set of
input instances across multiple random seeds, under an identical
frozen JIT defect prediction model.

Output structure:
results/
└── openstack/
    └── pyexplainer/
        ├── run_1/explanations.jsonl
        ├── run_2/explanations.jsonl
        ├── run_3/explanations.jsonl
        ├── run_4/explanations.jsonl
        └── run_5/explanations.jsonl
"""

import sys
import json
from pathlib import Path

# ------------------------------------------------------------
# Make project root importable
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from Baseline.jit_model import JITModelWrapper
from Baseline.baselines.pyexplainer import PyExplainer


# ============================================================
# Global config
# ============================================================

PROJECT = "qt"
N_RUNS = 5
BASE_SEED = 43
TOP_K = 5

INPUT_PATH = Path("data/rq1_samples/qt_rq1_input.jsonl")
OUTPUT_ROOT = Path("results/qt/pyexplainer")


# ============================================================
# Utilities
# ============================================================

def load_inputs(path: Path):
    """Load RQ1 input instances (JSONL)."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_X_train(inputs, feature_columns):
    """
    Build training matrix for PyExplainer.
    Only model-visible features are used.
    """
    return (
        pd.DataFrame([
            {f: item["input"]["features"].get(f, 0) for f in feature_columns}
            for item in inputs
        ])
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )


def extract_feature_row(features: dict, feature_columns):
    """
    Convert raw feature dict to a clean pandas row.
    - Missing -> 0
    - NaN -> 0
    - bool -> int
    """
    row = {}
    for f in feature_columns:
        v = features.get(f, 0)

        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = 0

        if isinstance(v, bool):
            v = int(v)

        row[f] = v

    return pd.DataFrame([row])


# ============================================================
# Main
# ============================================================

def main():
    print(f"[PyExplainer-RQ1] Project: {PROJECT}")

    # ---------- load inputs ----------
    inputs = load_inputs(INPUT_PATH)
    print(f"[PyExplainer-RQ1] Loaded {len(inputs)} instances")

    # ---------- load frozen JIT model ----------
    jit_model = JITModelWrapper(PROJECT)
    feature_columns = jit_model.feature_columns

    # ---------- build training data ----------
    X_train = build_X_train(inputs, feature_columns)

    # ---------- multiple runs ----------
    for run_id in range(1, N_RUNS + 1):
        seed = BASE_SEED + run_id
        print(f"[PyExplainer-RQ1] Run {run_id} (seed={seed})")

        explainer = PyExplainer(
            model=jit_model,
            X_train=X_train,
            random_state=seed
        )

        out_dir = OUTPUT_ROOT / f"run_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "explanations.jsonl"

        with open(out_file, "w", encoding="utf-8") as fw:
            for item in inputs:
                x_df = extract_feature_row(
                    item["input"]["features"],
                    feature_columns
                )

                explanation = explainer.explain(
                    x_df,
                    top_k=TOP_K
                )

                record = {
                    "commit_id": item["metadata"]["commit_id"],
                    "explanation": explanation
                }

                fw.write(json.dumps(record) + "\n")

    print("[PyExplainer-RQ1] Finished all runs.")


if __name__ == "__main__":
    main()
