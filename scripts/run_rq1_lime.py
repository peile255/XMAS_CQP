"""
run_rq1_lime.py

RQ1: Run-to-run stability evaluation for LIME explanations
-----------------------------------------------------------
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
from lime.lime_tabular import LimeTabularExplainer

from Baseline.jit_model import JITModelWrapper


# ============================================================
# Global config
# ============================================================

PROJECT = "qt"
N_RUNS = 5
BASE_SEED = 43
TOP_K = 5

INPUT_PATH = Path("data/rq1_samples/qt_rq1_input.jsonl")
OUTPUT_ROOT = Path("results/qt/lime")


# ============================================================
# Utilities
# ============================================================

def load_inputs(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_X_train(inputs, feature_columns):
    return (
        pd.DataFrame([
            {f: item["input"]["features"].get(f, 0) for f in feature_columns}
            for item in inputs
        ])
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )


def extract_feature_vector(features: dict, feature_columns: list):
    """
    Convert raw feature dict to a clean numeric vector for LIME.
    """
    vec = []
    for f in feature_columns:
        v = features.get(f, 0)

        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = 0

        if isinstance(v, bool):
            v = int(v)

        vec.append(v)

    return np.array(vec, dtype=float)


# ============================================================
# Main
# ============================================================

def main():
    print(f"[LIME-RQ1] Project: {PROJECT}")

    inputs = load_inputs(INPUT_PATH)
    print(f"[LIME-RQ1] Loaded {len(inputs)} instances")

    # ---------- frozen JIT model ----------
    jit_model = JITModelWrapper(PROJECT)
    feature_columns = jit_model.feature_columns

    # ---------- training data for LIME ----------
    X_train = build_X_train(inputs, feature_columns)

    for run_id in range(1, N_RUNS + 1):
        seed = BASE_SEED + run_id
        print(f"[LIME-RQ1] Run {run_id} (seed={seed})")

        explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=feature_columns,
            class_names=["clean", "buggy"],
            discretize_continuous=True,
            random_state=seed,
        )

        out_dir = OUTPUT_ROOT / f"run_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_file = out_dir / "explanations.jsonl"

        with open(out_file, "w", encoding="utf-8") as fw:
            for item in inputs:
                x = extract_feature_vector(
                    item["input"]["features"],
                    feature_columns
                )

                exp = explainer.explain_instance(
                    x,
                    jit_model.predict_proba,
                    num_features=TOP_K,
                )

                record = {
                    "commit_id": item["metadata"]["commit_id"],
                    "explanation": exp.as_list()
                }

                fw.write(json.dumps(record) + "\n")

    print("[LIME-RQ1] Finished all runs.")


if __name__ == "__main__":
    main()
