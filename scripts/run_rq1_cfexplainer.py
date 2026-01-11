"""
run_rq1_cfexplainer.py

RQ1: Run-to-run stability evaluation for CfExplainer
----------------------------------------------------

This script generates CfExplainer explanations for the same
input instances across multiple random seeds, under a frozen
JIT defect prediction model.

Output:
results/
└── openstack/
    └── cfexplainer/
        ├── run_1/explanations.jsonl
        ├── run_2/explanations.jsonl
        ├── run_3/explanations.jsonl
        ├── run_4/explanations.jsonl
        └── run_5/explanations.jsonl
"""

import sys
import json
import random
from pathlib import Path

# ------------------------------------------------------------
# Make project root importable
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from Baseline.jit_model import JITModelWrapper
from Baseline.baselines.cfexplainer import CfExplainer


# ============================================================
# Global config
# ============================================================

PROJECT = "qt"
N_RUNS = 5
BASE_SEED = 43
TOP_K = 5

INPUT_PATH = Path("data/rq1_samples/qt_rq1_input.jsonl")
OUTPUT_ROOT = Path("results/qt/cfexplainer")


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_inputs(path: Path):
    """Load RQ1 input instances (JSONL)."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_X_train(inputs, feature_columns):
    """
    Build training matrix for CfExplainer.
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


def extract_instance_df(features: dict, feature_columns):
    """
    Convert feature dict to single-row DataFrame.
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


def normalize_explanation(cf_rules, top_k):
    """
    Normalize CfExplainer output to:
    [(feature, weight, direction), ...]
    """
    normalized = []

    for r in cf_rules[:top_k]:
        feature = r["feature"]
        weight = float(abs(r["importance"]))
        direction = "increases_risk" if r["importance"] > 0 else "decreases_risk"

        normalized.append([feature, weight, direction])

    return normalized


# ============================================================
# Main
# ============================================================

def main():
    print(f"[CfExplainer-RQ1] Project: {PROJECT}")

    # ---------- load inputs ----------
    inputs = load_inputs(INPUT_PATH)
    print(f"[CfExplainer-RQ1] Loaded {len(inputs)} instances")

    # ---------- load frozen JIT model ----------
    jit_model = JITModelWrapper(PROJECT)
    feature_columns = jit_model.feature_columns

    # ---------- build training data ----------
    X_train = build_X_train(inputs, feature_columns)

    # ---------- multiple runs ----------
    for run_id in range(1, N_RUNS + 1):
        seed = BASE_SEED + run_id
        set_seed(seed)

        print(f"[CfExplainer-RQ1] Run {run_id} (seed={seed})")

        explainer = CfExplainer(
            model=jit_model,
            X_train=X_train,
            random_state=seed
        )

        out_dir = OUTPUT_ROOT / f"run_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "explanations.jsonl"

        with open(out_file, "w", encoding="utf-8") as fw:
            for item in inputs:
                x_df = extract_instance_df(
                    item["input"]["features"],
                    feature_columns
                )

                rules = explainer.explain(
                    x_df,
                    top_k=TOP_K
                )

                explanation = normalize_explanation(rules, TOP_K)

                record = {
                    "commit_id": item["metadata"]["commit_id"],
                    "explanation": explanation
                }

                fw.write(json.dumps(record) + "\n")

    print("[CfExplainer-RQ1] Finished all runs.")


if __name__ == "__main__":
    main()
