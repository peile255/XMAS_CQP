"""
RQ2: Faithfulness Evaluation for LIME (Final Stable Version)

Design decisions:
- LIME explanations are threshold-based rules, not raw features
- Rules are mapped back to base feature names
- Only Deletion@K and Insertion@K are evaluated
- Directional Faithfulness is NOT applicable to LIME

Compatible with XMAS-CQP project structure.
"""

import json
import argparse
import numpy as np
from tqdm import tqdm
import joblib
import re


# =========================================================
# Utilities
# =========================================================

def normalize_features(x: dict) -> dict:
    """Ensure model-compatible feature values."""
    return {
        k: int(v) if isinstance(v, bool) else float(v)
        for k, v in x.items()
    }


def predict_proba(model, x: dict, feature_names):
    """
    Predict defect probability using fixed feature order.
    """
    x_norm = normalize_features(x)
    x_vec = [x_norm[f] for f in feature_names]
    return model.predict_proba([x_vec])[0][1]


def lime_feature_to_base(rule: str) -> str:
    """
    Robustly extract base feature name from LIME rule string.

    Examples:
    - "tcmt > 15.00"            -> "tcmt"
    - "3.00 < tcmt <= 15.00"   -> "tcmt"
    - "la <= 33.00"            -> "la"
    - "33.00 < la"             -> "la"
    """

    # First try regex for valid identifier
    match = re.search(r"[A-Za-z_][A-Za-z0-9_]*", rule)
    if match:
        return match.group(0)

    raise ValueError(f"Cannot extract base feature from LIME rule: {rule}")


# =========================================================
# Data Loading
# =========================================================

def load_samples(jsonl_path: str):
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            samples.append(obj["input"]["features"])
    return samples


def load_explanations(jsonl_path: str):
    exps = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            exps.append(json.loads(line))
    return exps


# =========================================================
# LIME Explanation Adapter
# =========================================================

def extract_lime_base_features(explanation: dict, K: int):
    """
    Extract top-K base feature names from LIME explanation.

    Supported formats:
    - ["tcmt > 15.00", 0.32]
    - ("tcmt > 15.00", 0.32)
    - {"feature": "tcmt > 15.00", "weight": 0.32}
    """

    if "explanation" not in explanation:
        raise KeyError(f"Invalid LIME explanation schema: {explanation.keys()}")

    raw_items = explanation["explanation"]
    parsed = []

    for item in raw_items:
        # Case 1: tuple or list (standard LIME)
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            rule, weight = item[0], item[1]

        # Case 2: dict-based
        elif isinstance(item, dict):
            rule = item.get("feature") or item.get("name")
            weight = item.get("weight", 0.0)

        else:
            continue

        if rule is None:
            continue

        parsed.append((rule, float(weight)))

    # sort by absolute contribution
    parsed.sort(key=lambda x: abs(x[1]), reverse=True)

    base_features = []
    for rule, _ in parsed[:K]:
        base_features.append(lime_feature_to_base(rule))

    # remove duplicates, preserve order
    return list(dict.fromkeys(base_features))


# =========================================================
# Intervention Operators
# =========================================================

def delete_features(x, feature_list, median):
    x_new = x.copy()
    for f in feature_list:
        if f in median:
            x_new[f] = median[f]
    return x_new


def build_baseline(x, feature_list, median):
    x0 = x.copy()
    for f in feature_list:
        if f in median:
            x0[f] = median[f]
    return x0


# =========================================================
# Per-sample Evaluation (LIME)
# =========================================================

def evaluate_sample_lime(x, explanation, model, feature_names, stats, K):

    feat_names = extract_lime_base_features(explanation, K)

    if not feat_names:
        return 0.0, 0.0

    p0 = predict_proba(model, x, feature_names)

    # ---------- Deletion ----------
    del_deltas = []
    for m in range(1, len(feat_names) + 1):
        x_del = delete_features(x, feat_names[:m], stats["median"])
        p_del = predict_proba(model, x_del, feature_names)
        del_deltas.append(p0 - p_del)

    del_score = float(np.mean(del_deltas))

    # ---------- Insertion ----------
    x_base = build_baseline(x, feat_names, stats["median"])
    p_base = predict_proba(model, x_base, feature_names)

    ins_deltas = []
    for m in range(1, len(feat_names) + 1):
        x_ins = x_base.copy()
        for f in feat_names[:m]:
            x_ins[f] = x[f]
        p_ins = predict_proba(model, x_ins, feature_names)
        ins_deltas.append(p_ins - p_base)

    ins_score = float(np.mean(ins_deltas))

    return del_score, ins_score


# =========================================================
# Dataset-level Evaluation
# =========================================================

def run_rq2_lime(samples, explanations, model, feature_names, stats, K):

    del_all, ins_all = [], []

    for x, expl in tqdm(zip(samples, explanations), total=len(samples)):
        d, i = evaluate_sample_lime(
            x, expl, model, feature_names, stats, K
        )
        del_all.append(d)
        ins_all.append(i)

    return {
        "Del@K": {
            "mean": float(np.mean(del_all)),
            "std": float(np.std(del_all))
        },
        "Ins@K": {
            "mean": float(np.mean(ins_all)),
            "std": float(np.std(ins_all))
        },
        "DirFaith@K": "N/A (LIME explanations are rule-based)"
    }


# =========================================================
# Main CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="RQ2 Faithfulness Evaluation for LIME"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--explanations", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", default="rq2_faithfulness_lime.json")

    args = parser.parse_args()

    samples = load_samples(args.data)
    explanations = load_explanations(args.explanations)

    with open(args.stats, "r", encoding="utf-8") as f:
        stats = json.load(f)

    model_obj = joblib.load(args.model)

    # unwrap model + feature order
    if isinstance(model_obj, dict):
        model = model_obj["model"]
        feature_names = model_obj["feature_columns"]
    else:
        model = model_obj
        feature_names = model.feature_names_in_.tolist()

    results = run_rq2_lime(
        samples, explanations, model, feature_names, stats, args.k
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("RQ2 (LIME) Faithfulness evaluation finished.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
