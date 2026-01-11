"""
RQ2: Faithfulness Evaluation for PyExplainer (Final Stable Version)

PyExplainer explanations are typically rule/condition based, e.g.:
- "tcmt > 15"
- "3 < la <= 33"
- {"rule": "...", "conditions": [...]}

Therefore:
- We map rules/conditions back to base feature names
- We evaluate only Deletion@K and Insertion@K
- Direction Faithfulness is reported as N/A (not methodologically reliable)

Compatible with XMAS-CQP project structure.

Inputs:
- datasets/processed_<dataset>.jsonl
- results/<dataset>/pyexplainer/run_1/explanations.jsonl
- data/<dataset>/feature_stats.json
- models/<dataset>_jit_model.joblib
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
    out = {}
    for k, v in x.items():
        if isinstance(v, bool):
            out[k] = int(v)
        else:
            out[k] = float(v)
    return out


def predict_proba(model, x: dict, feature_names):
    """
    Predict defect probability using fixed feature order.
    """
    x_norm = normalize_features(x)
    x_vec = [x_norm[f] for f in feature_names]
    return model.predict_proba([x_vec])[0][1]


def rule_to_base_feature(text: str) -> str:
    """
    Extract base feature name from a rule/condition string.
    Examples:
    - "tcmt > 15.00" -> "tcmt"
    - "3.0 < la <= 33.0" -> "la"
    - "ndev <= 2" -> "ndev"
    """
    m = re.search(r"[A-Za-z_][A-Za-z0-9_]*", text)
    if m:
        return m.group(0)
    raise ValueError(f"Cannot extract base feature from rule: {text}")


def safe_float(v):
    try:
        if isinstance(v, bool):
            return float(int(v))
        return float(v)
    except Exception:
        return None


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
# PyExplainer Explanation Adapter
# =========================================================

def _get_candidate_rules(expl: dict):
    """
    Try many possible schema fields for PyExplainer outputs.
    Return something that can be interpreted as a list of rules/conditions.
    """

    # Most common: expl["explanation"]
    if "explanation" in expl:
        return expl["explanation"]

    # possible: rules / rule / conditions / if
    for key in ["rules", "rule", "conditions", "if", "conditions_list", "explain"]:
        if key in expl:
            return expl[key]

    # nested
    for key in ["result", "output", "explanation_obj"]:
        if key in expl and isinstance(expl[key], dict):
            for k2 in ["rules", "rule", "conditions", "explanation"]:
                if k2 in expl[key]:
                    return expl[key][k2]

    return None


def extract_pyexplainer_base_features(expl: dict, K: int):
    """
    Extract top-K base features from PyExplainer explanation.

    Supported item formats inside explanation:
    1) string rule: "tcmt > 15"
    2) dict condition: {"feature": "tcmt", "op": ">", "value": 15}
    3) dict rule: {"rule": "tcmt > 15", "importance": 0.2}
    4) list/tuple: ["tcmt > 15", 0.2] or ("tcmt > 15", 0.2)
    5) nested: {"conditions":[...]} where conditions can be dicts or strings

    Sorting:
    - by abs(weight/importance/score) if available
    - else keep order
    """

    items = _get_candidate_rules(expl)
    if items is None:
        raise KeyError(f"Unknown PyExplainer schema keys: {list(expl.keys())}")

    # If it's a dict, try pull list fields
    if isinstance(items, dict):
        for k in ["rules", "conditions", "items", "explanation"]:
            if k in items and isinstance(items[k], list):
                items = items[k]
                break

    # If single string/dict, wrap into list
    if isinstance(items, (str, dict, tuple)):
        items = [items]

    if not isinstance(items, list):
        raise KeyError(f"PyExplainer items is not list. type={type(items)}")

    parsed = []

    def add_rule(rule_text: str, weight: float | None):
        try:
            base = rule_to_base_feature(rule_text)
            parsed.append((base, weight))
        except Exception:
            pass

    for it in items:
        # string
        if isinstance(it, str):
            add_rule(it, None)
            continue

        # tuple/list
        if isinstance(it, (list, tuple)) and len(it) >= 1:
            rule_text = it[0]
            weight = safe_float(it[1]) if len(it) >= 2 else None
            if isinstance(rule_text, str):
                add_rule(rule_text, weight)
            continue

        # dict
        if isinstance(it, dict):
            # condition dict
            if "feature" in it and isinstance(it["feature"], str):
                base = it["feature"].strip()
                w = safe_float(it.get("weight") or it.get("importance") or it.get("score"))
                parsed.append((base, w))
                continue

            # rule text
            rule_text = it.get("rule") or it.get("condition") or it.get("text")
            w = safe_float(it.get("weight") or it.get("importance") or it.get("score"))
            if isinstance(rule_text, str):
                add_rule(rule_text, w)
                continue

            # nested conditions
            if "conditions" in it and isinstance(it["conditions"], list):
                for c in it["conditions"]:
                    if isinstance(c, str):
                        add_rule(c, w)
                    elif isinstance(c, dict) and "feature" in c:
                        parsed.append((c["feature"], w))
                continue

    if not parsed:
        return []

    # sort by abs(weight) if any weight exists
    if any(w is not None for _, w in parsed):
        parsed.sort(key=lambda t: abs(t[1]) if t[1] is not None else -1, reverse=True)

    # dedup preserve order, take K
    out = []
    seen = set()
    for base, _ in parsed:
        if base in seen:
            continue
        seen.add(base)
        out.append(base)
        if len(out) >= K:
            break

    return out


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
# Per-sample Evaluation (PyExplainer)
# =========================================================

def evaluate_sample_py(x, expl, model, feature_names, stats, K):

    feat_names = extract_pyexplainer_base_features(expl, K)

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

def run_rq2_py(samples, explanations, model, feature_names, stats, K):
    assert len(samples) == len(explanations), "Samples and explanations must be aligned."

    del_all, ins_all = [], []

    for x, e in tqdm(zip(samples, explanations), total=len(samples)):
        d, i = evaluate_sample_py(x, e, model, feature_names, stats, K)
        del_all.append(d)
        ins_all.append(i)

    return {
        "Del@K": {"mean": float(np.mean(del_all)), "std": float(np.std(del_all))},
        "Ins@K": {"mean": float(np.mean(ins_all)), "std": float(np.std(ins_all))},
        "DirFaith@K": "N/A (PyExplainer outputs rule/condition explanations)"
    }


# =========================================================
# Main CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="RQ2 Faithfulness Evaluation for PyExplainer (XMAS-CQP)"
    )
    parser.add_argument("--data", required=True,
                        help="datasets/processed_<dataset>.jsonl")
    parser.add_argument("--explanations", required=True,
                        help="results/<dataset>/pyexplainer/run_1/explanations.jsonl")
    parser.add_argument("--stats", required=True,
                        help="data/<dataset>/feature_stats.json")
    parser.add_argument("--model", required=True,
                        help="models/<dataset>_jit_model.joblib")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", default="rq2_faithfulness_pyexplainer.json")

    args = parser.parse_args()

    samples = load_samples(args.data)
    explanations = load_explanations(args.explanations)

    with open(args.stats, "r", encoding="utf-8") as f:
        stats = json.load(f)

    model_obj = joblib.load(args.model)

    # unwrap model + feature order (your pack format)
    if isinstance(model_obj, dict):
        model = model_obj.get("model")
        feature_names = model_obj.get("feature_columns") or model_obj.get("feature_names")
        if model is None or feature_names is None:
            raise KeyError(
                f"Loaded model dict missing keys. Keys: {list(model_obj.keys())}"
            )
    else:
        model = model_obj
        feature_names = model.feature_names_in_.tolist()

    results = run_rq2_py(samples, explanations, model, feature_names, stats, args.k)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("RQ2 (PyExplainer) Faithfulness evaluation finished.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
