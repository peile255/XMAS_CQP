"""
RQ2: Faithfulness Evaluation for CfExplainer (Final Stable Version)

Compatible with XMAS-CQP project structure.

Inputs:
- datasets/processed_<dataset>.jsonl
- results/<dataset>/cfexplainer/run_1/explanations.jsonl
- data/<dataset>/feature_stats.json
- models/<dataset>_jit_model.joblib

Metrics:
- Deletion@K
- Insertion@K
- Direction Faithfulness@K (DirFaith@K)

Notes:
- CfExplainer explanations are counterfactual-style: they provide feature changes.
- We convert counterfactual changes into "direction" (+1/-1) by comparing
  suggested value (or delta) with the original feature value.
"""

import json
import argparse
import numpy as np
from tqdm import tqdm
import joblib


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


def sign(val: float, eps: float = 1e-6) -> int:
    if val > eps:
        return +1
    if val < -eps:
        return -1
    return 0


def predict_proba(model, x: dict, feature_names):
    """
    Predict defect probability using fixed feature order.
    """
    x_norm = normalize_features(x)
    x_vec = [x_norm[f] for f in feature_names]
    return model.predict_proba([x_vec])[0][1]


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
# CfExplainer Explanation Adapter
# =========================================================

def _extract_candidate_items(expl: dict):
    """
    Pull potential "change list" candidates from multiple possible schema keys.
    We return a list of items (dict or tuple) describing feature changes.
    """
    # Common places: expl["explanation"], expl["changes"], expl["counterfactual"], etc.
    if "explanation" in expl:
        return expl["explanation"]
    if "changes" in expl:
        return expl["changes"]
    if "counterfactual" in expl:
        cf = expl["counterfactual"]
        if isinstance(cf, dict) and "changes" in cf:
            return cf["changes"]
        return cf
    if "cf" in expl:
        return expl["cf"]
    if "diff" in expl:
        return expl["diff"]
    if "edits" in expl:
        return expl["edits"]
    return None


def extract_cfexplainer_top_features(expl: dict, x: dict, K: int):
    """
    Normalize CfExplainer explanations to:
      [{"name": feature_name, "direction": +1/-1}]

    We try to detect:
    - direct target value: {"feature": "tcmt", "new_value": 10}
    - delta style: {"feature": "tcmt", "delta": -5}
    - pair style: ["tcmt", -5] or ("tcmt", -5)
    - nested: {"name": "...", "change": {...}}
    Sorting:
    - If "importance"/"score"/"rank" exists, use it.
    - Else sort by abs(delta) if delta exists.
    - Else keep original order.
    """

    items = _extract_candidate_items(expl)

    if items is None:
        raise KeyError(f"Unknown CfExplainer schema keys: {list(expl.keys())}")

    # Sometimes explanation is a dict that contains a list
    if isinstance(items, dict):
        # try common keys
        for key in ["features", "changes", "edits", "items"]:
            if key in items and isinstance(items[key], list):
                items = items[key]
                break

    if not isinstance(items, list):
        raise KeyError(f"CfExplainer 'items' is not a list. Type={type(items)}")

    parsed = []
    for it in items:
        feat = None
        delta = None
        new_val = None
        score = None
        rank = None

        # ---------- tuple/list form ----------
        if isinstance(it, (list, tuple)) and len(it) >= 2:
            feat = it[0]
            # second can be delta or new_value
            v2 = safe_float(it[1])
            if v2 is not None:
                delta = v2  # assume delta
            # optional score
            if len(it) >= 3:
                score = safe_float(it[2])

        # ---------- dict form ----------
        elif isinstance(it, dict):
            feat = it.get("feature") or it.get("name") or it.get("feature_name")

            # try many possible delta / target encodings
            delta = safe_float(it.get("delta") or it.get("change") or it.get("diff"))
            new_val = safe_float(it.get("new_value") or it.get("target") or it.get("value"))

            # nested change dict
            if delta is None and new_val is None and "change" in it and isinstance(it["change"], dict):
                ch = it["change"]
                delta = safe_float(ch.get("delta") or ch.get("diff") or ch.get("change"))
                new_val = safe_float(ch.get("new_value") or ch.get("target") or ch.get("value"))

            score = safe_float(it.get("score") or it.get("importance") or it.get("weight"))
            rank = it.get("rank")

        else:
            continue

        if feat is None or feat not in x:
            # ignore items that don't map to raw features
            continue

        # infer direction
        x0 = safe_float(x.get(feat))
        if x0 is None:
            continue

        # prefer new_val if present
        if new_val is not None:
            direction = sign(new_val - x0)
            magnitude = abs(new_val - x0)
        elif delta is not None:
            direction = sign(delta)
            magnitude = abs(delta)
        else:
            continue

        if direction == 0:
            continue

        parsed.append({
            "name": feat,
            "direction": int(direction),
            "score": score,
            "rank": rank,
            "magnitude": magnitude
        })

    if len(parsed) == 0:
        return []

    # sorting priority: rank -> abs(score) -> magnitude -> keep order
    if parsed[0].get("rank") is not None:
        parsed.sort(key=lambda z: (z["rank"] if z["rank"] is not None else 10**9))
    elif any(p.get("score") is not None for p in parsed):
        parsed.sort(key=lambda z: abs(z["score"]) if z.get("score") is not None else -1, reverse=True)
    else:
        parsed.sort(key=lambda z: z.get("magnitude", 0.0), reverse=True)

    # take top-K and deduplicate (preserve order)
    seen = set()
    out = []
    for p in parsed:
        if p["name"] in seen:
            continue
        seen.add(p["name"])
        out.append({"name": p["name"], "direction": p["direction"]})
        if len(out) >= K:
            break

    return out


# =========================================================
# Intervention Operators
# =========================================================

def delete_features(x, feature_list, median):
    """Replace selected features with dataset median."""
    x_new = x.copy()
    for f in feature_list:
        if f in median:
            x_new[f] = median[f]
    return x_new


def build_baseline(x, feature_list, median):
    """
    Baseline input for insertion:
    set explanation features to median.
    """
    x0 = x.copy()
    for f in feature_list:
        if f in median:
            x0[f] = median[f]
    return x0


def directional_perturb(x, feature, direction, q10, q90):
    """
    Directional perturbation for DirFaith.
    """
    x_new = x.copy()
    if feature in q10 and feature in q90:
        x_new[feature] = q90[feature] if direction == +1 else q10[feature]
    return x_new


# =========================================================
# Per-sample RQ2 Evaluation
# =========================================================

def evaluate_sample_cf(x, expl, model, feature_names, stats, K):

    feats = extract_cfexplainer_top_features(expl, x, K)
    if not feats:
        return 0.0, 0.0, 0.0

    feat_names = [f["name"] for f in feats]
    feat_dirs = {f["name"]: f["direction"] for f in feats}

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

    # ---------- Direction Faithfulness ----------
    hits = 0
    for f in feat_names:
        x_dir = directional_perturb(x, f, feat_dirs[f], stats["q10"], stats["q90"])
        p_dir = predict_proba(model, x_dir, feature_names)
        if sign(p_dir - p0) == feat_dirs[f]:
            hits += 1
    dirfaith = hits / len(feat_names)

    return del_score, ins_score, dirfaith


# =========================================================
# Dataset-level Evaluation
# =========================================================

def run_rq2_cf(samples, explanations, model, feature_names, stats, K):
    assert len(samples) == len(explanations), "Samples and explanations must be aligned."

    del_all, ins_all, dir_all = [], [], []

    for x, e in tqdm(zip(samples, explanations), total=len(samples)):
        d, i, df = evaluate_sample_cf(x, e, model, feature_names, stats, K)
        del_all.append(d)
        ins_all.append(i)
        dir_all.append(df)

    return {
        "Del@K": {"mean": float(np.mean(del_all)), "std": float(np.std(del_all))},
        "Ins@K": {"mean": float(np.mean(ins_all)), "std": float(np.std(ins_all))},
        "DirFaith@K": {"mean": float(np.mean(dir_all)), "std": float(np.std(dir_all))}
    }


# =========================================================
# Main CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="RQ2 Faithfulness Evaluation for CfExplainer (XMAS-CQP)"
    )
    parser.add_argument("--data", required=True,
                        help="datasets/processed_<dataset>.jsonl")
    parser.add_argument("--explanations", required=True,
                        help="results/<dataset>/cfexplainer/run_1/explanations.jsonl")
    parser.add_argument("--stats", required=True,
                        help="data/<dataset>/feature_stats.json")
    parser.add_argument("--model", required=True,
                        help="models/<dataset>_jit_model.joblib")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", default="rq2_faithfulness_cfexplainer.json")

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

    results = run_rq2_cf(samples, explanations, model, feature_names, stats, args.k)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("RQ2 (CfExplainer) Faithfulness evaluation finished.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
