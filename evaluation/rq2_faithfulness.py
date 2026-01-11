"""
RQ2: Faithfulness Evaluation (Final Ultimate Version)

Compatible with XMAS-CQP project structure and explanation schemas.
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
    return {k: int(v) if isinstance(v, bool) else float(v) for k, v in x.items()}


def sign(val: float, eps: float = 1e-6) -> int:
    if val > eps:
        return +1
    elif val < -eps:
        return -1
    return 0


def to_direction(val):
    if isinstance(val, (int, float)):
        return +1 if val > 0 else -1 if val < 0 else 0
    if isinstance(val, bool):
        return +1 if val else -1
    if isinstance(val, str):
        s = val.lower()
        if s in {"increase", "increases", "increases_risk", "positive", "up", "+"}:
            return +1
        if s in {"decrease", "decreases", "decreases_risk", "negative", "down", "-"}:
            return -1
        if s in {"neutral", "none", "no_change", "0"}:
            return 0
    raise ValueError(f"Unsupported direction: {val}")


def predict_proba(model, x: dict, feature_names):
    x_norm = normalize_features(x)
    x_vec = [x_norm[f] for f in feature_names]
    return model.predict_proba([x_vec])[0][1]


# =========================================================
# Data Loading
# =========================================================

def load_samples(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line)["input"]["features"])
    return samples


def load_explanations(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# =========================================================
# Explanation Adapter
# =========================================================

def extract_top_features(expl, K):
    if "key_factors" in expl:
        feats = expl["key_factors"]
    elif "explanation" in expl and isinstance(expl["explanation"], dict):
        feats = expl["explanation"].get("features", [])
    elif "features" in expl:
        feats = expl["features"]
    else:
        raise KeyError(f"Unknown explanation schema: {expl.keys()}")

    if feats and "rank" in feats[0]:
        feats = sorted(feats, key=lambda x: x["rank"])
    elif feats and "score" in feats[0]:
        feats = sorted(feats, key=lambda x: -abs(x["score"]))

    out = []
    for f in feats[:K]:
        name = f.get("name") or f.get("feature")
        direction = (
            to_direction(f["direction"])
            if "direction" in f
            else to_direction(f["risk_direction"])
        )
        out.append({"name": name, "direction": direction})
    return out


# =========================================================
# Interventions
# =========================================================

def delete_features(x, feats, median):
    y = x.copy()
    for f in feats:
        y[f] = median[f]
    return y


def build_baseline(x, feats, median):
    return delete_features(x, feats, median)


def directional_perturb(x, f, d, q10, q90):
    y = x.copy()
    y[f] = q90[f] if d == +1 else q10[f]
    return y


# =========================================================
# RQ2 Core
# =========================================================

def evaluate_sample(x, expl, model, feature_names, stats, K):
    feats = [f for f in extract_top_features(expl, K) if f["direction"] != 0]
    if not feats:
        return 0.0, 0.0, 0.0

    names = [f["name"] for f in feats]
    dirs = {f["name"]: f["direction"] for f in feats}

    p0 = predict_proba(model, x, feature_names)

    deltas = []
    for m in range(1, len(names) + 1):
        p = predict_proba(
            model,
            delete_features(x, names[:m], stats["median"]),
            feature_names,
        )
        deltas.append(p0 - p)

    base = build_baseline(x, names, stats["median"])
    p_base = predict_proba(model, base, feature_names)

    ins = []
    for m in range(1, len(names) + 1):
        xi = base.copy()
        for f in names[:m]:
            xi[f] = x[f]
        ins.append(predict_proba(model, xi, feature_names) - p_base)

    hits = 0
    for f in names:
        p = predict_proba(
            model,
            directional_perturb(x, f, dirs[f], stats["q10"], stats["q90"]),
            feature_names,
        )
        if sign(p - p0) == dirs[f]:
            hits += 1

    return float(np.mean(deltas)), float(np.mean(ins)), hits / len(names)


def run_rq2(samples, exps, model, feature_names, stats, K):
    D, I, F = [], [], []
    for x, e in tqdm(zip(samples, exps), total=len(samples)):
        d, i, f = evaluate_sample(x, e, model, feature_names, stats, K)
        D.append(d)
        I.append(i)
        F.append(f)
    return {
        "Del@K": {"mean": float(np.mean(D)), "std": float(np.std(D))},
        "Ins@K": {"mean": float(np.mean(I)), "std": float(np.std(I))},
        "DirFaith@K": {"mean": float(np.mean(F)), "std": float(np.std(F))},
    }


# =========================================================
# CLI
# =========================================================

def main():
    ap = argparse.ArgumentParser("RQ2 Faithfulness Evaluation")
    ap.add_argument("--data", required=True)
    ap.add_argument("--explanations", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--output", default="rq2_faithfulness.json")
    args = ap.parse_args()

    samples = load_samples(args.data)
    exps = load_explanations(args.explanations)

    with open(args.stats, "r") as f:
        stats = json.load(f)

    model_obj = joblib.load(args.model)
    model = model_obj["model"]

    feature_names = (
        model_obj.get("feature_names")
        or model_obj.get("feature_columns")
    )

    if feature_names is None:
        raise KeyError(f"No feature list found in model: {model_obj.keys()}")

    results = run_rq2(samples, exps, model, feature_names, stats, args.k)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("RQ2 finished.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
