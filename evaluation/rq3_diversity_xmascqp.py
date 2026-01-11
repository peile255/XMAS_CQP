#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQ3: Explanation Diversity / Informativeness Evaluation
(XMAS-CQP compatible, FINAL VERSION)

This script evaluates inter-sample explanation diversity based on
existing explanation outputs produced in RQ1 / RQ2.

Directory layout (REQUIRED):
results/
 └─ {project}/
     └─ {method}/
         ├─ run_1/explanations.jsonl
         ├─ run_2/explanations.jsonl
         ├─ ...
         └─ run_N/explanations.jsonl

Metrics:
  - Inter-sample Jaccard similarity (InterJ@K)
  - Inter-sample Mean Rank Difference (InterMRD@K)
  - Global feature frequency entropy

Design principles:
  - No new explanation generation
  - Run-isolated evaluation
  - Deterministic aggregation
  - Fail-fast on empty explanations
"""

import argparse
import json
import math
import os
from collections import Counter
from itertools import combinations
from statistics import mean, stdev


# ----------------------------------------------------------------------
# IO utilities
# ----------------------------------------------------------------------

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_explanations(base_dir, project, method, run_idx):
    """
    Load explanations from:
    results/{project}/{method}/run_{i}/explanations.jsonl
    """
    path = os.path.join(
        base_dir,
        project,
        method,
        f"run_{run_idx}",
        "explanations.jsonl"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing explanation file: {path}")

    return load_jsonl(path)


# ----------------------------------------------------------------------
# Explanation parsing (CRITICAL PART)
# ----------------------------------------------------------------------

def extract_topk_features(expl, k):
    """
    Extract ordered Top-K feature names from one explanation record.

    Primary supported schema (XMAS-CQP):
      expl["key_factors"] = [
        {
          "rank": 1,
          "importance": 1.0,
          "feature": "nd",
          ...
        },
        ...
      ]
    """

    # ===== XMAS-CQP canonical schema =====
    if "key_factors" in expl and isinstance(expl["key_factors"], list):
        factors = expl["key_factors"]

        # Prefer explicit rank if available
        if all("rank" in f for f in factors):
            factors = sorted(factors, key=lambda x: x["rank"])
        else:
            # Fallback: importance-based ordering
            factors = sorted(
                factors,
                key=lambda x: -abs(x.get("importance", 0))
            )

        return [
            f["feature"]
            for f in factors[:k]
            if "feature" in f
        ]

    # ===== Fallback schemas (for other explainers) =====
    if "features" in expl:
        feats = expl["features"]
        feats = sorted(feats, key=lambda x: x.get("rank", 1e9))
        return [f["name"] for f in feats[:k] if "name" in f]

    if "explanation" in expl:
        feats = expl["explanation"]
        feats = sorted(
            feats, key=lambda x: -abs(x.get("importance", 0))
        )
        return [f["feature"] for f in feats[:k] if "feature" in f]

    if "top_features" in expl:
        return [f[0] for f in expl["top_features"][:k]]

    return []


# ----------------------------------------------------------------------
# Diversity metrics
# ----------------------------------------------------------------------

def inter_jaccard(f_i, f_j):
    s_i, s_j = set(f_i), set(f_j)
    if not s_i and not s_j:
        return 0.0
    return len(s_i & s_j) / len(s_i | s_j)


def inter_mrd(f_i, f_j, k):
    """
    Mean Rank Difference (missing rank = K + 1)
    """
    rank_i = {f: idx + 1 for idx, f in enumerate(f_i)}
    rank_j = {f: idx + 1 for idx, f in enumerate(f_j)}

    all_feats = set(f_i) | set(f_j)
    diffs = []

    for f in all_feats:
        r_i = rank_i.get(f, k + 1)
        r_j = rank_j.get(f, k + 1)
        diffs.append(abs(r_i - r_j))

    return mean(diffs) if diffs else 0.0


def feature_entropy(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0

    h = 0.0
    for cnt in counter.values():
        p = cnt / total
        h -= p * math.log(p + 1e-12)
    return h


# ----------------------------------------------------------------------
# Core evaluation logic
# ----------------------------------------------------------------------

def evaluate_single_run(explanations, k):
    """
    Compute inter-sample diversity metrics for ONE run.
    """
    sample_ids = list(explanations.keys())
    pairs = combinations(sample_ids, 2)

    inter_j_vals = []
    inter_mrd_vals = []

    for i, j in pairs:
        f_i = explanations[i]
        f_j = explanations[j]
        inter_j_vals.append(inter_jaccard(f_i, f_j))
        inter_mrd_vals.append(inter_mrd(f_i, f_j, k))

    return inter_j_vals, inter_mrd_vals


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="RQ3 Explanation Diversity Evaluation (XMAS-CQP)"
    )

    parser.add_argument("--project", required=True, type=str)
    parser.add_argument("--method", required=True, type=str)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--input_dir", required=True, type=str,
                        help="Base results directory (e.g., results)")
    parser.add_argument("--output_dir", required=True, type=str)

    return parser.parse_args()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    args = parse_args()

    project = args.project
    method = args.method
    k = args.k

    all_inter_j = []
    all_inter_mrd = []

    feature_counter = Counter()
    n_samples = None
    empty_explanations = 0

    for run_idx in range(1, args.n_runs + 1):
        records = load_explanations(
            args.input_dir, project, method, run_idx
        )

        explanations = {}

        for rec in records:
            sid = (
                rec.get("sample_id")
                or rec.get("commit_id")
                or rec.get("id")
            )
            if sid is None:
                continue

            feats = extract_topk_features(rec, k)

            if not feats:
                empty_explanations += 1
                continue

            explanations[sid] = feats

            for f in feats:
                feature_counter[f] += 1

        if n_samples is None:
            n_samples = len(explanations)

        inter_j_vals, inter_mrd_vals = evaluate_single_run(
            explanations, k
        )

        all_inter_j.extend(inter_j_vals)
        all_inter_mrd.extend(inter_mrd_vals)

    # ------------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------------

    summary = {
        "project": project,
        "method": method,
        "k": k,
        "n_runs": args.n_runs,

        "sample_stats": {
            "n_samples": n_samples,
            "n_pairs": len(all_inter_j),
            "empty_explanations": empty_explanations
        },

        "inter_sample_metrics": {
            "InterJ@K": {
                "mean": mean(all_inter_j),
                "std": stdev(all_inter_j) if len(all_inter_j) > 1 else 0.0
            },
            "InterMRD@K": {
                "mean": mean(all_inter_mrd),
                "std": stdev(all_inter_mrd) if len(all_inter_mrd) > 1 else 0.0
            }
        },

        "feature_distribution": {
            "entropy": feature_entropy(feature_counter),
            "unique_features": len(feature_counter),
            "top_features": [
                {
                    "feature": f,
                    "frequency": c / (n_samples * k)
                }
                for f, c in feature_counter.most_common(10)
            ]
        }
    }

    out_dir = os.path.join(args.output_dir, project, method)
    ensure_dir(out_dir)

    with open(os.path.join(out_dir, "rq3_diversity.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    paper_summary = {
        f"InterJ@{k}_mean": summary["inter_sample_metrics"]["InterJ@K"]["mean"],
        f"InterJ@{k}_std": summary["inter_sample_metrics"]["InterJ@K"]["std"],
        f"InterMRD@{k}_mean": summary["inter_sample_metrics"]["InterMRD@K"]["mean"],
        f"InterMRD@{k}_std": summary["inter_sample_metrics"]["InterMRD@K"]["std"],
        "FeatureEntropy": summary["feature_distribution"]["entropy"]
    }

    with open(os.path.join(out_dir, "rq3_summary.json"), "w", encoding="utf-8") as f:
        json.dump(paper_summary, f, indent=2)

    print("[INFO] RQ3 diversity evaluation finished.")
    print(json.dumps(paper_summary, indent=2))


if __name__ == "__main__":
    main()
