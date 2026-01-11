#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQ3: Explanation Diversity / Informativeness Evaluation
(CFExplainer-compatible version)

Input schema (from your results/openstack/cfexplainer/run_i/explanations.jsonl):
{
  "commit_id": "...",
  "explanation": [
    ["rtime", 320930.78, "increases_risk"],
    ["rexp", 571.86, "increases_risk"],
    ["ld", 0.99, "increases_risk"]
  ]
}

Directory layout (REQUIRED):
results/
 └─ {project}/
     └─ cfexplainer/
         ├─ run_1/explanations.jsonl
         ├─ run_2/explanations.jsonl
         ├─ ...
         └─ run_N/explanations.jsonl

Metrics:
  - Inter-sample Jaccard similarity (InterJ@K)
  - Inter-sample Mean Rank Difference (InterMRD@K)
  - Global feature frequency entropy

Notes:
  - By default, diversity is computed at feature level: "rtime", "rexp", ...
  - Optional: include direction into the unit via --use_signed_feature
    e.g., "rtime::increases_risk" to measure direction-sensitive diversity.
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
# Explanation parsing (CFExplainer-specific)
# ----------------------------------------------------------------------

def extract_topk_units(rec, k, use_signed_feature=False):
    """
    Extract ordered Top-K explanation units for CFExplainer.

    Each element in rec["explanation"] is expected to be:
      [feature_name, value, direction]

    Ordering strategy:
      - preserve list order (CFExplainer already outputs a selected list)

    If use_signed_feature=True:
      unit = f"{feature}::{direction}"
    Else:
      unit = feature
    """
    items = rec.get("explanation", [])
    if not items:
        return []

    units = []
    for it in items[:k]:
        if not isinstance(it, list) or len(it) < 1:
            continue
        feat = it[0]
        direction = it[2] if len(it) >= 3 else None

        if use_signed_feature and direction is not None:
            units.append(f"{feat}::{direction}")
        else:
            units.append(str(feat))

    return units


# ----------------------------------------------------------------------
# Diversity metrics
# ----------------------------------------------------------------------

def inter_jaccard(a, b):
    s_a, s_b = set(a), set(b)
    if not s_a and not s_b:
        return 0.0
    return len(s_a & s_b) / len(s_a | s_b)


def inter_mrd(a, b, k):
    """
    Mean Rank Difference (missing rank = K + 1)
    """
    rank_a = {x: i + 1 for i, x in enumerate(a)}
    rank_b = {x: i + 1 for i, x in enumerate(b)}

    all_units = set(a) | set(b)
    diffs = []

    for x in all_units:
        ra = rank_a.get(x, k + 1)
        rb = rank_b.get(x, k + 1)
        diffs.append(abs(ra - rb))

    return mean(diffs) if diffs else 0.0


def entropy(counter):
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
    sample_ids = list(explanations.keys())
    pairs = combinations(sample_ids, 2)

    inter_j_vals = []
    inter_mrd_vals = []

    for i, j in pairs:
        a = explanations[i]
        b = explanations[j]
        inter_j_vals.append(inter_jaccard(a, b))
        inter_mrd_vals.append(inter_mrd(a, b, k))

    return inter_j_vals, inter_mrd_vals


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="RQ3 Explanation Diversity Evaluation (CFExplainer)"
    )

    parser.add_argument("--project", required=True, type=str)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n_runs", type=int, default=5)

    parser.add_argument("--input_dir", required=True, type=str,
                        help="Base results directory (e.g., results)")
    parser.add_argument("--output_dir", required=True, type=str)

    parser.add_argument("--use_signed_feature", action="store_true",
                        help="If set, treat unit as feature::direction")

    return parser.parse_args()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    args = parse_args()

    project = args.project
    method = "cfexplainer"
    k = args.k

    all_inter_j = []
    all_inter_mrd = []

    unit_counter = Counter()
    n_samples = None
    empty_explanations = 0

    for run_idx in range(1, args.n_runs + 1):
        records = load_explanations(
            args.input_dir, project, method, run_idx
        )

        explanations = {}

        for rec in records:
            sid = rec.get("commit_id") or rec.get("id")
            if sid is None:
                continue

            units = extract_topk_units(
                rec, k, use_signed_feature=args.use_signed_feature
            )
            if not units:
                empty_explanations += 1
                continue

            explanations[sid] = units
            for u in units:
                unit_counter[u] += 1

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
        "use_signed_feature": bool(args.use_signed_feature),

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

        "unit_distribution": {
            "entropy": entropy(unit_counter),
            "unique_units": len(unit_counter),
            "top_units": [
                {
                    "unit": u,
                    "frequency": c / (n_samples * k)
                }
                for u, c in unit_counter.most_common(10)
            ]
        }
    }

    out_dir = os.path.join(args.output_dir, project, method)
    ensure_dir(out_dir)

    fname = "rq3_diversity_signed.json" if args.use_signed_feature else "rq3_diversity.json"
    with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    paper_summary = {
        f"InterJ@{k}_mean": summary["inter_sample_metrics"]["InterJ@K"]["mean"],
        f"InterJ@{k}_std": summary["inter_sample_metrics"]["InterJ@K"]["std"],
        f"InterMRD@{k}_mean": summary["inter_sample_metrics"]["InterMRD@K"]["mean"],
        f"InterMRD@{k}_std": summary["inter_sample_metrics"]["InterMRD@K"]["std"],
        "Entropy": summary["unit_distribution"]["entropy"]
    }

    sf = "_signed" if args.use_signed_feature else ""
    with open(os.path.join(out_dir, f"rq3_summary{sf}.json"), "w", encoding="utf-8") as f:
        json.dump(paper_summary, f, indent=2)

    print("[INFO] RQ3 diversity evaluation finished (CFExplainer).")
    print(json.dumps(paper_summary, indent=2))


if __name__ == "__main__":
    main()
