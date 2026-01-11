# evaluation/rq1_lime.py
"""
RQ1 evaluator for multiple explainers:
    method ∈ {xmas, lime, cfexplainer, pyexplainer}

Metrics:
- J@K   : Jaccard similarity of top-K features
- DAR@K : Direction Agreement Ratio (method-aware)
- MRD@K : Mean Rank Difference
- SSS@K : Structural Stability Score (method-aware composite)

Key design decisions:
- DAR is excluded from SSS for surrogate-based explainers (LIME, PyExplainer)
- MRD uses missing-rank = K+1
- Directions marked as 'unknown' are ignored in DAR

This evaluator is deterministic given fixed input files.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
# Basic statistics
# ============================================================

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def std(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


# ============================================================
# Core metrics
# ============================================================

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def normalize_direction(d: Optional[str]) -> str:
    if d is None:
        return "unknown"
    d = str(d).lower().strip()

    if d in {"increase", "increases", "increases_risk", "positive", "+"}:
        return "increases_risk"
    if d in {"decrease", "decreases", "decreases_risk", "negative", "-"}:
        return "decreases_risk"
    if d in {"neutral", "none", "0"}:
        return "neutral"

    return "unknown"


def direction_from_weight(w: float, eps: float = 1e-12) -> str:
    if w > eps:
        return "increases_risk"
    if w < -eps:
        return "decreases_risk"
    return "neutral"


# ============================================================
# Parsed explanation abstraction
# ============================================================

@dataclass
class ParsedExplanation:
    ranked_features: List[str]
    feature_to_dir: Dict[str, str]


# ============================================================
# Parsing helpers
# ============================================================

def parse_lime_condition(cond: str) -> str:
    parts = cond.strip().split()
    if not parts:
        return cond
    if "<" in parts and "<=" in parts and len(parts) >= 3:
        return parts[2]
    return parts[0]


def parse_xmas(obj: dict, k: int) -> ParsedExplanation:
    kf = obj.get("key_factors", [])
    kf = sorted(kf, key=lambda x: x.get("rank", 1e9))[:k]

    feats, dirs = [], {}
    for item in kf:
        f = str(item.get("feature", "")).strip()
        if not f:
            continue
        feats.append(f)
        dirs[f] = normalize_direction(item.get("direction"))

    return ParsedExplanation(feats, dirs)


def parse_lime_like(obj: dict, k: int) -> ParsedExplanation:
    exp = obj.get("explanation", [])
    feats, dirs = [], {}

    for pair in exp[:k]:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        cond, w = pair[0], pair[1]
        f = parse_lime_condition(str(cond))
        feats.append(f)
        dirs[f] = direction_from_weight(float(w))

    return ParsedExplanation(feats, dirs)


def parse_fallback(obj: dict, k: int) -> ParsedExplanation:
    if "explanation" in obj:
        return parse_lime_like(obj, k)

    rules = obj.get("rules", [])
    feats = [str(r).split()[0] for r in rules[:k]]
    return ParsedExplanation(feats, {f: "unknown" for f in feats})


def parse_explanation(method: str, obj: dict, k: int) -> ParsedExplanation:
    if method == "xmas":
        return parse_xmas(obj, k)
    if method == "lime":
        return parse_lime_like(obj, k)
    if method in {"cfexplainer", "pyexplainer"}:
        return parse_fallback(obj, k)
    raise ValueError(method)


def get_commit_id(obj: dict) -> Optional[str]:
    if "commit_id" in obj:
        return str(obj["commit_id"])
    if "metadata" in obj and "commit_id" in obj["metadata"]:
        return str(obj["metadata"]["commit_id"])
    return None


# ============================================================
# Pairwise metrics
# ============================================================

def compute_pairwise(
    a: ParsedExplanation,
    b: ParsedExplanation,
    k: int
) -> Tuple[float, float, float]:

    fa, fb = a.ranked_features[:k], b.ranked_features[:k]

    # J
    j = jaccard(fa, fb)

    # DAR (ignore unknown)
    inter = set(fa) & set(fb)
    dar_vals = []
    for f in inter:
        da = a.feature_to_dir.get(f, "unknown")
        db = b.feature_to_dir.get(f, "unknown")
        if da != "unknown" and db != "unknown":
            dar_vals.append(1.0 if da == db else 0.0)
    dar = mean(dar_vals) if dar_vals else 0.0

    # MRD (missing = k+1)
    rank_a = {f: i + 1 for i, f in enumerate(fa)}
    rank_b = {f: i + 1 for i, f in enumerate(fb)}
    miss = k + 1
    union = set(fa) | set(fb)
    diffs = [abs(rank_a.get(f, miss) - rank_b.get(f, miss)) for f in union]
    mrd = mean(diffs) if diffs else 0.0

    return j, dar, mrd


# ============================================================
# Method-aware SSS
# ============================================================

def dar_weight(method: str) -> float:
    return 0.0 if method in {"lime", "pyexplainer"} else 1.0


def sss_from(j: float, dar: float, mrd: float, k: int, method: str) -> float:
    rank_sim = 1.0 - (mrd / (k + 1))
    rank_sim = max(0.0, min(1.0, rank_sim))

    if dar_weight(method) == 0.0:
        return (j + rank_sim) / 2.0
    else:
        return (j + dar + rank_sim) / 3.0


# ============================================================
# Evaluation
# ============================================================

def evaluate(project: str, method: str, base_dir: Path, runs: int, k: int):

    run_maps: Dict[int, Dict[str, ParsedExplanation]] = {}

    for r in range(1, runs + 1):
        p = base_dir / project / method / f"run_{r}" / "explanations.jsonl"
        if not p.exists():
            raise FileNotFoundError(p)

        objs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
        m = {}
        for obj in objs:
            cid = get_commit_id(obj)
            if cid:
                m[cid] = parse_explanation(method, obj, k)
        run_maps[r] = m

    common_ids = set.intersection(*(set(m.keys()) for m in run_maps.values()))
    if not common_ids:
        raise RuntimeError("No common commit_ids found")

    J, D, M, S = [], [], [], []

    for cid in common_ids:
        for i, j in combinations(range(1, runs + 1), 2):
            ja, da, ma = compute_pairwise(run_maps[i][cid], run_maps[j][cid], k)
            sa = sss_from(ja, da, ma, k, method)
            J.append(ja)
            D.append(da)
            M.append(ma)
            S.append(sa)

    return {
        "project": project,
        "method": method,
        "n_commits": len(common_ids),
        "n_runs": runs,
        "k": k,
        f"J@{k}": {"mean": mean(J), "std": std(J)},
        f"DAR@{k}": {"mean": mean(D), "std": std(D)},
        f"MRD@{k}": {"mean": mean(M), "std": std(M)},
        f"SSS@{k}": {
            "mean": mean(S),
            "std": std(S),
            "note": "DAR excluded from SSS"
            if method in {"lime", "pyexplainer"}
            else "DAR included",
        },
    }


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--method", required=True,
                    choices=["xmas", "lime", "cfexplainer", "pyexplainer"])
    ap.add_argument("--base_dir", default="results")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out_dir", default="evaluation/lime_results")
    args = ap.parse_args()

    summary = evaluate(
        project=args.project,
        method=args.method,
        base_dir=Path(args.base_dir),
        runs=args.runs,
        k=args.k,
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / f"rq1_{args.project}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
