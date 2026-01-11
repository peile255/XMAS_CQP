"""
rq1_pyexplainer.py

RQ1: Run-to-run stability evaluation for PyExplainer.

Metrics (aligned with unified RQ1 evaluator):
- J@K   : Jaccard similarity of top-K feature sets
- DAR@K : Direction Agreement Ratio (skip unknown directions)
- MRD@K : Mean Rank Difference (missing -> K+1)
- SSS@K : Structural Stability Score

Expected input:
results/{project}/pyexplainer/run_i/explanations.jsonl
"""

import json
import argparse
from pathlib import Path
from itertools import combinations
from typing import Dict, List
import math


# ============================================================
# Basic stats
# ============================================================

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def std(xs):
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


# ============================================================
# Metric primitives
# ============================================================

def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def direction_from_weight(w, eps=1e-12):
    if w > eps:
        return "increases_risk"
    if w < -eps:
        return "decreases_risk"
    return "neutral"


# ============================================================
# Loading
# ============================================================

def load_runs(project: str, k: int):
    """
    Returns:
      runs: List[Dict[commit_id, {
        "features": [f1, f2, ...],
        "dirs": {feature: direction},
        "ranks": {feature: rank}
      }]]
    """
    base_dir = Path(f"results/{project}/pyexplainer")
    runs = []

    for run_dir in sorted(base_dir.glob("run_*")):
        run_data = {}

        with open(run_dir / "explanations.jsonl", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)

                cid = obj.get("commit_id")
                if cid is None:
                    continue

                expl = obj.get("explanation", [])
                if not isinstance(expl, list):
                    continue

                features = []
                dirs = {}
                ranks = {}

                for idx, item in enumerate(expl[:k]):
                    # PyExplainer explanation item:
                    # {"feature": "...", "weight": ...}
                    if not isinstance(item, dict):
                        continue

                    feat = item.get("feature")
                    weight = item.get("weight", 0.0)

                    if not feat:
                        continue

                    features.append(feat)
                    dirs[feat] = direction_from_weight(weight)
                    ranks[feat] = idx + 1

                if features:
                    run_data[cid] = {
                        "features": features,
                        "dirs": dirs,
                        "ranks": ranks,
                    }

        runs.append(run_data)

    return runs


# ============================================================
# Pairwise metrics
# ============================================================

def compute_pairwise(a, b, k):
    """
    a, b: dict with keys features, dirs, ranks
    """
    fa, fb = a["features"], b["features"]

    # J@K
    j = jaccard(fa, fb)

    # DAR@K (skip unknown / neutral)
    inter = set(fa) & set(fb)
    dar_vals = []
    for f in inter:
        da = a["dirs"].get(f)
        db = b["dirs"].get(f)
        if da in {"neutral", None} or db in {"neutral", None}:
            continue
        dar_vals.append(1.0 if da == db else 0.0)
    dar = mean(dar_vals) if dar_vals else 0.0

    # MRD@K (missing -> K+1)
    union = set(fa) | set(fb)
    miss = k + 1
    diffs = []
    for f in union:
        ra = a["ranks"].get(f, miss)
        rb = b["ranks"].get(f, miss)
        diffs.append(abs(ra - rb))
    mrd = mean(diffs) if diffs else 0.0

    # SSS@K
    rank_sim = 1.0 - (mrd / (k + 1))
    rank_sim = max(0.0, min(1.0, rank_sim))
    sss = (j + dar + rank_sim) / 3.0

    return j, dar, mrd, sss


# ============================================================
# Evaluation
# ============================================================

def evaluate(project: str, k: int):
    runs = load_runs(project, k)
    n_runs = len(runs)

    commit_ids = set.intersection(*[set(r.keys()) for r in runs])
    commit_ids = sorted(commit_ids)

    J, D, M, S = [], [], [], []

    for cid in commit_ids:
        for i, j in combinations(range(n_runs), 2):
            ja, da, ma, sa = compute_pairwise(
                runs[i][cid],
                runs[j][cid],
                k
            )
            J.append(ja)
            D.append(da)
            M.append(ma)
            S.append(sa)

    summary = {
        "project": project,
        "method": "pyexplainer",
        "k": k,
        "n_commits": len(commit_ids),
        "n_runs": n_runs,
        "J@K": {"mean": mean(J), "std": std(J)},
        "DAR@K": {"mean": mean(D), "std": std(D)},
        "MRD@K": {"mean": mean(M), "std": std(M)},
        "SSS@K": {"mean": mean(S), "std": std(S)},
    }

    return summary


# ============================================================
# Output
# ============================================================

def save_outputs(summary, project, k):
    out_dir = Path("evaluation/pyexplainer_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"rq1_{project}_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    tex = rf"""
\begin{{table}}[htbp]
\centering
\small
\caption{{RQ1 stability results on {project} (PyExplainer, $K={k}$).}}
\label{{tab:rq1_{project}_pyexplainer}}
\begin{{tabular}}{{lcccc}}
\toprule
Method & $J@{k}$ & $DAR@{k}$ & $MRD@{k}$ & $SSS@{k}$ \\
\midrule
PyExplainer &
{summary["J@K"]["mean"]:.3f}$\pm${summary["J@K"]["std"]:.3f} &
{summary["DAR@K"]["mean"]:.3f}$\pm${summary["DAR@K"]["std"]:.3f} &
{summary["MRD@K"]["mean"]:.3f}$\pm${summary["MRD@K"]["std"]:.3f} &
{summary["SSS@K"]["mean"]:.3f}$\pm${summary["SSS@K"]["std"]:.3f} \\
\bottomrule
\end{{tabular}}
\end{{table}}
""".strip()

    tex_path = out_dir / f"rq1_{project}_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex + "\n")

    print(f"[RQ1] Saved: {json_path}")
    print(f"[RQ1] Saved: {tex_path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    summary = evaluate(args.project, args.k)
    print(json.dumps(summary, indent=2))

    save_outputs(summary, args.project, args.k)


if __name__ == "__main__":
    main()
