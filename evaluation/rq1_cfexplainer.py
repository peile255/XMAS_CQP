"""
rq1_cfexplainer.py

RQ1: Run-to-run stability evaluation for CfExplainer.

Metrics:
- J@K  : Jaccard similarity of top-K features
- DAR@K: Direction Agreement Ratio
- MRD@K: Mean Rank Difference
- SSS@K: Structural Stability Score (weighted)

Output:
- evaluation/cfexplainer_results/rq1_<project>_summary.json
- evaluation/cfexplainer_results/rq1_<project>_table.tex
"""

import json
import argparse
from pathlib import Path
from itertools import combinations
import numpy as np


# ============================================================
# Config
# ============================================================

N_RUNS = 5
DEFAULT_K = 5

# SSS weights (MUST be consistent across all methods)
ALPHA_J = 0.4
BETA_D  = 0.2
GAMMA_M = 0.4


# ============================================================
# Metric definitions (IDENTICAL to LIME / PyExplainer)
# ============================================================

def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dar(a, b):
    """
    Direction Agreement Ratio
    a, b: [(feature, weight), ...]
    """
    if not a or not b:
        return 0.0

    total = min(len(a), len(b))
    agree = 0

    for (fa, wa), (fb, wb) in zip(a, b):
        if fa == fb and np.sign(wa) == np.sign(wb):
            agree += 1

    return agree / total if total > 0 else 0.0


def mrd(a, b):
    """
    Mean Rank Difference
    """
    rank_a = {f: i for i, (f, _) in enumerate(a)}
    rank_b = {f: i for i, (f, _) in enumerate(b)}

    common = set(rank_a) & set(rank_b)
    if not common:
        return len(a)

    return float(np.mean([abs(rank_a[f] - rank_b[f]) for f in common]))


def sss(j, d, m, k):
    """
    Structural Stability Score (weighted)
    """
    rank_stability = 1.0 - m / (k + 1)
    return (
        ALPHA_J * j +
        BETA_D  * d +
        GAMMA_M * rank_stability
    )


# ============================================================
# Utilities
# ============================================================

def load_run(path: Path):
    """
    Load one run of CfExplainer explanations.

    Robust to multiple explanation formats.
    """
    data = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = obj["commit_id"]

            parsed = []
            for item in obj["explanation"]:

                # ---------- case 1: dict ----------
                if isinstance(item, dict):
                    cond = item.get("rule") or item.get("condition")
                    weight = item.get("weight")

                # ---------- case 2: list / tuple ----------
                elif isinstance(item, (list, tuple)):
                    cond = item[0]
                    weight = item[1]

                else:
                    continue  # skip unknown format

                # extract feature name from condition string
                # e.g. "la > 10" → "la"
                feature = cond.split()[0]
                parsed.append((feature, float(weight)))

            data[cid] = parsed

    return data

# ============================================================
# Evaluation
# ============================================================

def evaluate(project: str, k: int):
    root = Path("results") / project / "cfexplainer"

    runs = []
    for i in range(1, N_RUNS + 1):
        p = root / f"run_{i}" / "explanations.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        runs.append(load_run(p))

    commit_ids = list(runs[0].keys())

    J, D, M, S = [], [], [], []

    for cid in commit_ids:
        for i, j in combinations(range(N_RUNS), 2):
            a = runs[i][cid][:k]
            b = runs[j][cid][:k]

            ja = jaccard([x[0] for x in a], [x[0] for x in b])
            da = dar(a, b)
            ma = mrd(a, b)
            sa = sss(ja, da, ma, k)

            J.append(ja)
            D.append(da)
            M.append(ma)
            S.append(sa)

    def stat(x):
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
        }

    summary = {
        "project": project,
        "method": "cfexplainer",
        "k": k,
        "n_commits": len(commit_ids),
        "n_runs": N_RUNS,
        "J@K": stat(J),
        "DAR@K": stat(D),
        "MRD@K": stat(M),
        "SSS@K": stat(S),
    }

    return summary


# ============================================================
# Output
# ============================================================

def save_outputs(summary, project):
    out_dir = Path("evaluation") / "cfexplainer_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- JSON ----------
    json_path = out_dir / f"rq1_{project}_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ---------- LaTeX ----------
    tex_path = out_dir / f"rq1_{project}_table.tex"
    s = summary

    tex = rf"""
\begin{{table}}[htbp]
\centering
\small
\caption{{RQ1 stability results on {project} (CfExplainer, $K={s['k']}$).}}
\label{{tab:rq1_{project}_cfexplainer}}
\begin{{tabular}}{{lcccc}}
\toprule
Method & $J@{s['k']}$ & $DAR@{s['k']}$ & $MRD@{s['k']}$ & $SSS@{s['k']}$ \\
\midrule
CfExplainer &
{s['J@K']['mean']:.3f}$\pm${s['J@K']['std']:.3f} &
{s['DAR@K']['mean']:.3f}$\pm${s['DAR@K']['std']:.3f} &
{s['MRD@K']['mean']:.3f}$\pm${s['MRD@K']['std']:.3f} &
{s['SSS@K']['mean']:.3f}$\pm${s['SSS@K']['std']:.3f} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex.strip())

    print(f"[RQ1] Saved: {json_path}")
    print(f"[RQ1] Saved: {tex_path}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args()

    summary = evaluate(args.project, args.k)
    print(json.dumps(summary, indent=2))

    save_outputs(summary, args.project)
