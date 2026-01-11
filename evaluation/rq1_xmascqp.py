"""
rq1_xmascqp.py

RQ1: Run-to-run stability evaluation for XMAS-CQP.

Unified metrics (aligned with RQ1 evaluator):
- J@K   : Jaccard similarity of top-K features
- DAR@K : Direction Agreement Ratio (skip neutral/unknown)
- MRD@K : Mean Rank Difference (missing -> K+1)
- SSS@K : Structural Stability Score
"""

import json
import argparse
from pathlib import Path
from itertools import combinations
import math


# ============================================================
# Config
# ============================================================

N_RUNS = 5
DEFAULT_K = 5


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
# Metrics
# ============================================================

def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def compute_pairwise(a, b, k):
    """
    a, b: [(feature, direction), ...] already truncated to K
    """
    fa = [x[0] for x in a]
    fb = [x[0] for x in b]

    da = {f: d for f, d in a}
    db = {f: d for f, d in b}

    # ---------- J@K ----------
    j = jaccard(fa, fb)

    # ---------- DAR@K (skip neutral/unknown) ----------
    inter = set(fa) & set(fb)
    dar_vals = []
    for f in inter:
        d1 = da.get(f)
        d2 = db.get(f)
        if d1 in {"neutral", None} or d2 in {"neutral", None}:
            continue
        dar_vals.append(1.0 if d1 == d2 else 0.0)
    dar = mean(dar_vals) if dar_vals else 0.0

    # ---------- MRD@K (union, missing -> K+1) ----------
    union = set(fa) | set(fb)
    ra = {f: i + 1 for i, f in enumerate(fa)}
    rb = {f: i + 1 for i, f in enumerate(fb)}
    miss = k + 1

    diffs = []
    for f in union:
        diffs.append(abs(ra.get(f, miss) - rb.get(f, miss)))
    mrd = mean(diffs) if diffs else 0.0

    # ---------- SSS@K ----------
    rank_sim = 1.0 - (mrd / (k + 1))
    rank_sim = max(0.0, min(1.0, rank_sim))
    sss = (j + dar + rank_sim) / 3.0

    return j, dar, mrd, sss


# ============================================================
# Loading
# ============================================================

def load_run(path: Path, k: int):
    """
    Load one run of XMAS-CQP explanations.

    Expected format:
    {
      "commit_id": "...",
      "key_factors": [
        {"rank": 1, "feature": "...", "direction": "..."},
        ...
      ]
    }
    """
    data = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            cid = obj.get("commit_id")
            if cid is None:
                continue

            kf = obj.get("key_factors", [])
            parsed = []

            for item in kf[:k]:
                feat = item.get("feature")
                dire = item.get("direction", "neutral")
                if feat:
                    parsed.append((feat, dire))

            if parsed:
                data[cid] = parsed

    return data


# ============================================================
# Evaluation
# ============================================================

def evaluate(project: str, k: int):
    root = Path("results") / project / "default"

    runs = []
    for i in range(1, N_RUNS + 1):
        p = root / f"run_{i}" / "explanations.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        runs.append(load_run(p, k))

    commit_ids = set.intersection(*[set(r.keys()) for r in runs])
    commit_ids = sorted(commit_ids)

    J, D, M, S = [], [], [], []

    for cid in commit_ids:
        for i, j in combinations(range(N_RUNS), 2):
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
        "method": "xmascqp",
        "k": k,
        "n_commits": len(commit_ids),
        "n_runs": N_RUNS,
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
    out_dir = Path("evaluation/xmascqp_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"rq1_{project}_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    tex = rf"""
\begin{{table}}[htbp]
\centering
\small
\caption{{RQ1 stability results on {project} (XMAS-CQP, $K={k}$).}}
\label{{tab:rq1_{project}_xmascqp}}
\begin{{tabular}}{{lcccc}}
\toprule
Method & $J@{k}$ & $DAR@{k}$ & $MRD@{k}$ & $SSS@{k}$ \\
\midrule
XMAS-CQP &
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args()

    summary = evaluate(args.project, args.k)
    print(json.dumps(summary, indent=2))

    save_outputs(summary, args.project, args.k)
