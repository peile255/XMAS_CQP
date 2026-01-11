import json
import argparse
import numpy as np


def normalize_features(x: dict) -> dict:
    return {
        k: int(v) if isinstance(v, bool) else float(v)
        for k, v in x.items()
    }


def load_samples(jsonl_path):
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            feats = obj["input"]["features"]
            samples.append(normalize_features(feats))
    return samples


def main():
    parser = argparse.ArgumentParser("Compute Feature Statistics for RQ2")
    parser.add_argument("--data", required=True,
                        help="datasets/processed_<dataset>.jsonl")
    parser.add_argument("--output", required=True,
                        help="data/<dataset>/feature_stats.json")
    args = parser.parse_args()

    samples = load_samples(args.data)

    features = samples[0].keys()
    stats = {
        "median": {},
        "q10": {},
        "q90": {}
    }

    for f in features:
        vals = np.array([x[f] for x in samples], dtype=float)
        stats["median"][f] = float(np.median(vals))
        stats["q10"][f] = float(np.quantile(vals, 0.10))
        stats["q90"][f] = float(np.quantile(vals, 0.90))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("Feature statistics saved to:", args.output)


if __name__ == "__main__":
    main()
