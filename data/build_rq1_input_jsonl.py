from pathlib import Path
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier

# =========================
# Project root
# =========================
ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = ROOT / "data" / "rq1_samples" / "qt_rq1_sample.csv"
OUT_PATH = ROOT / "data" / "rq1_samples" / "qt_rq1_input.jsonl"


# =========================
# Label construction (JIT)
# =========================
def derive_buggy(row):
    return 1 if row["bugcount"] > 0 else 0

DROP_COLS = ["commit_id", "author_date"]

# 1. Load CSV
df = pd.read_csv(CSV_PATH)

# 2. Derive label
df["buggy"] = df.apply(derive_buggy, axis=1)

y = df["buggy"].astype(int)

# 3. Prepare features
X = df.drop(columns=DROP_COLS + ["buggy", "bugcount", "fixcount"])

# 4. Train deterministic model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)
model.fit(X, y)

# 5. Predict
probs = model.predict_proba(X)
preds = model.predict(X)

# 6. Build JSONL
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for i in range(len(df)):
        record = {
            "task": "explain_model_prediction",
            "input": {
                "features": X.iloc[i].to_dict(),
                "model_output": {
                    "prediction": "buggy" if preds[i] == 1 else "clean",
                    "probability": float(probs[i][preds[i]])
                }
            },
            "metadata": {
                "dataset": "qt",
                "sample_id": i,
                "commit_id": df.iloc[i]["commit_id"]
            }
        }
        f.write(json.dumps(record) + "\n")

print(f"[OK] Saved {OUT_PATH}")
