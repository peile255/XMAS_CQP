# XMAS-CQP

**XMAS-CQP (eXplainable Model-Aware Software Code Quality Prediction)**  
is a model-centric, reproducible explanation pipeline for software defect / code quality prediction.

The system strictly separates **decision construction** and **decision explanation**, enabling
faithful, schema-constrained, and repeatable explanations for empirical evaluation.

---

## 📁 Project Structure

```

xmas_cqp/
├── agents/                # Preprocessor & Explainer agents
├── cli.py                 # CLI entry point
├── config/
│   └── explainer.yaml     # Main experiment configuration
├── prompts/               # System & user prompts
├── schemas/               # JSON schema for explanations
├── llm/                   # LLM client & utilities
└── README.md

````

---

## 📥 Input Format

XMAS-CQP expects **JSONL input (NOT CSV)**.

Each line represents one model prediction task:

```json
{
  "task": "explain_model_prediction",
  "input": {
    "features": { "...": "model-visible features" },
    "model_output": {
      "prediction": "clean | buggy",
      "probability": 0.0
    }
  },
  "metadata": {
    "dataset": "openstack",
    "sample_id": 0,
    "commit_id": "..."
  }
}
````

---

## 📤 Output Structure

Each repeated run is written to an isolated directory:

```
results/{dataset}/{project_version}/run_{run_id}/
├── processed.jsonl        # Deterministic decision IR
├── explanations.jsonl     # Schema-constrained explanations
└── errors.jsonl           # Failure records (if any)
```

---

## ▶️ Running Experiments (Windows / PowerShell)

All commands below are written for **Windows PowerShell**.

General command format:

```powershell
python -m xmas_cqp.cli run `
  --config xmas_cqp/config/explainer.yaml `
  --dataset <DATASET_NAME> `
  --input <INPUT_JSONL> `
  --run_id <RUN_ID>
```

---

## 🔁 Repeated Runs (N = 5)

The following commands perform **five repeated runs** on the same dataset and input,
used for **stability, robustness, and variance analysis**.

---

### ▶️ Run 1

```powershell
python -m xmas_cqp.cli run `
  --config xmas_cqp/config/explainer.yaml `
  --dataset openstack `
  --input data/rq1_samples/openstack_rq1_input.jsonl `
  --run_id 1
```

---

### ▶️ Run 2

```powershell
python -m xmas_cqp.cli run `
  --config xmas_cqp/config/explainer.yaml `
  --dataset openstack `
  --input data/rq1_samples/openstack_rq1_input.jsonl `
  --run_id 2
```

---

### ▶️ Run 3

```powershell
python -m xmas_cqp.cli run `
  --config xmas_cqp/config/explainer.yaml `
  --dataset openstack `
  --input data/rq1_samples/openstack_rq1_input.jsonl `
  --run_id 3
```

---

### ▶️ Run 4

```powershell
python -m xmas_cqp.cli run `
  --config xmas_cqp/config/explainer.yaml `
  --dataset openstack `
  --input data/rq1_samples/openstack_rq1_input.jsonl `
  --run_id 4
```

---

### ▶️ Run 5

```powershell
python -m xmas_cqp.cli run `
  --config xmas_cqp/config/explainer.yaml `
  --dataset openstack `
  --input data/rq1_samples/openstack_rq1_input.jsonl `
  --run_id 5
```

---

## 🔬 Reproducibility Notes

* All preprocessing is **deterministic**
* Each `run_id` produces **independent outputs**
* Explanation randomness (LLM) is evaluated via repeated runs
* All failures are explicitly logged in `errors.jsonl`

This setup supports empirical analysis of:

* Explanation stability
* Feature attribution consistency
* Hallucination rate
* Run-to-run variance

---

## ⚠️ Common Pitfalls

* ❌ Do NOT use CSV as input
* ❌ Do NOT reuse the same `run_id`
* ✅ Always use JSONL input
* ✅ Increment `run_id` for each repetition

---

## 📖 Intended Use

XMAS-CQP is designed for:

* Explainable Software Defect Prediction (SDP)
* Faithfulness and stability evaluation of XAI methods
* RQ-driven empirical software engineering research
* Reproducible academic experimentation

---

## 📜 License

For research and academic use only.
