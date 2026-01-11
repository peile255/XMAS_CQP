# XMAS-CQP: Explainable Multi-Agent System for Code Quality Prediction

XMAS-CQP is a **reproducible, explanation-oriented pipeline** for software defect and code quality prediction.
It integrates **deterministic preprocessing** with **LLM-based structured explanation generation**, aiming to provide
**faithful, stable, and auditable explanations** for machine learning predictions in Software Defect Prediction (SDP).

This repository accompanies an academic study on **explainable AI (XAI) for code quality prediction** and is designed
to support **full experimental reproducibility**.

---

## ✨ Key Features

- 🔍 **Explainable Code Quality Prediction**
  - Local, feature-level explanations for defect risk predictions
- 🧠 **Multi-Agent Architecture**
  - Deterministic *Preprocessor Agent* + LLM-based *Explainer Agent*
- 📦 **Structured Explanations**
  - All explanations follow a strict JSON Schema
- ♻️ **Reproducibility First**
  - Fixed seeds, deterministic IR, repeated runs, and logged artifacts
- 📊 **Comprehensive Evaluation**
  - Faithfulness, stability, diversity, and hallucination analysis
- ⚖️ **Baseline Comparison**
  - LIME, PyExplainer, CFExplainer, RuleFit

---

## 📁 Repository Structure

```

XMAS-CQP/
├── Baseline/        # Baseline explanation methods (LIME, PyExplainer, CFExplainer, RuleFit)
├── data/            # Raw datasets and feature statistics
├── datasets/        # Preprocessed datasets (JSONL)
├── evaluation/      # RQ1–RQ4 evaluation scripts and result summaries
├── logs/            # Execution and agent interaction logs
├── models/          # Trained JIT / defect prediction models
├── results/         # Explanation outputs and evaluation artifacts
├── scripts/         # Entry scripts for running experiments
└── xmas_cqp/        # Core XMAS-CQP framework (agents, prompts, schemas)

````

---

## 🧠 XMAS-CQP Architecture

XMAS-CQP follows a **two-stage agent design**:

1. **Preprocessor Agent**
   - Deterministic feature extraction
   - Decision Intermediate Representation (Decision IR)
   - No randomness or LLM involvement

2. **Explainer Agent**
   - LLM-based explanation generation
   - Consumes Decision IR only
   - Produces structured JSON explanations under schema constraints

This design enforces a **strict separation between prediction logic and explanation logic**.

---

## 🧪 Research Questions (RQs)

The experimental design addresses the following research questions:

- **RQ1**: What features are most frequently used in explanations, and what risk directions do they imply?
- **RQ2**: Are the generated explanations *faithful* to the underlying predictive model?
- **RQ3**: Are explanations *stable and consistent* across repeated runs?
- **RQ4**: Do LLM-based explanations exhibit *hallucination*, and to what extent?

Each RQ has a corresponding evaluation script under `evaluation/`.

---

## ▶️ Running Experiments

### 1️⃣ Install Dependencies

```bash
pip install -r xmas_cqp/requirements.txt
````

### 2️⃣ Run Explanation Generation (Example)

```bash
python xmas_cqp/cli.py \
  --dataset openstack \
  --model models/openstack_jit_model.joblib \
  --config xmas_cqp/config/explainer.yaml
```

### 3️⃣ Run Evaluations

```bash
python evaluation/rq2_faithfulness.py
python evaluation/rq3_diversity_xmascqp.py
```

---

## 📊 Baselines

The following explanation baselines are implemented for comparison:

* **LIME**
* **PyExplainer**
* **CFExplainer**
* **RuleFit**

Baseline implementations are located in `Baseline/`.

---

## 📄 Output Artifacts

XMAS-CQP generates the following structured outputs:

* `explanations.jsonl` — structured explanation instances
* `summary.json` — aggregated statistics
* `.log` files — execution traces
* `.tex` tables — LaTeX-ready result tables for papers

---

## 🔁 Reproducibility Notes

* Fixed random seeds are used throughout the pipeline
* Multiple runs are executed for stability analysis
* All intermediate artifacts are logged
* Explanation schema validation is enforced

> ⚠️ Note: Large result files and logs may be excluded in the cleaned release version.

---

## 📜 License

This project is released under the **MIT License**.
See `xmas_cqp/LICENSE` for details.

---

## 📌 Citation

If you use XMAS-CQP in your research, please cite the accompanying paper:

```bibtex
@article{XMAS-CQP,
  title   = {XMAS-CQP: A Reproducible Multi-Agent Framework for Explainable Code Quality Prediction},
  author  = {Lei Pei et al.},
  year    = {2026}
}
```

---

## 🤝 Contact

For questions, issues, or collaboration, please open an issue on GitHub.
