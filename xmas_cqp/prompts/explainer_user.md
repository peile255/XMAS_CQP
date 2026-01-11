You are given a model prediction task.

INPUT
=====
Model-visible features:
{features}

Model output:
{model_output}

Metadata (optional, may be incomplete):
{meta}

TASK
====
Produce a model decision explanation as a SINGLE JSON object.

OUTPUT FORMAT (STRICT)
======================
The output MUST strictly conform to the JSON schema and include:

- explanation_structure (string, MUST be "model-centric")
- prediction (string)
- confidence (number between 0 and 1)
- summary (concise explanation)
- key_factors (ordered list)
- counterfactual_hint (optional)

For key_factors:
- Rank the factors by importance (rank = 1 is most influential).
- Assign an importance score in [0.0, 1.0].
- Importance scores should be relative and normalized.
- Use ONLY the provided features.

RULES
=====
- Output a SINGLE JSON object.
- Do NOT wrap the output inside an "explanation" field.
- Do NOT include any other top-level or nested fields.
- Do NOT mention source code, training data, or ground truth.
- Do NOT invent features.
- Base all reasoning strictly on the given features and model output.

Now generate the explanation JSON.
