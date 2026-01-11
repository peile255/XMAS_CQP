from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

# ============================================================
# FIXED imports: unified xmas_cqp namespace
# ============================================================
from xmas_cqp.llm.openai_client import OpenAIClient
from xmas_cqp.llm.utils.json_utils import load_json_schema, validate_json


class ExplainerAgent:
    """
    Explainer Agent for XMAS-CQP (Model-Centric Decision Explanation).

    This agent explains WHY a predictive model produced a given prediction,
    strictly based on model-visible input features and model outputs.

    Design principles:
    - No access to raw source code
    - No access to external knowledge
    - No feature invention
    - Deterministic behavior (temperature = 0)
    """

    def __init__(
        self,
        *,
        client: OpenAIClient,
        schema_path: str | Path,
        system_prompt_path: str | Path,
        user_prompt_path: str | Path,
        model_name: str,
        max_retries: int = 2,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.max_retries = max_retries

        # ----------------------------
        # Load explanation schema
        # ----------------------------
        self.schema = load_json_schema(schema_path)

        # ----------------------------
        # Load prompts
        # ----------------------------
        self.system_prompt = Path(system_prompt_path).read_text(
            encoding="utf-8"
        )
        self.user_prompt_template = Path(user_prompt_path).read_text(
            encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        *,
        record: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a model-centric decision explanation.

        Expected input record structure:
        {
          "task": "explain_model_prediction",
          "input": {
            "features": {...},
            "model_output": {
              "prediction": str,
              "probability": float
            }
          },
          "metadata": {...}
        }
        """

        # ----------------------------
        # Input validation
        # ----------------------------
        if "input" not in record:
            raise KeyError("Missing 'input' field in record")

        input_block = record["input"]

        if "features" not in input_block:
            raise KeyError("Missing 'features' in record['input']")

        if "model_output" not in input_block:
            raise KeyError("Missing 'model_output' in record['input']")

        features = input_block["features"]
        model_output = input_block["model_output"]
        metadata = record.get("metadata", {})

        # ----------------------------
        # Build user prompt
        # ----------------------------
        user_prompt = self._build_user_prompt(
            features=features,
            model_output=model_output,
            metadata=metadata,
        )

        last_error: Optional[Exception] = None

        # ----------------------------
        # LLM invocation with retry
        # ----------------------------
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.generate_json(
                    model=self.model_name,
                    system_prompt=self.system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.0,      # critical for RQ1 stability
                    strict=True,
                    return_raw=False,
                )

                # ----------------------------
                # Schema validation
                # ----------------------------
                validate_json(response, self.schema)

                return response

            except Exception as e:
                last_error = e
                if attempt >= self.max_retries:
                    break

        raise RuntimeError(
            "ExplainerAgent failed after retries. "
            f"Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_user_prompt(
        self,
        *,
        features: Dict[str, Any],
        model_output: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> str:
        """
        Render the user prompt from template.

        All content is serialized explicitly to avoid hidden context
        and ensure full reproducibility across runs.
        """
        return self.user_prompt_template.format(
            features=json.dumps(
                features, ensure_ascii=False, indent=2
            ),
            model_output=json.dumps(
                model_output, ensure_ascii=False, indent=2
            ),
            meta=json.dumps(
                metadata, ensure_ascii=False, indent=2
            ),
        )
