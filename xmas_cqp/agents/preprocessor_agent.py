"""
Preprocessor Agent for XMAS-CQP (Model Decision Explanation Version).

Purpose:
  Convert raw metric-based inputs + model predictions into a structured
  intermediate representation (IR) for downstream explanation.

This agent DOES NOT analyze source code.
This agent DOES NOT generate explanations.
This agent DOES NOT call an LLM.

Its sole responsibility is to:
  - Normalize model-visible features
  - Attach model outputs (prediction + confidence)
  - Prevent label leakage
  - Emit a clean, stable IR for explainer_agent

Expected output:
{
  "task": "explain_model_prediction",
  "input": {
    "features": { ... },
    "model_output": {
      "prediction": str,
      "probability": float
    }
  },
  "metadata": { ... }
}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


# ----------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------

class PreprocessorError(RuntimeError):
    """Base class for preprocessing errors."""


class MissingFieldError(PreprocessorError):
    """Raised when required fields are missing."""


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

@dataclass
class PreprocessorAgentConfig:
    """
    Configuration for model-decision preprocessor.
    """
    require_probability: bool = True
    probability_key: str = "probability"
    prediction_key: str = "prediction"
    label_keys_to_strip: tuple = (
        "RealBug",
        "RealBugCount",
        "HeuBug",
        "HeuBugCount",
    )


# ----------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------

class PreprocessorAgent:
    """
    Preprocessor Agent for model decision explanation.

    This agent is intentionally deterministic and LLM-free.
    """

    def __init__(
        self,
        config: Optional[PreprocessorAgentConfig] = None,
    ) -> None:
        self.cfg = config or PreprocessorAgentConfig()

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def process(
        self,
        features: Dict[str, Any],
        model_output: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Prepare a single instance for explanation.

        Args:
          features:
            Dictionary of model-visible input features (metrics only).
            MUST NOT contain ground-truth labels.

          model_output:
            {
              "prediction": str,
              "probability": float
            }

          metadata:
            Optional contextual metadata (project, version, etc.)

        Returns:
          A JSON-serializable dict suitable for processed.jsonl.
        """

        metadata = metadata or {}

        # ------------------------------
        # 1. Sanitize features
        # ------------------------------

        clean_features: Dict[str, Any] = {}
        for k, v in features.items():
            if k in self.cfg.label_keys_to_strip:
                continue
            clean_features[k] = v

        if not clean_features:
            raise MissingFieldError("No usable features provided.")

        # ------------------------------
        # 2. Validate model output
        # ------------------------------

        if self.cfg.prediction_key not in model_output:
            raise MissingFieldError(
                f"Model output missing '{self.cfg.prediction_key}'."
            )

        prediction = model_output[self.cfg.prediction_key]

        probability = model_output.get(self.cfg.probability_key, None)
        if self.cfg.require_probability:
            if probability is None:
                raise MissingFieldError(
                    f"Model output missing '{self.cfg.probability_key}'."
                )
            try:
                probability = float(probability)
            except Exception:
                probability = 0.0

        # Clamp probability to [0, 1]
        probability = max(0.0, min(1.0, probability))

        # ------------------------------
        # 3. Emit IR
        # ------------------------------

        return {
            "task": "explain_model_prediction",
            "input": {
                "features": clean_features,
                "model_output": {
                    "prediction": str(prediction),
                    "probability": probability,
                },
            },
            "metadata": metadata,
        }
