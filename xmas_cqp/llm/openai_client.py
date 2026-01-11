from __future__ import annotations

import json
import os
from typing import Dict, Any, Optional, Tuple

from openai import OpenAI


# ----------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------

class OpenAIClientError(RuntimeError):
    """Base class for OpenAI client errors."""


class EmptyResponseError(OpenAIClientError):
    """Raised when the model returns no textual output."""


class JSONExtractionError(OpenAIClientError):
    """Raised when JSON cannot be extracted or parsed."""


# ----------------------------------------------------------------------
# Client
# ----------------------------------------------------------------------

class OpenAIClient:
    """
    Thin, research-oriented wrapper over the OpenAI Responses API.

    Design goals:
    - JSON-first generation (schema-constrained downstream)
    - Deterministic, explicit failure modes
    - Multi-agent safe (structured prompts supported)
    - Faithfulness-first (no implicit post-processing)
    """

    # ============================================================
    # Construction
    # ============================================================

    def __init__(
        self,
        api_key: str,
        timeout: int = 60,
    ):
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not provided or empty."
            )

        self.client = OpenAI(
            api_key=api_key.strip(),
            timeout=timeout,
        )

    @classmethod
    def from_env(cls, *, timeout: int = 60) -> "OpenAIClient":
        """
        Construct client using OPENAI_API_KEY from environment.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. "
                "Please set it as an environment variable."
            )
        return cls(api_key=api_key, timeout=timeout)

    # ============================================================
    # Public API
    # ============================================================

    def generate_json(
        self,
        *,
        model: str,
        system_prompt: Any,
        user_prompt: Any,
        temperature: float = 0.0,
        strict: bool = True,
        return_raw: bool = False,
    ) -> Dict[str, Any] | Tuple[Dict[str, Any], str]:
        """
        Generate a JSON object from the model.

        Parameters
        ----------
        model : str
            Model name.
        system_prompt : Any
            System prompt (string or structured object).
        user_prompt : Any
            User prompt (string or structured object).
        temperature : float
            Sampling temperature.
        strict : bool
            If True, require the output to be pure JSON.
            If False, extract the first JSON object defensively.
        return_raw : bool
            If True, also return raw model output text.

        Returns
        -------
        dict or (dict, str)
            Parsed JSON object (and optional raw text).
        """

        system_prompt_str = self._ensure_str(system_prompt)
        user_prompt_str = self._ensure_str(user_prompt)

        response = self.client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        system_prompt_str
                        + "\n\n"
                        + "You MUST output a SINGLE valid JSON object. "
                        + "Do NOT include any extra text."
                    ),
                },
                {
                    "role": "user",
                    "content": user_prompt_str,
                },
            ],
            temperature=temperature,
        )

        json_obj, raw_text = self._extract_json(
            response,
            strict=strict,
        )

        if return_raw:
            return json_obj, raw_text
        return json_obj

    # Backward-compatible alias (used by older code paths if any)
    def explain(
        self,
        model: str,
        system_prompt: Any,
        user_prompt: Any,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        return self.generate_json(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            strict=True,
            return_raw=False,
        )

    # ============================================================
    # Internals
    # ============================================================

    def _ensure_str(self, value: Any) -> str:
        """
        Ensure prompt content is a string.

        Structured objects (dict / list) are JSON-serialized.
        This is REQUIRED for multi-agent pipelines where
        intermediate representations are structured data.
        """
        if value is None:
            return ""

        if isinstance(value, str):
            return value

        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except Exception:
            return str(value)

    def _extract_json(
        self,
        response,
        *,
        strict: bool = True,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Extract JSON object and raw text from Responses API output.
        """

        if not response.output:
            raise EmptyResponseError("Empty OpenAI response.")

        raw_text_parts = []

        for message in response.output:
            for block in message.content:
                if block.type == "output_text":
                    raw_text_parts.append(block.text)

        raw_text = "".join(raw_text_parts).strip()

        if not raw_text:
            raise EmptyResponseError("No textual output found in response.")

        # Strict mode: output must be pure JSON
        if strict:
            try:
                return json.loads(raw_text), raw_text
            except json.JSONDecodeError as e:
                raise JSONExtractionError(
                    "Model output is not valid JSON (strict mode)."
                ) from e

        # Relaxed mode: extract first JSON object
        try:
            json_text = self._extract_first_json_object(raw_text)
            return json.loads(json_text), raw_text
        except Exception as e:
            raise JSONExtractionError(
                "Failed to extract JSON object (relaxed mode)."
            ) from e

    @staticmethod
    def _extract_first_json_object(text: str) -> str:
        """
        Extract the first top-level {...} JSON object from text.
        """
        start = text.find("{")
        if start == -1:
            raise JSONExtractionError("No '{' found in text.")

        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        raise JSONExtractionError("Unbalanced JSON braces.")
