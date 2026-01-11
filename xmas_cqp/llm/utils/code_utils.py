"""
code_utils
==========

Lightweight, semantics-preserving utilities for handling code-like text.

IMPORTANT:
- XMAS-CQP does NOT analyze source code at the explainer stage.
- These utilities MUST NOT introduce assumptions about code structure,
  programming language, or line-level evidence.
- This module exists ONLY for safe text normalization and size control.

Design principles:
- NEVER change semantics
- NEVER infer meaning
- NEVER enable code analysis
- ONLY perform generic text hygiene
"""

from __future__ import annotations

from typing import Optional


# ----------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize text formatting without changing content semantics.

    Operations:
    - Strip trailing whitespace
    - Preserve line structure
    """
    if not text:
        return ""

    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Truncation
# ----------------------------------------------------------------------

TRUNCATION_MARKER = "[TRUNCATED]"


def truncate_text(
    text: str,
    *,
    max_lines: Optional[int] = None,
    max_chars: Optional[int] = None,
    marker: str = TRUNCATION_MARKER,
) -> str:
    """
    Safely truncate text to control length.

    This function:
    - Does NOT assume the text is executable code
    - Does NOT preserve syntactic correctness
    - Is intended ONLY for protecting LLM token limits
    """
    if not text:
        return ""

    truncated = False
    lines = text.splitlines()

    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True

    result = "\n".join(lines)

    if max_chars is not None and len(result) > max_chars:
        result = result[:max_chars]
        truncated = True

    if truncated:
        result = f"{result}\n\n{marker}"

    return result


# ----------------------------------------------------------------------
# Composite helper (optional)
# ----------------------------------------------------------------------

def prepare_text_for_llm(
    text: str,
    *,
    max_lines: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> str:
    """
    Apply a minimal, reproducible preprocessing pipeline for LLM input.

    Steps:
    1. Normalize whitespace
    2. Truncate to size limits if necessary

    This function intentionally:
    - Does NOT add line numbers
    - Does NOT detect language
    - Does NOT analyze structure
    """
    normalized = normalize_text(text)
    return truncate_text(
        normalized,
        max_lines=max_lines,
        max_chars=max_chars,
    )
