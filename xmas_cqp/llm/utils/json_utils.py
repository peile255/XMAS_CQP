"""
json_utils
==========

Utilities for JSON schema loading, validation, and error analysis
in XMAS-CQP.

Design principles:
- Schema-constrained generation
- Fail fast, fail explicitly
- Validation errors are first-class experimental signals
- Minimal assumptions, maximal transparency
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

from jsonschema import Draft7Validator, ValidationError


# ----------------------------------------------------------------------
# Schema loading & caching
# ----------------------------------------------------------------------

_VALIDATOR_CACHE: Dict[int, Draft7Validator] = {}


def load_json_schema(path: str | Path) -> Dict[str, Any]:
    """
    Load a JSON schema from disk.

    Parameters
    ----------
    path : str or Path
        Path to the JSON schema file.

    Returns
    -------
    dict
        Parsed JSON schema.

    Raises
    ------
    FileNotFoundError
        If the schema file does not exist.
    ValueError
        If the file is not valid JSON.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON schema not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            schema = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON schema file: {path}"
        ) from e

    if not isinstance(schema, dict):
        raise ValueError(
            f"Invalid JSON schema structure (expected object): {path}"
        )

    return schema


def get_validator(schema: Dict[str, Any]) -> Draft7Validator:
    """
    Get a cached Draft7Validator for the given schema.

    Validators are cached by object identity to:
    - Avoid repeated construction
    - Ensure consistent validation behavior
    """
    key = id(schema)
    validator = _VALIDATOR_CACHE.get(key)

    if validator is None:
        validator = Draft7Validator(schema)
        _VALIDATOR_CACHE[key] = validator

    return validator


# ----------------------------------------------------------------------
# Validation error formatting
# ----------------------------------------------------------------------

def _format_error(err: ValidationError) -> Dict[str, Any]:
    """
    Convert a jsonschema ValidationError into a structured dict.
    """
    return {
        "path": ".".join(str(p) for p in err.path) or "<root>",
        "message": err.message,
        "validator": err.validator,
        "validator_value": err.validator_value,
    }


# ----------------------------------------------------------------------
# Schema validation
# ----------------------------------------------------------------------

def validate_json(
    data: Dict[str, Any],
    schema: Dict[str, Any],
    *,
    strict: bool = True,
) -> None:
    """
    Validate JSON data against a schema.

    Parameters
    ----------
    data : dict
        JSON object to validate.
    schema : dict
        JSON schema definition.
    strict : bool
        If True, raise an exception on validation failure.
        If False, silently return.

    Raises
    ------
    ValueError
        If validation fails and strict=True.
    """
    errors = try_validate_json(data, schema)

    if errors and strict:
        messages = [
            f"{e['path']}: {e['message']}"
            for e in errors
        ]
        raise ValueError(
            "JSON schema validation failed:\n"
            + "\n".join(messages)
        )


def try_validate_json(
    data: Dict[str, Any],
    schema: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Validate JSON data against a schema.

    Returns
    -------
    list of dict
        Structured validation errors.
        Empty list indicates the data is valid.
    """
    validator = get_validator(schema)
    return [
        _format_error(e)
        for e in validator.iter_errors(data)
    ]


# ----------------------------------------------------------------------
# Defensive helpers
# ----------------------------------------------------------------------

def ensure_json_object(obj: Any) -> Dict[str, Any]:
    """
    Ensure the given object is a JSON object (dict).
    """
    if not isinstance(obj, dict):
        raise TypeError(
            f"Expected JSON object (dict), got {type(obj).__name__}"
        )
    return obj


def pretty_json(data: Dict[str, Any], indent: int = 2) -> str:
    """
    Render a JSON object as a pretty-printed string.
    """
    return json.dumps(data, ensure_ascii=False, indent=indent)
