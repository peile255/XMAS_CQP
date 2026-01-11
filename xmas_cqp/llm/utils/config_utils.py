"""
config_utils
============

Utilities for loading and validating experiment configuration files (YAML)
in XMAS-CQP.

Design principles:
- Explicit configuration over CLI flags
- Deterministic, reproducible experiments
- Fail fast on malformed or incomplete configs
- Minimal assumptions about experiment design
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Iterable

import yaml


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """
    Load and minimally validate a YAML experiment configuration file.

    Parameters
    ----------
    path : str or Path
        Path to YAML config file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If the YAML file cannot be parsed or is structurally invalid.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(
            f"Failed to parse YAML config file: {path}"
        ) from e

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid config format (expected a mapping at top level): {path}"
        )

    _validate_required_sections(
        config,
        required_sections=[
            "pipeline",
            "preprocessor",
            "explainer",
            "outputs",
            "logging",
        ],
        config_path=path,
    )

    return config


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _validate_required_sections(
    config: Dict[str, Any],
    *,
    required_sections: Iterable[str],
    config_path: Path,
) -> None:
    """
    Ensure required top-level sections exist in the config.

    This is a MINIMAL structural validation intended to:
    - Catch obvious configuration errors early
    - Avoid over-constraining experimental flexibility

    It does NOT validate nested schemas or values.
    """

    missing = [
        key for key in required_sections
        if key not in config
    ]

    if missing:
        raise ValueError(
            "Missing required top-level config section(s): "
            f"{', '.join(missing)} "
            f"in config file {config_path}"
        )
