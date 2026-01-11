"""
logging_utils
=============

Centralized logging utilities for XMAS-CQP.

Design principles:
- Logs are experimental evidence
- Runs must be traceable and reproducible
- Multi-stage and multi-agent friendly
- Minimal magic, maximal clarity
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import uuid


# ----------------------------------------------------------------------
# Logger creation
# ----------------------------------------------------------------------

def create_logger(
    name: str = "xmas_cqp",
    log_dir: Optional[str | Path] = None,
    level: int = logging.INFO,
    run_id: Optional[str] = None,
) -> logging.Logger:
    """
    Create or retrieve a configured logger.

    Logger creation is idempotent:
    repeated calls with the same name will return the same logger
    without duplicating handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # If handlers already exist, assume logger is configured
    if getattr(logger, "_xmas_cqp_initialized", False):
        return logger

    run_id = run_id or generate_run_id()
    formatter = _create_formatter(run_id)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"{name}_{run_id}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False

    # Attach run_id for downstream access (explicit, intentional)
    logger.run_id = run_id  # type: ignore[attr-defined]
    logger._xmas_cqp_initialized = True  # type: ignore[attr-defined]

    logger.info(f"Logger initialized | run_id={run_id}")

    return logger


# ----------------------------------------------------------------------
# Formatter
# ----------------------------------------------------------------------

def _create_formatter(run_id: str) -> logging.Formatter:
    """
    Create a consistent formatter embedding run_id.
    """
    return logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)-8s | "
            "%(name)s | "
            f"run={run_id} | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ----------------------------------------------------------------------
# Run ID utilities
# ----------------------------------------------------------------------

def generate_run_id() -> str:
    """
    Generate a globally unique, time-sortable run identifier.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{timestamp}-{short_uuid}"


# ----------------------------------------------------------------------
# Experiment boundary helpers
# ----------------------------------------------------------------------

def log_experiment_start(
    logger: logging.Logger,
    *,
    config_path: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log the start of an experiment run.
    """
    msg = "Experiment started"
    if config_path:
        msg += f" | config={config_path}"
    logger.info(msg)


def log_experiment_end(
    logger: logging.Logger,
    *,
    summary: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log the end of an experiment run.
    """
    if summary:
        logger.info(f"Experiment finished | summary={summary}")
    else:
        logger.info("Experiment finished")


# ----------------------------------------------------------------------
# Stage / agent helpers
# ----------------------------------------------------------------------

def log_stage(
    logger: logging.Logger,
    *,
    stage: str,
    agent: Optional[str] = None,
    message: str,
) -> None:
    """
    Log a structured stage or agent-level message.
    """
    prefix = f"[stage={stage}]"
    if agent:
        prefix += f"[agent={agent}]"
    logger.info(f"{prefix} {message}")


# ----------------------------------------------------------------------
# Exception helper
# ----------------------------------------------------------------------

def log_exception(
    logger: logging.Logger,
    exc: Exception,
    *,
    stage: Optional[str] = None,
    agent: Optional[str] = None,
    context: Optional[str] = None,
) -> None:
    """
    Log an exception with structured experimental context.
    """
    parts = []
    if stage:
        parts.append(f"stage={stage}")
    if agent:
        parts.append(f"agent={agent}")
    if context:
        parts.append(context)

    prefix = " | ".join(parts)
    if prefix:
        logger.error(f"{prefix} | {exc}", exc_info=True)
    else:
        logger.error(str(exc), exc_info=True)
