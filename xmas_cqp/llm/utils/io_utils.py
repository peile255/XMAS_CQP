"""
io_utils
========

Utilities for reading datasets and writing results in XMAS-CQP.

Design principles:
- Line-oriented (JSONL) for scalability
- Streaming-friendly (constant memory)
- Fault-tolerant (failures are first-class signals)
- Research-friendly (explicit, reproducible formats)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Iterable, Iterator, Optional


# ----------------------------------------------------------------------
# JSONL Reading
# ----------------------------------------------------------------------

def read_jsonl(
    path: str | Path,
    *,
    skip_invalid: bool = False,
    error_path: Optional[str | Path] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Lazily read a JSON Lines (JSONL) file.

    Parameters
    ----------
    path : str or Path
        Path to the JSONL file.
    skip_invalid : bool
        If True, invalid JSON lines are skipped and optionally logged.
        If False, invalid JSON raises an exception immediately.
    error_path : Optional[str or Path]
        If provided, invalid JSON lines are written to this file
        as structured error records.

    Yields
    ------
    dict
        Parsed JSON objects, one per line.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                if not skip_invalid:
                    raise ValueError(
                        f"Invalid JSON on line {line_num} in {path}"
                    ) from e

                if error_path is not None:
                    append_jsonl(
                        error_path,
                        {
                            "stage": "read_jsonl",
                            "line_number": line_num,
                            "raw_line": raw_line.rstrip("\n"),
                            "error": {
                                "type": type(e).__name__,
                                "message": str(e),
                            },
                        },
                    )
                continue


# ----------------------------------------------------------------------
# JSONL Writing
# ----------------------------------------------------------------------

def append_jsonl(
    path: str | Path,
    record: Dict[str, Any],
) -> None:
    """
    Append a single JSON object as one line to a JSONL file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


def write_jsonl(
    path: str | Path,
    records: Iterable[Dict[str, Any]],
) -> None:
    """
    Write an iterable of JSON objects to a JSONL file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


# ----------------------------------------------------------------------
# Structured experiment helpers
# ----------------------------------------------------------------------

def write_result(
    output_path: str | Path,
    *,
    input_record: Dict[str, Any],
    explanation: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
    sample_id: Optional[str] = None,
    stage: str = "explanation",
) -> None:
    """
    Write a single structured experiment result entry.

    Results are intentionally verbose to support:
    - Post-hoc analysis
    - Error tracing
    - Reproducibility audits
    """
    record = {
        "stage": stage,
        "sample_id": sample_id or input_record.get("sample_id"),
        "input": input_record,
        "explanation": explanation,
    }

    if meta is not None:
        record["run_meta"] = meta

    append_jsonl(output_path, record)


def write_error(
    path: str | Path,
    *,
    input_record: Dict[str, Any],
    error: Exception,
    stage: str,
    sample_id: Optional[str] = None,
    run_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write a structured error entry without stopping the experiment.

    Errors are treated as first-class experimental outcomes.
    """
    error_record = {
        "stage": stage,
        "sample_id": sample_id or input_record.get("sample_id"),
        "input": input_record,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }

    if run_meta is not None:
        error_record["run_meta"] = run_meta

    append_jsonl(path, error_record)
