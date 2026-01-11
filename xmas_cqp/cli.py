"""
CLI entry point for XMAS-CQP.

Execution modes:
  - preprocess : raw model inputs -> decision IR
  - explain    : decision IR -> schema-constrained decision explanations
  - run        : full pipeline
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

# =========================
# Imports
# =========================
from xmas_cqp.agents.explainer_agent import ExplainerAgent
from xmas_cqp.agents.preprocessor_agent import PreprocessorAgent
from xmas_cqp.llm.openai_client import OpenAIClient
from xmas_cqp.llm.utils.io_utils import (
    read_jsonl,
    write_jsonl,
    write_error,
)
from xmas_cqp.llm.utils.logging_utils import create_logger
from xmas_cqp.llm.utils.config_utils import load_yaml_config


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def render_path(template: str, exp: Dict[str, Any]) -> str:
    """
    Render output paths using experiment variables.
    """
    return template.format(
        dataset=exp["dataset"],
        project_version=exp.get("project_version") or "default",
        run_id=exp["run_id"],
    )


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """
    Append ONE record to a jsonl file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        import json
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XMAS-CQP: Model-Centric Explainable Code Quality Prediction"
    )

    parser.add_argument(
        "command",
        choices=["preprocess", "explain", "run"],
        help="Execution mode",
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config YAML",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., openstack, qt, activemq)",
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file (decision input records)",
    )

    parser.add_argument(
        "--run_id",
        type=int,
        required=True,
        help="Repeated run id (e.g., 1..N)",
    )

    parser.add_argument(
        "--project_version",
        type=str,
        default=None,
        help="Project version (for multi-version datasets)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (debugging)",
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# API client
# ----------------------------------------------------------------------

def create_openai_client() -> OpenAIClient:
    return OpenAIClient.from_env()


# ----------------------------------------------------------------------
# Preprocessing stage
# ----------------------------------------------------------------------

def run_preprocessing(
    cfg: Dict[str, Any],
    logger,
    limit: int | None,
) -> None:
    logger.info("Starting preprocessing stage (decision IR construction)")

    pre_cfg = cfg["preprocessor"]
    out_path = Path(pre_cfg["output"]["path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    agent = PreprocessorAgent()

    samples: list[Dict[str, Any]] = []
    failed = 0

    for idx, record in enumerate(read_jsonl(pre_cfg["input"]["source"])):
        if limit is not None and idx >= limit:
            break

        try:
            input_block = record["input"]

            processed = agent.process(
                features=input_block["features"],
                model_output=input_block["model_output"],
                metadata=record.get("metadata", {}),
            )
            samples.append(processed)

        except Exception as e:
            failed += 1
            write_error(
                cfg["outputs"]["errors"]["path"],
                input_record=record,
                error=e,
                stage="preprocessing",
            )

    write_jsonl(out_path, samples)

    logger.info(
        f"Preprocessing completed. Success: {len(samples)}, Failed: {failed}"
    )


# ----------------------------------------------------------------------
# Explanation stage
# ----------------------------------------------------------------------

def run_explanation(
    cfg: Dict[str, Any],
    client: OpenAIClient,
    logger,
    limit: int | None,
) -> None:
    """
    Generate schema-constrained explanations.
    Write ONLY explanation objects (pure JSON schema).
    """
    logger.info("Starting explanation stage (model decision explanation)")

    exp_cfg = cfg["explainer"]

    agent = ExplainerAgent(
        client=client,
        schema_path=Path(exp_cfg["schema"]["path"]),
        system_prompt_path=Path(exp_cfg["prompts"]["system_prompt"]),
        user_prompt_path=Path(exp_cfg["prompts"]["user_prompt"]),
        model_name=exp_cfg["model"]["name"],
    )

    input_path = Path(cfg["outputs"]["processed"]["path"])
    explanations_path = Path(cfg["outputs"]["explanations"]["path"])
    explanations_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    failed = 0

    for idx, record in enumerate(read_jsonl(input_path)):
        if limit is not None and idx >= limit:
            break

        try:
            explanation = agent.explain(record=record)

            # Append ONE explanation per line
            append_jsonl(explanations_path, explanation)

            processed += 1

        except Exception as e:
            failed += 1
            write_error(
                cfg["outputs"]["errors"]["path"],
                input_record=record,
                error=e,
                stage="explanation",
            )

    logger.info(
        f"Explanation completed. Processed: {processed}, Failed: {failed}"
    )


# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)

    # ------------------------------------
    # Inject experiment parameters
    # ------------------------------------
    cfg["experiment"]["dataset"] = args.dataset
    cfg["experiment"]["run_id"] = args.run_id
    cfg["experiment"]["project_version"] = args.project_version

    # ------------------------------------
    # Render output paths (CRITICAL)
    # ------------------------------------
    exp = cfg["experiment"]
    for key in ["processed", "explanations", "errors"]:
        raw = cfg["outputs"][key]["path"]
        cfg["outputs"][key]["path"] = render_path(raw, exp)

    # Set preprocessing input
    cfg["preprocessor"]["input"]["source"] = str(args.input)

    logger = create_logger(
        name="xmas_cqp",
        log_dir=Path(cfg["logging"]["log_dir"]),
    )

    logger.info(
        f"XMAS-CQP command={args.command}, "
        f"dataset={args.dataset}, run_id={args.run_id}, "
        f"project_version={args.project_version}"
    )

    client = create_openai_client()

    if args.command == "preprocess":
        run_preprocessing(cfg, logger, args.limit)

    elif args.command == "explain":
        run_explanation(cfg, client, logger, args.limit)

    elif args.command == "run":
        if cfg["pipeline"]["enable_preprocessing"]:
            run_preprocessing(cfg, logger, args.limit)
        run_explanation(cfg, client, logger, args.limit)

    logger.info("XMAS-CQP execution finished")


if __name__ == "__main__":
    main()
