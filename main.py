"""Command-line entry point for the Ghana Indigenous Intel rainfall pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline import run_pipeline
from src.utils import configure_logging

LOGGER = configure_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train models and generate rainfall predictions using indigenous indicators.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metrics.json",
        help="File name to store pipeline metrics in JSON format (saved inside models/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOGGER.info("Starting Ghana Indigenous Intel pipeline run...")
    metrics = run_pipeline()

    output_path = Path("models") / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    LOGGER.info("Pipeline complete. Metrics saved to %s", output_path)


if __name__ == "__main__":
    main()
