#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.common import run_cli_with_progress, stdout_logger
from core.datasets.prepare_yolo_dataset import PrepareYoloDatasetOptions, prepare_yolo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply an in-place preparation recipe to a YOLO-styled dataset.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to the YOLO-styled dataset directory to modify.")
    parser.add_argument("--recipe", type=Path, required=True, help="YAML file describing sampling, class drops, and class remaps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = PrepareYoloDatasetOptions(dataset_dir=args.dataset_dir, recipe_path=args.recipe)
    run_cli_with_progress(
        desc="prepare dataset",
        unit="item",
        action=lambda progress_callback: prepare_yolo_dataset(
            options,
            logger=stdout_logger,
            progress_callback=progress_callback,
        ),
    )


if __name__ == "__main__":
    main()
