#!/usr/bin/env python3
from __future__ import annotations

import argparse

from pathlib import Path
from typing import Any, Dict

from _runtime import bootstrap_project_root

bootstrap_project_root(__file__, levels=1)

from core.common import format_info, format_warning, run_cli_with_progress
from core.training.train import build_training_plan, load_cfg, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Ultralytics YOLO model on a YOLO dataset YAML.")
    parser.add_argument("--cfg", type=Path, help="YAML config with train parameters (optional)")
    parser.add_argument("--data", type=str, help="Path to YOLO dataset YAML (or via cfg: data)")
    parser.add_argument("--model", type=str, help="Path to YOLO checkpoint/config (or via cfg: model)")
    parser.add_argument("--epochs", type=int, help="Override the number of training epochs from the config.")
    parser.add_argument("--imgsz", type=int, help="Override the square training image size from the config.")
    parser.add_argument("--batch", type=int, help="Override the training batch size from the config.")
    parser.add_argument("--device", type=str, help='Examples: "cpu", "0", "cuda:0"')
    parser.add_argument("--name", type=str, help="Override the run name used under the training project directory.")
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "name": args.name,
    }
    return overrides


def main() -> None:
    args = parse_args()
    def _setup(progress_callback):
        progress_callback("train:setup:init", 0, 3, "train:setup: load config")
        cfg = load_cfg(args.cfg)
        progress_callback("train:setup", 1, 3, "train:setup: build plan")
        plan = build_training_plan(cfg, build_overrides(args))
        progress_callback("train:setup", 2, 3, "train:setup: ready")
        progress_callback("train:setup", 3, 3, "train:setup: done")
        return plan

    plan = run_cli_with_progress(
        desc="train setup",
        unit="stage",
        action=_setup,
    )

    for warning in plan["warnings"]:
        print(format_warning(warning), flush=True)

    print(format_info(f"Model: {plan['model']}"), flush=True)
    print(format_info(f"Dataset YAML: {plan['data']}"), flush=True)
    print(format_info(f"Device: {plan['device']}"), flush=True)
    for key, value in plan["env_info"].items():
        if key == "device":
            continue
        print(format_info(f"{key}: {value}"), flush=True)

    print(format_info("Training arguments:"), flush=True)
    for key in sorted(plan["train_kwargs"].keys()):
        print(f"  - {key}: {plan['train_kwargs'][key]}", flush=True)

    def _launch(progress_callback):
        progress_callback("train:launch:init", 0, 2, "train:launch: initialize model")
        progress_callback("train:launch", 1, 2, "train:launch: run training")
        run_training(plan["model"], plan["train_kwargs"])
        progress_callback("train:launch", 2, 2, "train:launch: done")

    run_cli_with_progress(
        desc="train launch",
        unit="stage",
        action=_launch,
    )


if __name__ == "__main__":
    main()
