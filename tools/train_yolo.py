#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULTS: Dict[str, Any] = {
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "device": "cpu",
    "workers": 8,
    "project": "runs/train",
    "name": "yolo-run",
    "seed": 42,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO (any supported generation) on a prepared data.yaml")
    parser.add_argument("--cfg", type=Path, help="Path to YAML config with train parameters")
    parser.add_argument("--data", type=str, help="Path to data.yaml")
    parser.add_argument("--model", type=str, help="Path to YOLO checkpoint/config")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--project", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def _load_cfg(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return cfg


def _pick(name: str, cli_value: Any, cfg: Dict[str, Any], *, required: bool = False) -> Any:
    if cli_value is not None:
        return cli_value
    if name in cfg:
        return cfg[name]
    if required:
        raise ValueError(f"Missing required parameter '{name}'. Provide via --{name} or --cfg.")
    return DEFAULTS[name]


def main() -> None:
    args = parse_args()
    cfg = _load_cfg(args.cfg)

    model_value = _pick("model", args.model, cfg, required=True)
    data_value = _pick("data", args.data, cfg, required=True)

    project_value = _pick("project", args.project, cfg)
    # Use absolute project path to avoid nested Ultralytics default task folders.
    project_value = str(Path(project_value).expanduser().resolve())

    train_kwargs = {
        "data": data_value,
        "epochs": _pick("epochs", args.epochs, cfg),
        "imgsz": _pick("imgsz", args.imgsz, cfg),
        "batch": _pick("batch", args.batch, cfg),
        "device": _pick("device", args.device, cfg),
        "workers": _pick("workers", args.workers, cfg),
        "project": project_value,
        "name": _pick("name", args.name, cfg),
        "seed": _pick("seed", args.seed, cfg),
    }

    from ultralytics import YOLO

    model = YOLO(model_value)
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
