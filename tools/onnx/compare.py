#!/usr/bin/env python3
from __future__ import annotations

from tools._runtime import bootstrap_project_root
bootstrap_project_root(__file__, levels=2)

import argparse

from pathlib import Path
from pprint import pprint
from core.common import PipelineError
from core.onnx.infer import create_onnx_detector, inspect_raw_outputs


def _existing_file(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare raw output statistics between FP32 and INT8 ONNX models on the same image.",
    )
    parser.add_argument("--image",
        type=_existing_file,
        required=True,
        help="Path to a test image.",
    )
    parser.add_argument("--model-fp32",
        dest="model_fp32",
        type=_existing_file,
        required=True,
        help="Path to an FP32 ONNX model.",
    )
    parser.add_argument("--model-int8",
        dest="model_int8",
        type=_existing_file,
        required=True,
        help="Path to an INT8 ONNX model.",
    )
    parser.add_argument("--device",
        type=str,
        default="cpu",
        help='Inference device for ONNX Runtime, for example "cpu" or "cuda". Default: cpu.',
    )
    return parser.parse_args()


def _print_stats(title: str, model_path: Path, image_path: Path, device: str) -> None:
    print(title)
    print(f"model: {model_path}")
    detector = create_onnx_detector(model_path, device=device)
    stats = inspect_raw_outputs(detector, image_path)
    pprint(stats, sort_dicts=False)


def main() -> None:
    args = parse_args()

    try:
        _print_stats("FP32", args.model_fp32, args.image, args.device)
        print()
        _print_stats("INT8", args.model_int8, args.image, args.device)
    except PipelineError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    except Exception as exc:
        raise SystemExit(f"UNEXPECTED ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
    