#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from _runtime import bootstrap_project_root

bootstrap_project_root(__file__, levels=2)

from core.common import format_info, run_cli_with_progress
from core.onnx.common import parse_hw
from core.onnx.optimizer import OptimizeConfig, optimize_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize an ONNX model for a target runtime.")
    parser.add_argument("--input", type=str, required=True, help="Path to the source .onnx model.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory where optimized artifacts will be written.")
    parser.add_argument("--target", type=str, choices=["cpu", "cuda"], default="cpu", help="Target runtime provider family.")
    parser.add_argument("--graph-level", type=str, choices=["basic", "extended", "all"], default="extended", help="ONNX Runtime graph optimization level.")
    parser.add_argument("--tag", type=str, default=None, help="Optional tag appended to generated artifact names.")
    parser.add_argument("--imgsz", nargs="+", default=["768"], help="Calibration/export image size as one integer or two integers: H W.")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip ONNX Runtime quantization preprocessing.")
    parser.add_argument("--int8", action="store_true", help="Generate INT8 QDQ artifacts for CPU inference.")
    parser.add_argument("--fp16", action="store_true", help="Generate FP16 artifacts for CUDA inference.")
    parser.add_argument("--calib-dir", type=str, default=None, help="Calibration image directory required for INT8 quantization.")
    parser.add_argument("--calib-size", type=int, default=256, help="Maximum number of calibration images to use.")
    parser.add_argument("--calibration-method", type=str, choices=["minmax", "entropy", "percentile"], default="minmax", help="Calibration method for INT8 quantization.")
    parser.add_argument("--u8u8", action="store_true", help="Use QUInt8 activations and weights instead of QInt8.")
    parser.add_argument("--reduce-range", action="store_true", help="Enable reduced-range INT8 quantization.")
    parser.add_argument("--no-per-channel", action="store_true", help="Disable per-channel weight quantization.")
    parser.add_argument("--keep-io-types", action="store_true", help="Keep original model IO tensor types during FP16 conversion.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    artifacts = run_cli_with_progress(
        desc="onnx optimize",
        unit="stage",
        action=lambda progress_callback: optimize_onnx(
            OptimizeConfig(
                input_model=Path(args.input),
                output_dir=Path(args.output_dir),
                target=args.target,
                graph_level=args.graph_level,
                tag=args.tag,
                preprocess=not args.no_preprocess,
                int8=args.int8,
                fp16=args.fp16,
                calib_dir=Path(args.calib_dir) if args.calib_dir else None,
                calib_size=args.calib_size,
                input_hw=parse_hw(args.imgsz),
                calibration_method=args.calibration_method,
                per_channel=not args.no_per_channel,
                reduce_range=args.reduce_range,
                u8u8=args.u8u8,
                keep_io_types=args.keep_io_types,
            ),
            progress_callback=progress_callback,
        ),
    )
    for name, path in artifacts.items():
        print(format_info(f"{name}: {path}"))


if __name__ == "__main__":
    main()
