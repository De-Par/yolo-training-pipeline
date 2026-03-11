#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.common import format_info, run_cli_with_progress
from core.onnx.common import build_onnx_artifact_name, ensure_dir, parse_hw, parse_imgsz
from core.onnx.exporter import ExportConfig
from core.onnx.optimizer import OptimizeConfig
from core.onnx.pipeline import PipelineConfig, run_export_and_optimize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a YOLO checkpoint to ONNX and optimize it for a target runtime.")
    parser.add_argument("--weights", type=str, required=True, help="Path to a YOLO .pt checkpoint.")
    parser.add_argument("--artifact-dir", type=str, required=True, help="Directory where exported and optimized ONNX artifacts will be written.")
    parser.add_argument("--imgsz", nargs="+", default=["768"], help="Export/calibration image size as one integer or two integers: H W.")
    parser.add_argument("--batch", type=int, default=1, help="Export batch size.")
    parser.add_argument("--export-device", type=str, default="cpu", help='Device used during Ultralytics ONNX export, for example "cpu" or "0".')
    parser.add_argument("--opset", type=int, default=None, help="Optional ONNX opset override.")
    parser.add_argument("--dynamic", action="store_true", help="Export dynamic axes instead of fixed input shapes.")
    parser.add_argument("--no-simplify", action="store_true", help="Disable ONNX graph simplification during export.")
    parser.add_argument("--target", type=str, choices=["cpu", "cuda"], default="cpu", help="Target runtime provider family.")
    parser.add_argument("--graph-level", type=str, choices=["basic", "extended", "all"], default="extended", help="ONNX Runtime graph optimization level.")
    parser.add_argument("--tag", type=str, default=None, help="Optional tag appended to generated artifact names.")
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

    def _run(progress_callback):
        artifact_dir = ensure_dir(Path(args.artifact_dir).expanduser().resolve())
        weights_path = Path(args.weights).expanduser().resolve()
        export_path = artifact_dir / build_onnx_artifact_name(weights_path.stem, stage="export", precision="fp32", tag=args.tag)
        export_cfg = ExportConfig(
            weights_path=weights_path,
            output_path=export_path,
            imgsz=parse_imgsz(args.imgsz),
            batch=args.batch,
            device=args.export_device,
            dynamic=args.dynamic,
            simplify=not args.no_simplify,
            opset=args.opset,
        )
        optimize_cfg = OptimizeConfig(
            input_model=export_path,
            output_dir=artifact_dir,
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
        )
        return run_export_and_optimize(
            PipelineConfig(export=export_cfg, optimize=optimize_cfg),
            progress_callback=progress_callback,
        )

    artifacts = run_cli_with_progress(desc="onnx pipeline", unit="stage", action=_run)
    for name, path in artifacts.items():
        print(format_info(f"{name}: {path}"))


if __name__ == "__main__":
    main()
