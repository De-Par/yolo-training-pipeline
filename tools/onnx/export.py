#!/usr/bin/env python3
from __future__ import annotations

from tools._runtime import bootstrap_project_root
bootstrap_project_root(__file__, levels=2)

import argparse

from pathlib import Path
from core.common import format_info, run_cli_with_progress
from core.onnx.common import parse_imgsz
from core.onnx.exporter import ExportConfig, export_yolo_to_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a YOLO .pt checkpoint to ONNX.")
    parser.add_argument("--weights", type=str, required=True, help="Path to a YOLO .pt checkpoint.")
    parser.add_argument("--imgsz", nargs="+", default=["768"], help="Export image size as one integer or two integers: H W.")
    parser.add_argument("--batch", type=int, default=1, help="Export batch size.")
    parser.add_argument("--device", type=str, default="cpu", help='Export device, for example "cpu" or "0".')
    parser.add_argument("--opset", type=int, default=None, help="Optional ONNX opset override.")
    parser.add_argument("--dynamic", action="store_true", help="Export dynamic axes instead of fixed input shapes.")
    parser.add_argument("--no-simplify", action="store_true", help="Disable ONNX graph simplification during export.")
    parser.add_argument("--output", type=str, default=None, help="Optional output path for the exported .onnx file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_path = run_cli_with_progress(
        desc="onnx export",
        unit="stage",
        action=lambda progress_callback: export_yolo_to_onnx(
            ExportConfig(
                weights_path=Path(args.weights),
                output_path=Path(args.output) if args.output else None,
                imgsz=parse_imgsz(args.imgsz),
                batch=args.batch,
                device=args.device,
                dynamic=args.dynamic,
                simplify=not args.no_simplify,
                opset=args.opset,
            ),
            progress_callback=progress_callback,
        ),
    )
    print(format_info(f"Exported ONNX: {output_path}"))


if __name__ == "__main__":
    main()
    