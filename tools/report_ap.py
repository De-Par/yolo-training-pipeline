#!/usr/bin/env python3
from __future__ import annotations

import argparse

from pathlib import Path

from _runtime import bootstrap_project_root

bootstrap_project_root(__file__, levels=1)

from core.common import format_info, run_cli_with_progress
from core.training.report_ap import export_per_class_ap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a YOLO checkpoint and export per-class AP metrics."
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to .pt model checkpoint.")
    parser.add_argument("--data", type=Path, required=True, help="Path to dataset data.yaml.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to validate on before exporting per-class AP.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size passed to Ultralytics.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--device", type=str, default=None, help='Validation device override, for example "cpu", "0", or "mps".')
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers used during validation.")
    parser.add_argument("--conf", type=float, default=None, help="Optional confidence threshold override for validation.")
    parser.add_argument("--iou", type=float, default=None, help="Optional IoU threshold override for validation.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory where CSV and JSON reports will be written.")
    parser.add_argument("--top-k", type=int, default=12, help="How many best and worst classes to print in the terminal summary.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose Ultralytics validation output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_cli_with_progress(
        desc="report ap",
        unit="stage",
        action=lambda progress_callback: export_per_class_ap(
            model_path=args.model,
            data_path=args.data,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            conf=args.conf,
            iou=args.iou,
            output_dir=args.output_dir,
            verbose=args.verbose,
            progress_callback=progress_callback,
        ),
    )

    print(format_info(f"Wrote CSV report: {report['csv_path']}"))
    print(format_info(f"Wrote JSON report: {report['json_path']}"))
    print(format_info(f"Classes with metrics: {len(report['valid_rows'])} / {len(report['per_class'])}"))
    print(format_info(f"Classes without validation instances: {len(report['missing_rows'])}"))

    top_k = max(1, int(args.top_k))
    print(f"\n{format_info('Worst classes by mAP50-95')}:")
    for row in report["valid_sorted"][:top_k]:
        print(
            f"  {row['class_id']:>2d} | {row['class_name']:<34} "
            f"inst={row['instances']:<4d} AP50={row['map50']:.4f} AP50-95={row['map50_95']:.4f}"
        )

    print(f"\n{format_info('Best classes by mAP50-95')}:")
    for row in report["valid_sorted"][-top_k:][::-1]:
        print(
            f"  {row['class_id']:>2d} | {row['class_name']:<34} "
            f"inst={row['instances']:<4d} AP50={row['map50']:.4f} AP50-95={row['map50_95']:.4f}"
        )


if __name__ == "__main__":
    main()
