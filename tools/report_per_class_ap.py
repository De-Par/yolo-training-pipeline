#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json

from pathlib import Path
from typing import Any, Dict, List
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO validation and export per-class AP metrics (AP50/AP50-95) to CSV/JSON."
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to .pt model checkpoint.")
    parser.add_argument("--data", type=Path, required=True, help="Path to dataset data.yaml.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--device", type=str, default=None, help='Device, e.g. "cpu", "mps", "0", "cuda:0".')
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers for validation.")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold override.")
    parser.add_argument("--iou", type=float, default=None, help="IoU threshold override.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for generated reports.")
    parser.add_argument("--top-k", type=int, default=12, help="How many best/worst classes to print.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose val logs.")
    return parser.parse_args()


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _safe_slug(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    slug = "".join(keep).strip("_")
    return slug or "report"


def main() -> None:
    args = parse_args()

    model = YOLO(str(args.model))

    val_kwargs: Dict[str, Any] = {
        "data": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "plots": False,
        "verbose": args.verbose,
    }
    if args.conf is not None:
        val_kwargs["conf"] = args.conf
    if args.iou is not None:
        val_kwargs["iou"] = args.iou

    results = model.val(**val_kwargs)
    summary_rows = results.summary()  # list[dict]: includes Class, Instances, Box-P, Box-R, mAP50, mAP50-95

    names = getattr(results, "names", None) or {}
    if not isinstance(names, dict):
        names = {i: str(v) for i, v in enumerate(names)}

    row_by_name = {str(r["Class"]): r for r in summary_rows}
    per_class: List[Dict[str, Any]] = []
    for class_id in sorted(names.keys(), key=int):
        class_name = str(names[class_id])
        row = row_by_name.get(class_name)
        if row is None:
            per_class.append(
                {
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "instances": 0,
                    "box_p": None,
                    "box_r": None,
                    "box_f1": None,
                    "map50": None,
                    "map50_95": None,
                }
            )
            continue

        per_class.append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "instances": int(row.get("Instances", 0)),
                "box_p": _to_float(row.get("Box-P")),
                "box_r": _to_float(row.get("Box-R")),
                "box_f1": _to_float(row.get("Box-F1")),
                "map50": _to_float(row.get("mAP50")),
                "map50_95": _to_float(row.get("mAP50-95")),
            }
        )

    out_dir = args.output_dir or (Path("runs") / "analysis")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = _safe_slug(f"{args.model.stem}_{args.split}_per_class_ap")
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_id", "class_name", "instances", "box_p", "box_r", "box_f1", "map50", "map50_95"],
        )
        writer.writeheader()
        writer.writerows(per_class)

    global_metrics = {
        k: _to_float(v) for k, v in (results.results_dict.items() if hasattr(results, "results_dict") else [])
    }
    payload = {
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "global_metrics": global_metrics,
        "per_class": per_class,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    valid_rows = [r for r in per_class if r["map50_95"] is not None]
    missing_rows = [r for r in per_class if r["map50_95"] is None]
    valid_sorted = sorted(valid_rows, key=lambda r: r["map50_95"])
    k = max(1, int(args.top_k))

    print(f"[INFO] Saved CSV:  {csv_path}")
    print(f"[INFO] Saved JSON: {json_path}")
    print(f"[INFO] Classes with metrics: {len(valid_rows)} / {len(per_class)}")
    print(f"[INFO] Classes without val instances: {len(missing_rows)}")

    print("\n[INFO] Worst classes by mAP50-95:")
    for r in valid_sorted[:k]:
        print(
            f"  {r['class_id']:>2d} | {r['class_name']:<34} "
            f"inst={r['instances']:<4d} AP50={r['map50']:.4f} AP50-95={r['map50_95']:.4f}"
        )

    print("\n[INFO] Best classes by mAP50-95:")
    for r in valid_sorted[-k:][::-1]:
        print(
            f"  {r['class_id']:>2d} | {r['class_name']:<34} "
            f"inst={r['instances']:<4d} AP50={r['map50']:.4f} AP50-95={r['map50_95']:.4f}"
        )


if __name__ == "__main__":
    main()
