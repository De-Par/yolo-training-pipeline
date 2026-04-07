from __future__ import annotations

import csv
import json
import math

from pathlib import Path
from typing import Any, Dict, List, Optional
from core.common import PipelineError, ProgressCallback, ensure_local_mplconfigdir


def _to_float(value: Any) -> float:
    try:
        return float(value)
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


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _get_any(data: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in data:
            return data.get(key)
    return None


def export_per_class_ap(
    model_path: Path,
    data_path: Path,
    split: str = "val",
    imgsz: int = 640,
    batch: int = 16,
    device: Optional[str] = None,
    workers: int = 8,
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> Dict[str, Any]:
    if not model_path.exists():
        raise PipelineError(
            f"Model checkpoint does not exist: {model_path}",
            hint="Train a model first or point --model to an existing .pt checkpoint.",
        )
    if not data_path.exists():
        raise PipelineError(
            f"Dataset YAML does not exist: {data_path}",
            hint="Point --data to a YOLO dataset YAML produced by conversion or preparation.",
        )

    ensure_local_mplconfigdir()
    from ultralytics import YOLO

    if progress_callback is not None:
        progress_callback("report:export:init", 0, 4, "report:export: load model")

    model = YOLO(str(model_path))
    if progress_callback is not None:
        progress_callback("report:export", 1, 4, "report:export: validate")

    val_kwargs: Dict[str, Any] = {
        "data": str(data_path),
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": workers,
        "plots": False,
        "verbose": verbose,
    }
    if conf is not None:
        val_kwargs["conf"] = conf
    if iou is not None:
        val_kwargs["iou"] = iou

    results = model.val(**val_kwargs)
    summary_rows = results.summary()
    if progress_callback is not None:
        progress_callback("report:export", 2, 4, "report:export: summarize")
    if not isinstance(summary_rows, list):
        raise PipelineError(
            "Unexpected results.summary() format from Ultralytics.",
            hint="Check the installed ultralytics version. The report exporter expects a list of dict rows.",
            details=f"results.summary() type: {type(summary_rows).__name__}",
        )

    names = getattr(results, "names", None) or {}
    if not isinstance(names, dict):
        names = {index: str(value) for index, value in enumerate(names)}

    row_by_id: Dict[int, Dict[str, Any]] = {}
    row_by_name: Dict[str, Dict[str, Any]] = {}
    for row in summary_rows:
        if not isinstance(row, dict):
            continue
        class_value = row.get("Class")
        if isinstance(class_value, int):
            row_by_id[class_value] = row
        else:
            row_by_name[str(class_value)] = row

    per_class: List[Dict[str, Any]] = []
    for class_id in sorted(names.keys(), key=int):
        class_index = int(class_id)
        class_name = str(names[class_id])
        row = row_by_id.get(class_index) or row_by_name.get(class_name) or row_by_name.get(str(class_index))
        if row is None:
            per_class.append(
                {
                    "class_id": class_index,
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

        try:
            instances = int(float(row.get("Instances", 0)))
        except Exception:
            instances = 0

        box_p = _get_any(row, ["Box-P", "P", "precision", "Precision"])
        box_r = _get_any(row, ["Box-R", "R", "recall", "Recall"])
        box_f1 = _get_any(row, ["Box-F1", "F1", "f1", "F1-score"])
        map50 = _to_float(_get_any(row, ["mAP50", "mAP50(B)", "AP50", "AP@0.5"]))
        map50_95 = _to_float(_get_any(row, ["mAP50-95", "mAP50-95(B)", "AP", "AP@0.5:0.95"]))
        if instances == 0 or not _is_finite_number(map50_95):
            map50 = None
            map50_95 = None

        per_class.append(
            {
                "class_id": class_index,
                "class_name": class_name,
                "instances": instances,
                "box_p": _to_float(box_p) if _is_finite_number(box_p) else None,
                "box_r": _to_float(box_r) if _is_finite_number(box_r) else None,
                "box_f1": _to_float(box_f1) if _is_finite_number(box_f1) else None,
                "map50": map50 if (map50 is None or _is_finite_number(map50)) else None,
                "map50_95": map50_95 if (map50_95 is None or _is_finite_number(map50_95)) else None,
            }
        )

    out_dir = (output_dir or (Path("runs") / "analysis")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_slug(f"{model_path.stem}_{split}_per_class_ap")
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_id", "class_name", "instances", "box_p", "box_r", "box_f1", "map50", "map50_95"],
        )
        writer.writeheader()
        writer.writerows(per_class)

    global_metrics = {
        key: _to_float(value) for key, value in (results.results_dict.items() if hasattr(results, "results_dict") else [])
    }
    payload = {
        "model": str(model_path.resolve()),
        "data": str(data_path.resolve()),
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": workers,
        "global_metrics": global_metrics,
        "speed": dict(results.speed) if hasattr(results, "speed") else {},
        "per_class": per_class,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if progress_callback is not None:
        progress_callback("report:export", 3, 4, "report:export: write outputs")

    valid_rows = [row for row in per_class if row["map50_95"] is not None]
    missing_rows = [row for row in per_class if row["map50_95"] is None]
    valid_sorted = sorted(valid_rows, key=lambda row: row["map50_95"])
    if progress_callback is not None:
        progress_callback("report:export", 4, 4, "report:export: done")

    return {
        "csv_path": csv_path,
        "json_path": json_path,
        "per_class": per_class,
        "valid_rows": valid_rows,
        "missing_rows": missing_rows,
        "valid_sorted": valid_sorted,
        "global_metrics": global_metrics,
    }
