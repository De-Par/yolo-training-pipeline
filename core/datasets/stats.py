from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np

from core.common import PipelineError, ProgressCallback, format_info
from core.datasets.common import count_images, iter_image_files, load_class_names


def _parse_label_line(line: str) -> tuple[int, float, float, float, float] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(float(parts[0]))
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
    except Exception:
        return None
    return class_id, x, y, w, h


def _list_label_files(labels_dir: Path) -> List[Path]:
    if not labels_dir.exists():
        return []
    return sorted(labels_dir.glob("*.txt"))


def compute_label_stats(
    labels_dir: Path,
    num_classes: int,
    label_files: List[Path] | None = None,
    progress_callback: ProgressCallback | None = None,
    progress_stage: str = "",
) -> Dict[str, Any]:
    instances = [0] * num_classes
    images_with_class = [0] * num_classes
    num_label_files = 0
    empty_label_files = 0
    total_boxes = 0
    invalid_lines = 0
    x_sum = 0.0
    y_sum = 0.0
    w_sum = 0.0
    h_sum = 0.0
    area_sum = 0.0
    area_bins = {"tiny": 0, "small": 0, "medium": 0, "large": 0}
    if not labels_dir.exists():
        return {
            "label_files": 0,
            "empty_label_files": 0,
            "instances": instances,
            "images_with_class": images_with_class,
            "total_boxes": 0,
            "invalid_lines": 0,
            "mean_x": None,
            "mean_y": None,
            "mean_width": None,
            "mean_height": None,
        "mean_area": None,
        "area_bins": area_bins,
    }

    label_paths = label_files if label_files is not None else _list_label_files(labels_dir)
    total_label_files = len(label_paths)
    for index, label_path in enumerate(label_paths, start=1):
        num_label_files += 1
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            empty_label_files += 1
            if progress_callback is not None:
                progress_callback(progress_stage, index, total_label_files, f"{progress_stage}: {label_path.name}")
            continue
        seen: Set[int] = set()
        for line in text.splitlines():
            parsed = _parse_label_line(line)
            if parsed is None:
                invalid_lines += 1
                continue
            class_id, x, y, w, h = parsed
            if not (0 <= class_id < num_classes):
                invalid_lines += 1
                continue
            area = w * h
            total_boxes += 1
            instances[class_id] += 1
            seen.add(class_id)
            x_sum += x
            y_sum += y
            w_sum += w
            h_sum += h
            area_sum += area
            if area < 0.001:
                area_bins["tiny"] += 1
            elif area < 0.01:
                area_bins["small"] += 1
            elif area < 0.1:
                area_bins["medium"] += 1
            else:
                area_bins["large"] += 1
        for class_id in seen:
            images_with_class[class_id] += 1
        if progress_callback is not None:
            progress_callback(progress_stage, index, total_label_files, f"{progress_stage}: {label_path.name}")

    divisor = total_boxes if total_boxes > 0 else None
    return {
        "label_files": num_label_files,
        "empty_label_files": empty_label_files,
        "instances": instances,
        "images_with_class": images_with_class,
        "total_boxes": total_boxes,
        "invalid_lines": invalid_lines,
        "mean_x": (x_sum / divisor) if divisor else None,
        "mean_y": (y_sum / divisor) if divisor else None,
        "mean_width": (w_sum / divisor) if divisor else None,
        "mean_height": (h_sum / divisor) if divisor else None,
        "mean_area": (area_sum / divisor) if divisor else None,
        "area_bins": area_bins,
    }


def build_class_stats_rows(class_names: List[str], train_stats: Dict[str, Any], val_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, name in enumerate(class_names):
        train_inst = train_stats["instances"][index]
        val_inst = val_stats["instances"][index]
        train_imgs = train_stats["images_with_class"][index]
        val_imgs = val_stats["images_with_class"][index]
        total_inst = train_inst + val_inst
        total_imgs = train_imgs + val_imgs
        if total_inst == 0 and total_imgs == 0:
            continue
        rows.append(
            {
                "id": index,
                "train_inst": train_inst,
                "val_inst": val_inst,
                "total_inst": total_inst,
                "train_imgs": train_imgs,
                "val_imgs": val_imgs,
                "total_imgs": total_imgs,
                "name": name,
            }
        )
    return rows


def format_class_stats_table(title: str, rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return f"{title}\nNo non-empty classes"

    headers = ["id", "train_inst", "val_inst", "total_inst", "train_imgs", "val_imgs", "total_imgs", "name"]
    widths = {
        "id": max(len(headers[0]), max(len(str(row["id"])) for row in rows)),
        "train_inst": max(len(headers[1]), max(len(str(row["train_inst"])) for row in rows)),
        "val_inst": max(len(headers[2]), max(len(str(row["val_inst"])) for row in rows)),
        "total_inst": max(len(headers[3]), max(len(str(row["total_inst"])) for row in rows)),
        "train_imgs": max(len(headers[4]), max(len(str(row["train_imgs"])) for row in rows)),
        "val_imgs": max(len(headers[5]), max(len(str(row["val_imgs"])) for row in rows)),
        "total_imgs": max(len(headers[6]), max(len(str(row["total_imgs"])) for row in rows)),
        "name": max(len(headers[7]), max(len(row["name"]) for row in rows)),
    }

    header_line = (
        f"{headers[0]:>{widths['id']}}  {headers[1]:>{widths['train_inst']}}  {headers[2]:>{widths['val_inst']}}  "
        f"{headers[3]:>{widths['total_inst']}}  {headers[4]:>{widths['train_imgs']}}  {headers[5]:>{widths['val_imgs']}}  "
        f"{headers[6]:>{widths['total_imgs']}}  {headers[7]:<{widths['name']}}"
    )
    separator = (
        f"{'-' * widths['id']}  {'-' * widths['train_inst']}  {'-' * widths['val_inst']}  "
        f"{'-' * widths['total_inst']}  {'-' * widths['train_imgs']}  {'-' * widths['val_imgs']}  "
        f"{'-' * widths['total_imgs']}  {'-' * widths['name']}"
    )
    lines = [title, header_line, separator]
    for row in rows:
        lines.append(
            f"{row['id']:>{widths['id']}}  {row['train_inst']:>{widths['train_inst']}}  {row['val_inst']:>{widths['val_inst']}}  "
            f"{row['total_inst']:>{widths['total_inst']}}  {row['train_imgs']:>{widths['train_imgs']}}  {row['val_imgs']:>{widths['val_imgs']}}  "
            f"{row['total_imgs']:>{widths['total_imgs']}}  {row['name']:<{widths['name']}}"
        )
    return "\n".join(lines)


def _summarize_split(
    dataset_dir: Path,
    split: str,
    num_classes: int,
    label_files: List[Path] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Dict[str, Any]:
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split
    image_count = count_images(images_dir)
    label_paths = label_files if label_files is not None else _list_label_files(labels_dir)
    label_stats = compute_label_stats(
        labels_dir,
        num_classes,
        label_files=label_paths,
        progress_callback=progress_callback,
        progress_stage=f"stats:collect:{split}",
    )
    label_stems = {path.stem for path in label_paths} if labels_dir.exists() else set()
    image_stems = {path.stem for path in iter_image_files(images_dir)} if images_dir.exists() else set()
    return {
        "split": split,
        "image_count": image_count,
        "label_file_count": label_stats["label_files"],
        "empty_label_files": label_stats["empty_label_files"],
        "missing_label_files": max(0, len(image_stems - label_stems)),
        "orphan_label_files": max(0, len(label_stems - image_stems)),
        "label_stats": label_stats,
    }


def _collect_plot_points(
    dataset_dir: Path,
    split: str,
    num_classes: int,
    label_files: List[Path] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Dict[str, np.ndarray]:
    labels_dir = dataset_dir / "labels" / split
    cls_values: List[int] = []
    box_values: List[List[float]] = []
    invalid_lines = 0
    if not labels_dir.exists():
        return {
            "cls": np.zeros((0,), dtype=np.int64),
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "invalid_lines": np.array([0], dtype=np.int64),
        }

    label_paths = label_files if label_files is not None else _list_label_files(labels_dir)
    total_label_files = len(label_paths)
    for index, label_path in enumerate(label_paths, start=1):
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            if progress_callback is not None:
                progress_callback(f"stats:plot:{split}", index, total_label_files, f"stats:plot:{split}: {label_path.name}")
            continue
        for line in text.splitlines():
            parsed = _parse_label_line(line)
            if parsed is None:
                invalid_lines += 1
                continue
            class_id, x, y, w, h = parsed
            if not (0 <= class_id < num_classes):
                invalid_lines += 1
                continue
            cls_values.append(class_id)
            box_values.append([x, y, w, h])
        if progress_callback is not None:
            progress_callback(f"stats:plot:{split}", index, total_label_files, f"stats:plot:{split}: {label_path.name}")

    return {
        "cls": np.asarray(cls_values, dtype=np.int64),
        "boxes": np.asarray(box_values, dtype=np.float32) if box_values else np.zeros((0, 4), dtype=np.float32),
        "invalid_lines": np.array([invalid_lines], dtype=np.int64),
    }


def collect_yolo_dataset_stats(dataset_dir: Path, progress_callback: ProgressCallback | None = None) -> Dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.exists():
        raise PipelineError(
            f"Dataset directory not found: {dataset_dir}",
            hint="Run yolo-convert-dataset first, then point --dataset-dir to the converted YOLO dataset directory.",
        )
    classes_path = dataset_dir / "classes.txt"
    if not classes_path.exists():
        raise PipelineError(
            f"classes.txt not found in dataset directory: {dataset_dir}",
            hint="The directory must be a converted/prepared YOLO dataset containing classes.txt and labels/.",
        )

    class_names = load_class_names(classes_path)
    if not class_names:
        raise PipelineError(
            f"No class names found in {classes_path}",
            hint="Regenerate the dataset with yolo-convert-dataset or inspect classes.txt for empty/invalid content.",
        )

    train_label_files = _list_label_files(dataset_dir / "labels" / "train")
    val_label_files = _list_label_files(dataset_dir / "labels" / "val")
    if progress_callback is not None:
        total_steps = (len(train_label_files) + len(val_label_files)) * 2
        progress_callback("stats:collect:init", 0, total_steps, "stats:collect")
    train = _summarize_split(
        dataset_dir,
        "train",
        len(class_names),
        label_files=train_label_files,
        progress_callback=progress_callback,
    )
    val = _summarize_split(
        dataset_dir,
        "val",
        len(class_names),
        label_files=val_label_files,
        progress_callback=progress_callback,
    )
    train_plot = _collect_plot_points(
        dataset_dir,
        "train",
        len(class_names),
        label_files=train_label_files,
        progress_callback=progress_callback,
    )
    val_plot = _collect_plot_points(
        dataset_dir,
        "val",
        len(class_names),
        label_files=val_label_files,
        progress_callback=progress_callback,
    )
    class_rows = build_class_stats_rows(class_names, train["label_stats"], val["label_stats"])
    class_rows_sorted = sorted(class_rows, key=lambda row: (-row["total_inst"], row["id"]))
    payload = {
        "dataset_dir": str(dataset_dir),
        "num_classes": len(class_names),
        "classes": [{"id": index, "name": name} for index, name in enumerate(class_names)],
        "train": train,
        "val": val,
        "class_rows": class_rows_sorted,
        "totals": {
            "images": train["image_count"] + val["image_count"],
            "label_files": train["label_file_count"] + val["label_file_count"],
            "empty_label_files": train["empty_label_files"] + val["empty_label_files"],
            "instances": train["label_stats"]["total_boxes"] + val["label_stats"]["total_boxes"],
        },
        "plot_data": {
            "train": train_plot,
            "val": val_plot,
        },
    }
    return payload


def render_dataset_summary(stats: Dict[str, Any]) -> str:
    train = stats["train"]
    val = stats["val"]
    lines = [
        format_info("Dataset summary"),
        f"dataset_dir:       {stats['dataset_dir']}",
        f"num_classes:       {stats['num_classes']}",
        f"train_images:      {train['image_count']}",
        f"val_images:        {val['image_count']}",
        f"train_label_files: {train['label_file_count']}",
        f"val_label_files:   {val['label_file_count']}",
        f"train_missing_labels: {train['missing_label_files']}",
        f"val_missing_labels:   {val['missing_label_files']}",
        f"train_orphan_labels:  {train['orphan_label_files']}",
        f"val_orphan_labels:    {val['orphan_label_files']}",
        f"empty_labels:      {stats['totals']['empty_label_files']}",
        f"instances_total:   {stats['totals']['instances']}",
        f"train_mean_area:   {train['label_stats']['mean_area']}",
        f"val_mean_area:     {val['label_stats']['mean_area']}",
        f"train_mean_xy:     ({train['label_stats']['mean_x']}, {train['label_stats']['mean_y']})",
        f"val_mean_xy:       ({val['label_stats']['mean_x']}, {val['label_stats']['mean_y']})",
        f"train_area_bins:   {train['label_stats']['area_bins']}",
        f"val_area_bins:     {val['label_stats']['area_bins']}",
    ]
    return "\n".join(lines)


def render_class_table(stats: Dict[str, Any]) -> str:
    return format_class_stats_table(format_info("Per-class instance table"), stats["class_rows"])


def write_dataset_stats_json(output_path: Path, stats: Dict[str, Any]) -> Path:
    payload = dict(stats)
    payload.pop("plot_data", None)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def _resolve_plot_output_paths(output_path: Path) -> Dict[str, Path]:
    suffix = output_path.suffix or ".png"
    stem = output_path.stem if output_path.suffix else output_path.name
    directory = output_path.parent
    return {
        "train": directory / f"{stem}_train{suffix}",
        "val": directory / f"{stem}_val{suffix}",
    }


def _write_split_dataset_plot(
    output_path: Path,
    stats: Dict[str, Any],
    split: str,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    import os

    mpl_cache_dir = (Path(".cache") / "matplotlib").resolve()
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from PIL import Image, ImageDraw

    try:
        from ultralytics.utils.plotting import colors as ultra_colors
    except Exception:
        ultra_colors = None

    plot_data = (stats.get("plot_data") or {}).get(split) or {}
    cls = np.asarray(plot_data.get("cls", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    boxes = np.asarray(plot_data.get("boxes", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32)
    names = {entry["id"]: entry["name"] for entry in stats["classes"]}
    nc = stats["num_classes"]
    split_stats = stats[split]
    class_counts = split_stats["label_stats"]["instances"]

    def pick_color(index: int) -> tuple[int, int, int]:
        if ultra_colors is not None:
            color = ultra_colors(int(index))
            return int(color[0]), int(color[1]), int(color[2])
        palette = [
            (4, 90, 141),
            (0, 158, 115),
            (230, 159, 0),
            (213, 94, 0),
            (204, 121, 167),
            (86, 180, 233),
        ]
        return palette[int(index) % len(palette)]

    subplot_cmap = LinearSegmentedColormap.from_list("white_blue", ["#ffffff", "#1f5bff"])
    non_zero_rows = [(class_id, count) for class_id, count in enumerate(class_counts) if count > 0]
    dynamic_height = max(12.0, min(22.0, 9.0 + 0.18 * max(1, len(non_zero_rows))))
    fig, axes = plt.subplots(2, 2, figsize=(18, dynamic_height), constrained_layout=True)
    fig.set_facecolor("white")
    ax0, ax1, ax2, ax3 = axes.flatten()
    fig.suptitle(f"YOLO Dataset Stats: {split}", fontsize=22, fontweight="bold")

    if non_zero_rows:
        ordered_rows = sorted(non_zero_rows, key=lambda item: (-item[1], item[0]))
        class_ids = [class_id for class_id, _ in ordered_rows]
        counts = [count for _, count in ordered_rows]
        labels = [names[class_id] for class_id in class_ids]
        y_pos = np.arange(len(labels))
        colors = [tuple(channel / 255.0 for channel in pick_color(class_id)) for class_id in class_ids]
        bars = ax0.barh(y_pos, counts, color=colors, edgecolor="none", alpha=0.95)
        ax0.set_yticks(y_pos)
        ax0.set_yticklabels(labels, fontsize=7)
        ax0.invert_yaxis()
        ax0.set_xlabel("instances")
        ax0.set_title(f"{split} class instances", fontsize=14, pad=12)
        x_padding = max(counts) * 0.015 if max(counts) > 0 else 1.0
        for bar, count in zip(bars, counts):
            ax0.text(
                bar.get_width() + x_padding,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                va="center",
                ha="left",
                fontsize=7,
            )
        ax0.set_xlim(0, max(counts) * 1.18 if max(counts) > 0 else 1.0)
        ax0.grid(axis="x", alpha=0.18, linewidth=0.8)
    else:
        ax0.text(0.5, 0.5, "No labels", ha="center", va="center", fontsize=14)
        ax0.set_axis_off()

    img = Image.fromarray(np.ones((1000, 1000, 3), dtype=np.uint8) * 255)
    if len(boxes):
        draw_boxes = np.column_stack([0.5 - boxes[:, 2:4] / 2, 0.5 + boxes[:, 2:4] / 2]) * 1000
        for class_id, box in zip(cls[:800], draw_boxes[:800]):
            ImageDraw.Draw(img).rectangle(box.tolist(), width=1, outline=pick_color(int(class_id)))
    ax1.imshow(img)
    ax1.set_title(f"{split} bbox shapes", fontsize=14, pad=12)
    ax1.axis("off")

    if len(boxes):
        ax2.hist2d(boxes[:, 0], boxes[:, 1], bins=60, range=[[0, 1], [0, 1]], cmap=subplot_cmap)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title(f"{split} center heatmap", fontsize=14, pad=12)

    if len(boxes):
        ax3.hist2d(boxes[:, 2], boxes[:, 3], bins=60, range=[[0, 1], [0, 1]], cmap=subplot_cmap)
    ax3.set_xlabel("width")
    ax3.set_ylabel("height")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title(f"{split} width/height heatmap", fontsize=14, pad=12)

    for axis in (ax0, ax1, ax2, ax3):
        axis.set_facecolor("white")
        if axis is not ax1:
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if progress_callback is not None:
        progress_callback(f"stats:render:{split}", 1, 1, f"stats:render:{split}")
    return output_path


def write_dataset_stats_plot(
    output_path: Path,
    stats: Dict[str, Any],
    progress_callback: ProgressCallback | None = None,
) -> Dict[str, Path]:
    output_paths = _resolve_plot_output_paths(output_path)
    if progress_callback is not None:
        progress_callback("stats:render:init", 0, len(output_paths), "stats:render")
    for split, split_output_path in output_paths.items():
        _write_split_dataset_plot(split_output_path, stats, split, progress_callback=progress_callback)
    return output_paths
