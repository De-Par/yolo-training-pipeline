from __future__ import annotations

import json
import random

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from core.common import PipelineError, ProgressCallback, format_info
from core.common.fs import clean_split_dirs, safe_link_or_copy
from core.datasets.common import clean_output_dir, safe_dataset_key, write_classes_txt, write_dataset_yaml
from core.datasets.pipeline_utils import clamp_fraction, require_existing, slow_path_warning

LogFn = Callable[[str], None]
SUPPORTED_BBOX_FORMATS = {"xyxy", "xywh"}


@dataclass(slots=True)
class ConvertDatasetOptions:
    dataset_name: str
    input_format: str
    output_root: Path = Path("data/converted")
    link_mode: str = "symlink"
    train_fraction: float = 1.0
    val_fraction: float = 1.0
    sample_seed: int = 42
    clean: bool = False
    train_images_dir: Optional[Path] = None
    train_annotations: Optional[Path] = None
    val_images_dir: Optional[Path] = None
    val_annotations: Optional[Path] = None
    class_names_file: Optional[Path] = None
    object_prefix: str = "item"
    category_id_key: str = "category_id"
    bbox_key: str = "bounding_box"
    bbox_format: str = "xyxy"
    image_width_key: str = "width"
    image_height_key: str = "height"


@dataclass(slots=True)
class ConversionContext:
    options: ConvertDatasetOptions
    output_dir: Path
    dataset_key: str
    train_images_dir: Path
    train_annotations: Path
    val_images_dir: Path
    val_annotations: Path
    log: LogFn
    progress_callback: ProgressCallback | None


ConversionAdapter = Callable[[ConversionContext], dict[str, Any]]


def _noop(_: str) -> None:
    return None


def supported_input_formats() -> list[str]:
    return sorted(CONVERSION_ADAPTERS.keys())


def _safe_rel_stem(path_like: str) -> str:
    return "__".join(Path(path_like).with_suffix("").parts)


def _build_basename_index(images_dir: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for path in images_dir.rglob("*"):
        if path.is_file():
            index.setdefault(path.name, []).append(path)
    return index


def _convert_xywh_bbox_to_yolo(bbox_xywh: Iterable[float], img_w: int, img_h: int) -> Tuple[float, float, float, float] | None:
    x, y, w, h = map(float, bbox_xywh)
    if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
        return None
    x1 = max(0.0, min(float(img_w), x))
    y1 = max(0.0, min(float(img_h), y))
    x2 = max(0.0, min(float(img_w), x + w))
    y2 = max(0.0, min(float(img_h), y + h))
    w = x2 - x1
    h = y2 - y1
    if w <= 1e-6 or h <= 1e-6:
        return None
    return ((x1 + x2) / 2.0 / float(img_w), (y1 + y2) / 2.0 / float(img_h), w / float(img_w), h / float(img_h))


def _convert_xyxy_bbox_to_yolo(bbox_xyxy: Iterable[float], img_w: int, img_h: int) -> Tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    if img_w <= 0 or img_h <= 0:
        return None
    x1 = max(0.0, min(float(img_w), x1))
    y1 = max(0.0, min(float(img_h), y1))
    x2 = max(0.0, min(float(img_w), x2))
    y2 = max(0.0, min(float(img_h), y2))
    w = x2 - x1
    h = y2 - y1
    if w <= 1e-6 or h <= 1e-6:
        return None
    return ((x1 + x2) / 2.0 / float(img_w), (y1 + y2) / 2.0 / float(img_h), w / float(img_w), h / float(img_h))


def _sample_sequence(items: Sequence[Any], fraction: float, seed: int) -> list[Any]:
    if fraction >= 1.0 or not items:
        return list(items)
    selected_count = max(1, int(round(len(items) * fraction)))
    randomizer = random.Random(seed)
    selected_indices = set(randomizer.sample(range(len(items)), selected_count))
    return [item for index, item in enumerate(items) if index in selected_indices]


def _load_class_names(path: Optional[Path]) -> list[str]:
    class_file = require_existing(path, "--class-names-file")
    class_names = [line.strip() for line in class_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not class_names:
        raise PipelineError(f"No class names found in {class_file}")
    return class_names


def _resolve_context(
    options: ConvertDatasetOptions,
    log: LogFn,
    progress_callback: ProgressCallback | None = None,
) -> ConversionContext:
    train_fraction = clamp_fraction(options.train_fraction, "--train-fraction")
    val_fraction = clamp_fraction(options.val_fraction, "--val-fraction")
    supported_formats = supported_input_formats()
    if options.input_format not in supported_formats:
        raise PipelineError(
            f"Unsupported --input-format '{options.input_format}'. Supported: {', '.join(supported_formats)}"
        )
    if options.link_mode not in {"symlink", "copy"}:
        raise PipelineError(f"Unsupported --link-mode '{options.link_mode}'")
    if options.bbox_format not in SUPPORTED_BBOX_FORMATS:
        raise PipelineError(
            f"Unsupported --bbox-format '{options.bbox_format}'. Supported: {', '.join(sorted(SUPPORTED_BBOX_FORMATS))}"
        )

    dataset_key = safe_dataset_key(options.dataset_name)
    train_images_dir = require_existing(options.train_images_dir, "--train-images-dir")
    val_images_dir = require_existing(options.val_images_dir, "--val-images-dir")
    train_annotations = require_existing(options.train_annotations, "--train-annotations")
    val_annotations = require_existing(options.val_annotations, "--val-annotations")

    output_dir = (options.output_root / dataset_key).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for message in slow_path_warning(options.output_root, "output-root"):
        log(message)
    log(
        format_info(
            f"Starting dataset conversion: dataset={dataset_key}, input_format={options.input_format}, "
            f"output_dir={output_dir}, train_fraction={train_fraction}, val_fraction={val_fraction}, "
            f"link_mode={options.link_mode}"
        )
    )
    if options.clean:
        log(format_info(f"Cleaning output directory: {output_dir}"))
        clean_output_dir(output_dir, dataset_key)

    for message in slow_path_warning(train_images_dir, "train-images-dir"):
        log(message)
    for message in slow_path_warning(val_images_dir, "val-images-dir"):
        log(message)

    return ConversionContext(
        options=options,
        output_dir=output_dir,
        dataset_key=dataset_key,
        train_images_dir=train_images_dir,
        train_annotations=train_annotations,
        val_images_dir=val_images_dir,
        val_annotations=val_annotations,
        log=log,
        progress_callback=progress_callback,
    )


def _convert_coco_detection_split(
    *,
    images_dir: Path,
    annotations_path: Path,
    output_dir: Path,
    split: str,
    link_mode: str,
    sample_fraction: float,
    sample_seed: int,
    class_names: Optional[list[str]] = None,
    class_id_map: Optional[dict[int, int]] = None,
    write_classes: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int | str]:
    with annotations_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    categories: List[dict] = sorted(data["categories"], key=lambda category: int(category["id"]))
    if class_id_map is None:
        class_id_map = {int(category["id"]): index for index, category in enumerate(categories)}
    if class_names is None:
        class_names = [str(category.get("name", category["id"])) for category in categories]

    all_images = {int(image["id"]): image for image in data["images"]}
    sampled_image_ids = _sample_sequence(sorted(all_images.keys()), sample_fraction, sample_seed)
    images = {image_id: all_images[image_id] for image_id in sampled_image_ids}

    annotations_by_image: Dict[int, List[dict]] = {}
    for annotation in data["annotations"]:
        if "bbox" not in annotation or annotation.get("iscrowd", 0) == 1:
            continue
        image_id = int(annotation["image_id"])
        if image_id not in images:
            continue
        annotations_by_image.setdefault(image_id, []).append(annotation)

    images_out = output_dir / "images" / split
    labels_out = output_dir / "labels" / split
    clean_split_dirs(images_out, labels_out)
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    written_images = 0
    written_labels = 0
    missing_images = 0
    ambiguous_images = 0
    missing_records: List[dict] = []
    ambiguous_records: List[dict] = []
    basename_index = _build_basename_index(images_dir)

    total_images = len(images)
    if progress_callback is not None:
        progress_callback(f"convert:items:{split}:init", 0, total_images, f"convert:items:{split}")

    for index, (image_id, image_info) in enumerate(images.items(), start=1):
        file_name = image_info["file_name"]
        img_w = int(image_info["width"])
        img_h = int(image_info["height"])

        src_img = images_dir / file_name
        if not src_img.exists():
            candidates = basename_index.get(Path(file_name).name, [])
            if len(candidates) == 1:
                src_img = candidates[0]
            elif len(candidates) > 1:
                ambiguous_images += 1
                ambiguous_records.append(
                    {"image_id": image_id, "file_name": file_name, "candidates": [str(path) for path in candidates]}
                )
                if progress_callback is not None:
                    progress_callback(f"convert:items:{split}", index, total_images, f"convert:items:{split}: {file_name}")
                continue
        if not src_img.exists():
            missing_images += 1
            missing_records.append({"image_id": image_id, "file_name": file_name})
            if progress_callback is not None:
                progress_callback(f"convert:items:{split}", index, total_images, f"convert:items:{split}: {file_name}")
            continue

        safe_stem = _safe_rel_stem(file_name)
        ext = Path(file_name).suffix or src_img.suffix
        safe_link_or_copy(src_img, images_out / f"{safe_stem}{ext}", link_mode)
        written_images += 1

        lines: List[str] = []
        for annotation in annotations_by_image.get(image_id, []):
            mapped_class = class_id_map.get(int(annotation["category_id"]))
            if mapped_class is None:
                continue
            converted = _convert_xywh_bbox_to_yolo(annotation["bbox"], img_w, img_h)
            if converted is None:
                continue
            xc, yc, wn, hn = converted
            lines.append(f"{mapped_class} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        (labels_out / f"{safe_stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        written_labels += 1
        if progress_callback is not None:
            progress_callback(f"convert:items:{split}", index, total_images, f"convert:items:{split}: {safe_stem}")

    if write_classes:
        write_classes_txt(output_dir, class_names)

    report_path = output_dir / f"conversion_report_{split}.json"
    report_path.write_text(
        json.dumps(
            {
                "split": split,
                "input_format": "coco-detection",
                "num_classes": len(class_names),
                "total_images_in_annotations": len(all_images),
                "sample_fraction": sample_fraction,
                "sample_seed": sample_seed,
                "sampled_images": len(images),
                "written_images": written_images,
                "written_labels": written_labels,
                "missing_images": missing_images,
                "ambiguous_images": ambiguous_images,
                "missing_records": missing_records,
                "ambiguous_records": ambiguous_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "num_classes": len(class_names),
        "written_images": written_images,
        "written_labels": written_labels,
        "missing_images": missing_images,
        "ambiguous_images": ambiguous_images,
        "report_path": str(report_path),
    }


def _find_image_for_stem(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".webp", ".bmp"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _convert_per_image_json_split(
    *,
    images_dir: Path,
    annotations_dir: Path,
    output_dir: Path,
    split: str,
    link_mode: str,
    sample_fraction: float,
    sample_seed: int,
    class_names: list[str],
    object_prefix: str,
    category_id_key: str,
    bbox_key: str,
    bbox_format: str,
    image_width_key: str,
    image_height_key: str,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int | str]:
    annotation_files = _sample_sequence(sorted(annotations_dir.glob("*.json")), sample_fraction, sample_seed)
    images_out = output_dir / "images" / split
    labels_out = output_dir / "labels" / split
    clean_split_dirs(images_out, labels_out)
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    written_images = 0
    written_labels = 0
    missing_images = 0
    invalid_annotations = 0

    bbox_converter = _convert_xyxy_bbox_to_yolo if bbox_format == "xyxy" else _convert_xywh_bbox_to_yolo

    total_annotations = len(annotation_files)
    if progress_callback is not None:
        progress_callback(f"convert:items:{split}:init", 0, total_annotations, f"convert:items:{split}")

    for index, annotation_path in enumerate(annotation_files, start=1):
        stem = annotation_path.stem
        image_path = _find_image_for_stem(images_dir, stem)
        if image_path is None:
            missing_images += 1
            if progress_callback is not None:
                progress_callback(f"convert:items:{split}", index, total_annotations, f"convert:items:{split}: {stem}")
            continue

        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        img_w = int(annotation.get(image_width_key, 0))
        img_h = int(annotation.get(image_height_key, 0))
        if img_w <= 0 or img_h <= 0:
            invalid_annotations += 1
            if progress_callback is not None:
                progress_callback(f"convert:items:{split}", index, total_annotations, f"convert:items:{split}: {stem}")
            continue

        lines: List[str] = []
        for key, item in annotation.items():
            if not key.startswith(object_prefix) or not isinstance(item, Mapping):
                continue
            try:
                category_id = int(item.get(category_id_key, -1))
            except Exception:
                continue
            if category_id < 1 or category_id > len(class_names):
                continue
            bbox = item.get(bbox_key)
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            converted = bbox_converter(bbox, img_w, img_h)
            if converted is None:
                continue
            xc, yc, wn, hn = converted
            lines.append(f"{category_id - 1} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        safe_link_or_copy(image_path, images_out / image_path.name, link_mode)
        written_images += 1
        (labels_out / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        written_labels += 1
        if progress_callback is not None:
            progress_callback(f"convert:items:{split}", index, total_annotations, f"convert:items:{split}: {stem}")

    write_classes_txt(output_dir, class_names)
    report_path = output_dir / f"conversion_report_{split}.json"
    report_path.write_text(
        json.dumps(
            {
                "split": split,
                "input_format": "per-image-json",
                "num_classes": len(class_names),
                "total_annotation_files": len(sorted(annotations_dir.glob('*.json'))),
                "sample_fraction": sample_fraction,
                "sample_seed": sample_seed,
                "sampled_annotations": len(annotation_files),
                "written_images": written_images,
                "written_labels": written_labels,
                "missing_images": missing_images,
                "invalid_annotations": invalid_annotations,
                "bbox_format": bbox_format,
                "object_prefix": object_prefix,
                "category_id_key": category_id_key,
                "bbox_key": bbox_key,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "num_classes": len(class_names),
        "written_images": written_images,
        "written_labels": written_labels,
        "missing_images": missing_images,
        "invalid_annotations": invalid_annotations,
        "report_path": str(report_path),
    }


def _run_coco_detection_adapter(context: ConversionContext) -> dict[str, Any]:
    options = context.options
    with context.train_annotations.open("r", encoding="utf-8") as handle:
        train_data = json.load(handle)
    with context.val_annotations.open("r", encoding="utf-8") as handle:
        val_data = json.load(handle)

    train_categories = sorted(train_data["categories"], key=lambda category: int(category["id"]))
    val_categories = sorted(val_data["categories"], key=lambda category: int(category["id"]))
    train_signature = [(int(category["id"]), str(category.get("name", category["id"]))) for category in train_categories]
    val_signature = [(int(category["id"]), str(category.get("name", category["id"]))) for category in val_categories]
    if train_signature != val_signature:
        raise PipelineError(
            "Train and val category definitions differ. Use identical category ids/names across splits before conversion."
        )

    class_names = [str(category.get("name", category["id"])) for category in train_categories]
    class_id_map = {int(category["id"]): index for index, category in enumerate(train_categories)}
    train_stats = _convert_coco_detection_split(
        images_dir=context.train_images_dir,
        annotations_path=context.train_annotations,
        output_dir=context.output_dir,
        split="train",
        link_mode=options.link_mode,
        sample_fraction=options.train_fraction,
        sample_seed=options.sample_seed,
        class_names=class_names,
        class_id_map=class_id_map,
        write_classes=False,
        progress_callback=context.progress_callback,
    )
    val_stats = _convert_coco_detection_split(
        images_dir=context.val_images_dir,
        annotations_path=context.val_annotations,
        output_dir=context.output_dir,
        split="val",
        link_mode=options.link_mode,
        sample_fraction=options.val_fraction,
        sample_seed=options.sample_seed,
        class_names=class_names,
        class_id_map=class_id_map,
        write_classes=False,
        progress_callback=context.progress_callback,
    )
    write_classes_txt(context.output_dir, class_names)
    return {"class_names": class_names, "train_stats": train_stats, "val_stats": val_stats}


def _run_per_image_json_adapter(context: ConversionContext) -> dict[str, Any]:
    options = context.options
    class_names = _load_class_names(options.class_names_file)
    train_stats = _convert_per_image_json_split(
        images_dir=context.train_images_dir,
        annotations_dir=context.train_annotations,
        output_dir=context.output_dir,
        split="train",
        link_mode=options.link_mode,
        sample_fraction=options.train_fraction,
        sample_seed=options.sample_seed,
        class_names=class_names,
        object_prefix=options.object_prefix,
        category_id_key=options.category_id_key,
        bbox_key=options.bbox_key,
        bbox_format=options.bbox_format,
        image_width_key=options.image_width_key,
        image_height_key=options.image_height_key,
        progress_callback=context.progress_callback,
    )
    val_stats = _convert_per_image_json_split(
        images_dir=context.val_images_dir,
        annotations_dir=context.val_annotations,
        output_dir=context.output_dir,
        split="val",
        link_mode=options.link_mode,
        sample_fraction=options.val_fraction,
        sample_seed=options.sample_seed,
        class_names=class_names,
        object_prefix=options.object_prefix,
        category_id_key=options.category_id_key,
        bbox_key=options.bbox_key,
        bbox_format=options.bbox_format,
        image_width_key=options.image_width_key,
        image_height_key=options.image_height_key,
        progress_callback=context.progress_callback,
    )
    return {"class_names": class_names, "train_stats": train_stats, "val_stats": val_stats}


CONVERSION_ADAPTERS: dict[str, ConversionAdapter] = {
    "coco-detection": _run_coco_detection_adapter,
    "per-image-json": _run_per_image_json_adapter,
}


def convert_dataset_to_yolo(
    options: ConvertDatasetOptions,
    logger: Optional[LogFn] = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    log = logger or _noop
    context = _resolve_context(options, log, progress_callback=progress_callback)
    adapter = CONVERSION_ADAPTERS[options.input_format]
    result = adapter(context)
    class_names = list(result["class_names"])
    train_stats = dict(result["train_stats"])
    val_stats = dict(result["val_stats"])

    data_yaml = write_dataset_yaml(context.output_dir, class_names, context.dataset_key)

    log(
        format_info(
            f"Converted dataset: dataset={context.dataset_key}, classes={len(class_names)}, "
            f"train_images={train_stats.get('written_images', 0)}, val_images={val_stats.get('written_images', 0)}"
        )
    )
    log(format_info(f"Wrote train conversion report: {train_stats.get('report_path')}"))
    log(format_info(f"Wrote val conversion report: {val_stats.get('report_path')}"))
    log(format_info(f"Wrote classes.txt: {context.output_dir / 'classes.txt'}"))
    log(format_info(f"Wrote data.yaml: {data_yaml}"))
    log(format_info("Completed"))

    return {
        "dataset": context.dataset_key,
        "output_dir": context.output_dir,
        "data_yaml": data_yaml,
        "class_names": class_names,
        "input_format": options.input_format,
        "train_stats": train_stats,
        "val_stats": val_stats,
        "supported_input_formats": supported_input_formats(),
    }
