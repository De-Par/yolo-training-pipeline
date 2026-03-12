from __future__ import annotations

import math
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from PIL import Image

from core.common import PipelineError

from .models import DatasetStats
from .utils import load_yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def resolve_dataset_root(data_cfg: Dict[str, Any], data_yaml_path: Path) -> Path:
    root = data_cfg.get("path")
    if root is None:
        return data_yaml_path.parent.resolve()
    root_path = Path(str(root))
    if not root_path.is_absolute():
        root_path = (data_yaml_path.parent / root_path).resolve()
    return root_path


def resolve_split_entry(data_cfg: Dict[str, Any], split: str, data_yaml_path: Path) -> Path:
    if split not in data_cfg:
        raise PipelineError(f"Split '{split}' not found in dataset YAML: {data_yaml_path}")
    root = resolve_dataset_root(data_cfg, data_yaml_path)
    entry = Path(str(data_cfg[split]))
    if entry.is_absolute():
        return entry
    return (root / entry).resolve()


def infer_images_dir_from_data_yaml(data_yaml: Path, split: str) -> Optional[Path]:
    cfg = load_yaml(data_yaml)
    split_path = resolve_split_entry(cfg, split, data_yaml)
    return split_path.resolve() if split_path.is_dir() else None


def list_images_from_split(data_yaml: Path, split: str) -> List[Path]:
    cfg = load_yaml(data_yaml)
    split_path = resolve_split_entry(cfg, split, data_yaml)

    images: List[Path] = []
    if split_path.is_dir():
        for path in sorted(split_path.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTS:
                images.append(path.absolute())
    elif split_path.is_file() and split_path.suffix.lower() == ".txt":
        for line in split_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            image_path = Path(line)
            if not image_path.is_absolute():
                image_path = (split_path.parent / image_path).absolute()
            images.append(image_path)
    else:
        raise PipelineError(f"Unsupported dataset split entry: {split_path}")

    if not images:
        raise PipelineError(f"No images found for split '{split}' in {split_path}")
    return images


def list_images_from_dir(images_dir: Path) -> List[Path]:
    images = [path.absolute() for path in sorted(images_dir.rglob("*")) if path.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise PipelineError(f"No images found in directory: {images_dir}")
    return images


def get_source_cfg(cfg: Dict[str, Any], source_name: str) -> Dict[str, Any]:
    return cfg["dataset"][source_name]


def get_source_split(source_cfg: Dict[str, Any], default: str) -> str:
    return str(source_cfg.get("split", default)).strip() or default


def default_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = parts.index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    except ValueError:
        return image_path.with_suffix(".txt")


def infer_labels_dir_from_data_yaml(data_yaml: Path, split: str) -> Optional[Path]:
    cfg = load_yaml(data_yaml)
    split_path = resolve_split_entry(cfg, split, data_yaml)
    if split_path.is_file():
        return None
    labels_dir = default_label_path(split_path / "placeholder.jpg").parent
    return labels_dir.resolve() if labels_dir.exists() else None


def resolve_label_path(image_path: Path, *, labels_dir: Optional[Path] = None, image_root: Optional[Path] = None) -> Path:
    if labels_dir is None:
        return default_label_path(image_path)
    if image_root is not None:
        try:
            relative = image_path.relative_to(image_root)
            return (labels_dir / relative).with_suffix('.txt')
        except ValueError:
            pass
    return (labels_dir / f"{image_path.stem}.txt").resolve()


def parse_label_counts(
    image_paths: Sequence[Path],
    labels_dir: Optional[Path] = None,
    image_root: Optional[Path] = None,
) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for image_path in image_paths:
        label_path = resolve_label_path(image_path, labels_dir=labels_dir, image_root=image_root)
        if not label_path.exists():
            continue
        try:
            text = label_path.read_text(encoding="utf-8")
        except Exception:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                class_id = int(float(line.split()[0]))
            except Exception:
                continue
            counts[class_id] = counts.get(class_id, 0) + 1
    return counts


def letterbox_shape(orig_h: int, orig_w: int, cfg: Dict[str, Any]) -> Tuple[int, int]:
    mode = str(cfg["imgsz"]["mode"]).lower()
    stride = int(cfg["imgsz"].get("stride", 32))
    if mode == "square":
        size = int(cfg["imgsz"]["value"])
        return size, size
    if mode == "rect":
        value = cfg["imgsz"]["value"]
        return int(value[0]), int(value[1])
    return int(math.ceil(orig_h / stride) * stride), int(math.ceil(orig_w / stride) * stride)


def preprocess_image_to_nchw(image_path: Path, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        orig_w, orig_h = image.size
        target_h, target_w = letterbox_shape(orig_h, orig_w, cfg)
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (target_w, target_h), color=(114, 114, 114))
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        canvas.paste(resized, (pad_x, pad_y))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return arr, (orig_h, orig_w), (target_h, target_w)


def sample_images(image_paths: Sequence[Path], cfg: Dict[str, Any]) -> List[Path]:
    max_images = int(cfg["benchmark"].get("max_images", 0) or 0)
    shuffle = bool(cfg["benchmark"].get("shuffle", False))
    seed = int(cfg["benchmark"].get("seed", 42))
    items = list(image_paths)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)
    if max_images > 0:
        items = items[:max_images]
    return items


def get_class_names_from_data_yaml(data_yaml: Path) -> List[str]:
    cfg = load_yaml(data_yaml)
    names = cfg.get("names", {})
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x) if str(x).isdigit() else str(x))]
    if isinstance(names, list):
        return [str(item) for item in names]
    return []


def infer_class_names_from_labels(labels_dir: Path) -> List[str]:
    max_class_id = -1
    for label_path in labels_dir.rglob('*.txt'):
        text = label_path.read_text(encoding='utf-8').strip()
        if not text:
            continue
        for line in text.splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                class_id = int(float(parts[0]))
            except Exception:
                continue
            max_class_id = max(max_class_id, class_id)
    return [str(i) for i in range(max_class_id + 1)] if max_class_id >= 0 else []


def resolve_class_names(cfg: Dict[str, Any], data_yaml: Optional[Path], labels_dir: Optional[Path]) -> List[str]:
    dataset_cfg = cfg["dataset"]
    if data_yaml is not None:
        names = get_class_names_from_data_yaml(data_yaml)
        if names:
            return names
    if dataset_cfg.get("class_names"):
        values = dataset_cfg["class_names"]
        if isinstance(values, list):
            return [str(item) for item in values]
        raise PipelineError("dataset.class_names must be a list when provided")
    if dataset_cfg.get("class_names_file"):
        path = Path(str(dataset_cfg["class_names_file"])).expanduser().resolve()
        if not path.exists():
            raise PipelineError(f"dataset.class_names_file does not exist: {path}")
        return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if labels_dir is not None:
        return infer_class_names_from_labels(labels_dir)
    return []


def build_quality_rows(class_names: Sequence[str], class_counts: Dict[int, int], per_class_maps: Sequence[float]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    total_classes = len(class_names)
    for index in range(total_classes):
        rows.append({
            'class_id': index,
            'class_name': str(class_names[index]),
            'instances': int(class_counts.get(index, 0)),
            'map50_95': float(per_class_maps[index]) if index < len(per_class_maps) else 0.0,
        })
    return rows


def collect_dataset_stats(
    image_paths: Sequence[Path],
    cfg: Dict[str, Any],
    labels_dir: Optional[Path],
    image_root: Optional[Path] = None,
) -> DatasetStats:
    orig_hs: List[int] = []
    orig_ws: List[int] = []
    eff_hs: List[int] = []
    eff_ws: List[int] = []
    for path in image_paths:
        with Image.open(path) as image:
            width, height = image.size
        eff_h, eff_w = letterbox_shape(height, width, cfg)
        orig_hs.append(height)
        orig_ws.append(width)
        eff_hs.append(eff_h)
        eff_ws.append(eff_w)
    return DatasetStats(
        image_paths=list(image_paths),
        avg_orig_h=float(np.mean(orig_hs)),
        avg_orig_w=float(np.mean(orig_ws)),
        avg_effective_h=float(np.mean(eff_hs)),
        avg_effective_w=float(np.mean(eff_ws)),
        class_counts=parse_label_counts(image_paths, labels_dir=labels_dir, image_root=image_root),
    )


def ensure_speed_source(cfg: Dict[str, Any]) -> tuple[List[Path], Optional[Path], Optional[Path], str]:
    source_cfg = get_source_cfg(cfg, 'speed')
    if source_cfg.get('data_yaml'):
        data_yaml = Path(str(source_cfg['data_yaml'])).expanduser().resolve()
        split = get_source_split(source_cfg, 'test')
        image_root = infer_images_dir_from_data_yaml(data_yaml, split)
        if source_cfg.get('annotations_dir'):
            labels_dir = Path(str(source_cfg['annotations_dir'])).expanduser().resolve()
        else:
            labels_dir = infer_labels_dir_from_data_yaml(data_yaml, split)
        return list_images_from_split(data_yaml, split), labels_dir, image_root, f"yaml:{data_yaml.name}:{split}"

    images_dir = Path(str(source_cfg['images_dir'])).expanduser().resolve()
    labels_dir = Path(str(source_cfg['annotations_dir'])).expanduser().resolve() if source_cfg.get('annotations_dir') else None
    return list_images_from_dir(images_dir), labels_dir, images_dir, f"dir:{images_dir}"


def ensure_quality_eval_dataset(cfg: Dict[str, Any]) -> tuple[Path, str, Path, Optional[Path], Optional[tempfile.TemporaryDirectory[str]], str]:
    source_cfg = get_source_cfg(cfg, 'quality')
    split = get_source_split(source_cfg, 'test')
    if source_cfg.get('data_yaml'):
        data_yaml = Path(str(source_cfg['data_yaml'])).expanduser().resolve()
        image_root = infer_images_dir_from_data_yaml(data_yaml, split)
        labels_dir = Path(str(source_cfg['annotations_dir'])).expanduser().resolve() if source_cfg.get('annotations_dir') else infer_labels_dir_from_data_yaml(data_yaml, split)
        if labels_dir is None:
            raise PipelineError(
                'Unable to resolve label directory for quality evaluation.',
                hint='Set dataset.quality.annotations_dir or use a standard YOLO dataset YAML with a labels/<split> directory.',
            )
        return data_yaml, split, labels_dir, image_root, None, f"yaml:{data_yaml.name}:{split}"

    images_dir = Path(str(source_cfg['images_dir'])).expanduser().resolve()
    labels_dir = Path(str(source_cfg['annotations_dir'])).expanduser().resolve()
    class_names = resolve_class_names(cfg, None, labels_dir)
    if not class_names:
        raise PipelineError(
            'Unable to resolve class names for benchmark quality evaluation.',
            hint='Set dataset.class_names, dataset.class_names_file, or point dataset.quality.data_yaml to an existing YOLO dataset YAML.',
        )

    tmp_dir = tempfile.TemporaryDirectory(prefix='yolo_benchmark_dataset_')
    root = Path(tmp_dir.name)
    (root / 'images').mkdir(parents=True, exist_ok=True)
    (root / 'labels').mkdir(parents=True, exist_ok=True)
    (root / 'images' / split).symlink_to(images_dir, target_is_directory=True)
    (root / 'labels' / split).symlink_to(labels_dir, target_is_directory=True)
    data_yaml = root / 'benchmark_dataset.yaml'
    payload = {
        'path': str(root),
        'train': f'images/{split}',
        'val': f'images/{split}',
        split: f'images/{split}',
        'names': {idx: name for idx, name in enumerate(class_names)},
    }
    data_yaml.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding='utf-8')
    return data_yaml, split, labels_dir, images_dir, tmp_dir, f"dir:{images_dir}"
