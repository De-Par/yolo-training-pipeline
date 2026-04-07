from __future__ import annotations

import shutil
import yaml

from pathlib import Path
from typing import Dict, List, Sequence

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DATASET_SPLIT_ORDER = ("train", "val", "test")


def safe_dataset_key(name: str) -> str:
    out = []
    for ch in name.strip():
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    key = "".join(out).strip("_")
    return key or "dataset"


def load_class_names(classes_txt: Path) -> List[str]:
    return [line.strip() for line in classes_txt.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_classes_txt(output_dir: Path, class_names: List[str]) -> Path:
    path = output_dir / "classes.txt"
    path.write_text("\n".join(class_names) + "\n", encoding="utf-8")
    return path


def write_dataset_yaml(
    output_dir: Path,
    class_names: List[str],
    dataset_name: str,
    *,
    split_names: Sequence[str] | None = None,
) -> Path:
    yaml_path = output_dir / f"{dataset_name}.yaml"
    active_splits = list(split_names or ("train", "val"))
    cfg = {
        "path": str(output_dir.resolve()),
        "names": {i: name for i, name in enumerate(class_names)},
        "nc": len(class_names),
    }
    for split in active_splits:
        cfg[split] = f"images/{split}"
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
    return yaml_path


def remove_tree(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    shutil.rmtree(path, ignore_errors=True)


def clean_output_dir(output_dir: Path, dataset_key: str) -> None:
    remove_tree(output_dir / "images")
    remove_tree(output_dir / "labels")
    remove_tree(output_dir / "labels.cache")
    remove_tree(output_dir / "images.cache")
    (output_dir / "classes.txt").unlink(missing_ok=True)
    (output_dir / "dataset_stats.json").unlink(missing_ok=True)
    for path in output_dir.glob("dataset_stats*.png"):
        path.unlink(missing_ok=True)
    (output_dir / "prepare_report.json").unlink(missing_ok=True)
    (output_dir / f"{dataset_key}.yaml").unlink(missing_ok=True)
    for path in output_dir.glob("**/*.cache"):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def iter_image_files(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        return []
    return [path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def count_images(images_dir: Path) -> int:
    return len(iter_image_files(images_dir))


def build_image_stem_map(images_dir: Path) -> Dict[str, List[Path]]:
    stems: Dict[str, List[Path]] = {}
    for path in iter_image_files(images_dir):
        stems.setdefault(path.stem, []).append(path)
    return stems


def detect_dataset_splits(dataset_dir: Path) -> List[str]:
    splits: List[str] = []
    for split in DATASET_SPLIT_ORDER:
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split
        if images_dir.exists() or labels_dir.exists():
            splits.append(split)
    return splits
