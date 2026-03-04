#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tqdm import tqdm

DEEPFASHION2_CLASSES = [
    "short_sleeve_top",
    "long_sleeve_top",
    "short_sleeve_outwear",
    "long_sleeve_outwear",
    "vest",
    "sling",
    "shorts",
    "trousers",
    "skirt",
    "short_sleeve_dress",
    "long_sleeve_dress",
    "vest_dress",
    "sling_dress",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DeepFashion2 original annotations to YOLO detection format."
    )
    parser.add_argument("--images-dir", type=Path, required=True, help="DeepFashion2 images directory")
    parser.add_argument("--annos-dir", type=Path, required=True, help="DeepFashion2 annotation JSON directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dataset root")
    parser.add_argument("--split", type=str, required=True, help="Split name: train/val")
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How images are placed into output/images/<split>",
    )
    return parser.parse_args()


def _safe_link_or_copy(src: Path, dst: Path, link_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link_mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _find_img_for_stem(images_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _convert_xyxy_to_yolo(
    bbox_xyxy: Iterable[float], img_w: int, img_h: int
) -> Tuple[float, float, float, float] | None:
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

    xc = (x1 + x2) / 2.0 / float(img_w)
    yc = (y1 + y2) / 2.0 / float(img_h)
    wn = w / float(img_w)
    hn = h / float(img_h)
    return xc, yc, wn, hn


def convert_deepfashion2_to_yolo(
    images_dir: Path,
    annos_dir: Path,
    output_dir: Path,
    split: str,
    link_mode: str = "symlink",
) -> Dict[str, int]:
    images_out = output_dir / "images" / split
    labels_out = output_dir / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(annos_dir.glob("*.json"))
    written_images = 0
    written_labels = 0
    missing_images = 0

    for ann_path in tqdm(ann_files, total=len(ann_files), desc=f"Converting {split}", unit="ann"):
        stem = ann_path.stem
        img_path = _find_img_for_stem(images_dir, stem)
        if img_path is None:
            missing_images += 1
            continue

        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        img_w = int(ann.get("width", 0))
        img_h = int(ann.get("height", 0))
        if img_w <= 0 or img_h <= 0:
            continue

        lines: List[str] = []
        for key, item in ann.items():
            if not key.startswith("item") or not isinstance(item, dict):
                continue
            cat_id = int(item.get("category_id", -1))
            if cat_id < 1 or cat_id > len(DEEPFASHION2_CLASSES):
                continue
            bbox = item.get("bounding_box")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            converted = _convert_xyxy_to_yolo(bbox, img_w, img_h)
            if converted is None:
                continue
            cls_idx = cat_id - 1
            xc, yc, wn, hn = converted
            lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        dst_img = images_out / img_path.name
        _safe_link_or_copy(img_path, dst_img, link_mode)
        written_images += 1

        (labels_out / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        written_labels += 1

    (output_dir / "classes.txt").write_text("\n".join(DEEPFASHION2_CLASSES), encoding="utf-8")

    return {
        "num_classes": len(DEEPFASHION2_CLASSES),
        "written_images": written_images,
        "written_labels": written_labels,
        "missing_images": missing_images,
        "ann_files": len(ann_files),
    }


def main() -> None:
    args = parse_args()
    stats = convert_deepfashion2_to_yolo(
        images_dir=args.images_dir,
        annos_dir=args.annos_dir,
        output_dir=args.output_dir,
        split=args.split,
        link_mode=args.link_mode,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
