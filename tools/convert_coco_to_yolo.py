#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert COCO/Fashionpedia style annotations to YOLO detection format."
    )
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory with source images")
    parser.add_argument("--annotations", type=Path, required=True, help="COCO JSON path")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dataset root")
    parser.add_argument("--split", type=str, required=True, help="Split name, e.g. train/val")
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


def _convert_bbox_to_yolo(
    bbox_xywh: Iterable[float], img_w: int, img_h: int
) -> Tuple[float, float, float, float] | None:
    x, y, w, h = bbox_xywh

    if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
        return None

    x1 = max(0.0, min(float(img_w), float(x)))
    y1 = max(0.0, min(float(img_h), float(y)))
    x2 = max(0.0, min(float(img_w), float(x + w)))
    y2 = max(0.0, min(float(img_h), float(y + h)))

    w = x2 - x1
    h = y2 - y1
    if w <= 1e-6 or h <= 1e-6:
        return None

    xc = (x1 + x2) / 2.0 / float(img_w)
    yc = (y1 + y2) / 2.0 / float(img_h)
    wn = w / float(img_w)
    hn = h / float(img_h)
    return xc, yc, wn, hn


def _safe_rel_stem(path_like: str) -> str:
    p = Path(path_like)
    stem_parts = p.with_suffix("").parts
    # Preserve path uniqueness for nested datasets while remaining filename-safe.
    return "__".join(stem_parts)


def _build_basename_index(images_dir: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for p in images_dir.rglob("*"):
        if not p.is_file():
            continue
        index.setdefault(p.name, []).append(p)
    return index


def convert_coco_to_yolo(
    images_dir: Path,
    annotations_path: Path,
    output_dir: Path,
    split: str,
    link_mode: str = "symlink",
) -> Dict[str, int]:
    with annotations_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    categories: List[dict] = sorted(data["categories"], key=lambda c: int(c["id"]))
    cat_id_to_idx = {int(c["id"]): i for i, c in enumerate(categories)}

    images = {int(img["id"]): img for img in data["images"]}
    anns_by_image: Dict[int, List[dict]] = {}
    for ann in data["annotations"]:
        if "bbox" not in ann or ann.get("iscrowd", 0) == 1:
            continue
        img_id = int(ann["image_id"])
        anns_by_image.setdefault(img_id, []).append(ann)

    images_out = output_dir / "images" / split
    labels_out = output_dir / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    written_images = 0
    written_labels = 0
    missing_images = 0
    ambiguous_images = 0
    missing_records: List[dict] = []
    ambiguous_records: List[dict] = []
    basename_index = _build_basename_index(images_dir)

    for image_id, image_info in tqdm(
        images.items(),
        total=len(images),
        desc=f"Converting {split}",
        unit="img",
    ):
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
                    {
                        "image_id": image_id,
                        "file_name": file_name,
                        "candidates": [str(p) for p in candidates],
                    }
                )
                continue
        if not src_img.exists():
            missing_images += 1
            missing_records.append({"image_id": image_id, "file_name": file_name})
            continue

        safe_stem = _safe_rel_stem(file_name)
        ext = Path(file_name).suffix or src_img.suffix
        dst_img = images_out / f"{safe_stem}{ext}"
        _safe_link_or_copy(src_img, dst_img, link_mode)
        written_images += 1

        lines: List[str] = []
        for ann in anns_by_image.get(image_id, []):
            mapped_class = cat_id_to_idx.get(int(ann["category_id"]))
            if mapped_class is None:
                continue
            converted = _convert_bbox_to_yolo(ann["bbox"], img_w, img_h)
            if converted is None:
                continue
            xc, yc, wn, hn = converted
            lines.append(f"{mapped_class} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        label_name = f"{safe_stem}.txt"
        (labels_out / label_name).write_text("\n".join(lines), encoding="utf-8")
        written_labels += 1

    names = [c.get("name", str(c["id"])) for c in categories]
    (output_dir / "classes.txt").write_text("\n".join(names), encoding="utf-8")

    report_path = output_dir / f"conversion_report_{split}.json"
    report = {
        "split": split,
        "num_classes": len(names),
        "total_images_in_annotations": len(images),
        "written_images": written_images,
        "written_labels": written_labels,
        "missing_images": missing_images,
        "ambiguous_images": ambiguous_images,
        "missing_records": missing_records,
        "ambiguous_records": ambiguous_records,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "num_classes": len(names),
        "written_images": written_images,
        "written_labels": written_labels,
        "missing_images": missing_images,
        "ambiguous_images": ambiguous_images,
        "report_path": str(report_path),
    }


def main() -> None:
    args = parse_args()
    stats = convert_coco_to_yolo(
        images_dir=args.images_dir,
        annotations_path=args.annotations,
        output_dir=args.output_dir,
        split=args.split,
        link_mode=args.link_mode,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
