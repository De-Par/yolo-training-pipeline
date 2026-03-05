#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import yaml

from pathlib import Path
from typing import Dict, List, Set


def die(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_csv_tokens(values: List[str] | None) -> List[str]:
    if not values:
        return []
    out: List[str] = []
    for value in values:
        for token in str(value).split(","):
            t = token.strip()
            if t:
                out.append(t)
    return out


def safe_link_or_copy(src: Path, dst: Path, link_mode: str) -> None:
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


def find_splits(root: Path) -> List[str]:
    splits: List[str] = []
    for split in ("train", "val", "test"):
        if (root / "labels" / split).exists() and (root / "images" / split).exists():
            splits.append(split)
    return splits


def load_class_names(classes_txt: Path) -> List[str]:
    if not classes_txt.exists():
        die(f"Missing classes file: {classes_txt}")
    names = [line.strip() for line in classes_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        die(f"classes.txt is empty: {classes_txt}")
    return names


def parse_keep_ids(ids_raw: List[str], names_raw: List[str], class_names: List[str]) -> Set[int]:
    keep_ids: Set[int] = set()

    for tok in ids_raw:
        s = tok.strip()
        if s.startswith("+"):
            s = s[1:]
        if not s.isdigit():
            die(f"Invalid class id: '{tok}'")
        cid = int(s)
        if cid < 0 or cid >= len(class_names):
            die(f"Class id out of range: {cid}. Valid range: [0, {len(class_names) - 1}]")
        keep_ids.add(cid)

    name_to_id = {name: i for i, name in enumerate(class_names)}
    unknown: List[str] = []
    for name in names_raw:
        if name in name_to_id:
            keep_ids.add(name_to_id[name])
        else:
            unknown.append(name)
    if unknown:
        die(f"Unknown class names: {', '.join(unknown)}")

    if not keep_ids:
        die("No classes selected. Use --class-ids and/or --class-names.")
    return keep_ids


def build_image_index(images_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        index[p.stem] = p
    return index


def write_dataset_yaml(output_dir: Path, dataset_name: str, class_names: List[str], splits: List[str]) -> Path:
    cfg = {
        "path": str(output_dir.resolve()),
        "names": {i: name for i, name in enumerate(class_names)},
        "nc": len(class_names),
    }
    if "train" in splits:
        cfg["train"] = "images/train"
    if "val" in splits:
        cfg["val"] = "images/val"
    if "test" in splits:
        cfg["test"] = "images/test"

    out_yaml = output_dir / f"{dataset_name}.yaml"
    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
    return out_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a prepared YOLO dataset by class IDs/names and remap kept classes to contiguous IDs."
    )
    parser.add_argument("--src", type=Path, required=True, help="Source prepared YOLO dataset root.")
    parser.add_argument("--dst", type=Path, required=True, help="Output filtered dataset root.")
    parser.add_argument("--class-ids", action="append", help='IDs to keep, comma-separated. Example: "23,24".')
    parser.add_argument(
        "--class-names",
        action="append",
        help='Class names to keep, comma-separated. Example: "shoe,bag, wallet".',
    )
    parser.add_argument(
        "--splits",
        type=str,
        default=None,
        help='Optional split filter, comma-separated. Example: "train,val". Default: all available.',
    )
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep images whose labels become empty after filtering (background negatives).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing destination folder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()

    if not src.exists():
        die(f"Source dataset does not exist: {src}")

    all_splits = find_splits(src)
    if not all_splits:
        die(f"No valid splits found under {src}/labels and {src}/images")

    requested_splits = (
        parse_csv_tokens([args.splits]) if args.splits is not None else all_splits
    )
    if not requested_splits:
        die("Empty --splits value.")
    splits = [s for s in requested_splits if s in all_splits]
    invalid = [s for s in requested_splits if s not in all_splits]
    if invalid:
        die(f"Requested splits not found in source dataset: {', '.join(invalid)}")

    class_names = load_class_names(src / "classes.txt")
    ids_raw = parse_csv_tokens(args.class_ids)
    names_raw = parse_csv_tokens(args.class_names)
    keep_ids = parse_keep_ids(ids_raw, names_raw, class_names)
    kept_ids_sorted = sorted(keep_ids)
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(kept_ids_sorted)}
    kept_class_names = [class_names[i] for i in kept_ids_sorted]

    if dst.exists() and not args.overwrite and any(dst.iterdir()):
        die(f"Destination exists and is not empty: {dst}. Use --overwrite to allow.")
    dst.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, int]] = {}

    log(f"[INFO] Source: {src}")
    log(f"[INFO] Destination: {dst}")
    log(f"[INFO] Splits: {', '.join(splits)}")
    log(f"[INFO] Keep class IDs: {kept_ids_sorted}")
    log(f"[INFO] Keep class names: {kept_class_names}")

    for split in splits:
        labels_in = src / "labels" / split
        images_in = src / "images" / split
        labels_out = dst / "labels" / split
        images_out = dst / "images" / split
        labels_out.mkdir(parents=True, exist_ok=True)
        images_out.mkdir(parents=True, exist_ok=True)

        image_index = build_image_index(images_in)
        written_images = 0
        written_labels = 0
        skipped_empty = 0
        skipped_missing_image = 0
        total_label_files = 0
        kept_boxes = 0
        removed_boxes = 0

        for label_path in sorted(labels_in.glob("*.txt")):
            total_label_files += 1
            lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            out_lines: List[str] = []

            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    old_cls = int(float(parts[0]))
                except ValueError:
                    continue
                if old_cls not in old_to_new:
                    removed_boxes += 1
                    continue
                parts[0] = str(old_to_new[old_cls])
                out_lines.append(" ".join(parts))
                kept_boxes += 1

            if not out_lines and not args.keep_empty:
                skipped_empty += 1
                continue

            stem = label_path.stem
            src_img = image_index.get(stem)
            if src_img is None or not src_img.exists():
                skipped_missing_image += 1
                continue

            dst_img = images_out / src_img.name
            safe_link_or_copy(src_img, dst_img, args.link_mode)
            written_images += 1

            dst_lbl = labels_out / label_path.name
            dst_lbl.write_text("\n".join(out_lines), encoding="utf-8")
            written_labels += 1

        summary[split] = {
            "total_label_files": total_label_files,
            "written_images": written_images,
            "written_labels": written_labels,
            "skipped_empty": skipped_empty,
            "skipped_missing_image": skipped_missing_image,
            "kept_boxes": kept_boxes,
            "removed_boxes": removed_boxes,
        }

        log(
            f"[INFO] {split}: total={total_label_files} written={written_labels} "
            f"skipped_empty={skipped_empty} kept_boxes={kept_boxes} removed_boxes={removed_boxes}"
        )

    (dst / "classes.txt").write_text("\n".join(kept_class_names), encoding="utf-8")
    dataset_yaml = write_dataset_yaml(dst, dst.name, kept_class_names, splits)

    report = {
        "source": str(src),
        "destination": str(dst),
        "splits": splits,
        "keep_ids_old": kept_ids_sorted,
        "keep_class_names": kept_class_names,
        "old_to_new": {str(k): v for k, v in old_to_new.items()},
        "keep_empty": bool(args.keep_empty),
        "link_mode": args.link_mode,
        "summary": summary,
        "dataset_yaml": str(dataset_yaml),
    }
    report_path = dst / "filter_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"[INFO] classes.txt: {dst / 'classes.txt'}")
    log(f"[INFO] data.yaml:   {dataset_yaml}")
    log(f"[INFO] report:      {report_path}")
    log("[INFO] Done.")


if __name__ == "__main__":
    main()
