#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import yaml

from pathlib import Path
from typing import Any, Dict, List
from convert_coco_to_yolo import convert_coco_to_yolo
from convert_deepfashion2_to_yolo import convert_deepfashion2_to_yolo


def log(msg: str) -> None:
    print(msg, flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def is_wsl() -> bool:
    try:
        return "WSL_DISTRO_NAME" in os.environ or "microsoft" in Path("/proc/version").read_text().lower()
    except Exception:
        return False


def warn_on_slow_paths(path: Path, what: str) -> None:
    try:
        p = str(path.resolve())
    except Exception:
        p = str(path)
    if is_wsl() and (p.startswith("/mnt/c/") or p.startswith("/mnt/C/")):
        log(f"[WARN] {what} is on /mnt/c (NTFS) which is often slow in WSL: {p}")
        log("[WARN] Prefer Linux FS (~/...) or /mnt/e on SSD for better performance.")


def clamp_fraction(x: float, name: str) -> float:
    if not (0.0 < x <= 1.0):
        die(f"{name} must be in (0, 1], got {x}")
    return x


def load_class_names(classes_txt: Path) -> List[str]:
    if not classes_txt.exists():
        die(f"Missing {classes_txt}. Conversion step should have created classes.txt")
    return [line.strip() for line in classes_txt.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_dataset_yaml(output_dir: Path, class_names: List[str], dataset_name: str) -> Path:
    yaml_path = output_dir / f"{dataset_name}.yaml"
    cfg = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
        "nc": len(class_names),
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
    return yaml_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare YOLO dataset (images/labels + data.yaml) for Fashionpedia, DeepFashion2, or custom COCO."
    )

    parser.add_argument("--dataset", choices=["fashionpedia", "deepfashion2", "custom-coco"], required=True)
    parser.add_argument(
        "--raw-root",
        type=Path,
        help="Root folder for built-in dataset modes (fashionpedia/deepfashion2).",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("data/processed"),
        help="Where converted YOLO dataset and data.yaml are saved.",
    )

    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")

    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=42)

    # custom-coco mode
    parser.add_argument("--custom-name", type=str, default="custom", help="Dataset folder name for custom-coco mode.")
    parser.add_argument("--train-images-dir", type=Path, help="custom-coco: train images directory.")
    parser.add_argument("--train-annotations", type=Path, help="custom-coco: train COCO annotations JSON.")
    parser.add_argument("--val-images-dir", type=Path, help="custom-coco: val images directory.")
    parser.add_argument("--val-annotations", type=Path, help="custom-coco: val COCO annotations JSON.")

    return parser.parse_args()


def prepare_fashionpedia(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    train_stats = convert_coco_to_yolo(
        images_dir=args.raw_root / "train" / "images",
        annotations_path=args.raw_root / "train" / "annotations.json",
        output_dir=output_dir,
        split="train",
        link_mode=args.link_mode,
        sample_fraction=args.train_fraction,
        sample_seed=args.sample_seed,
    )
    val_stats = convert_coco_to_yolo(
        images_dir=args.raw_root / "val" / "images",
        annotations_path=args.raw_root / "val" / "annotations.json",
        output_dir=output_dir,
        split="val",
        link_mode=args.link_mode,
        sample_fraction=args.val_fraction,
        sample_seed=args.sample_seed,
    )
    return {
        "num_classes": train_stats["num_classes"],
        "train_images": train_stats["written_images"],
        "val_images": val_stats["written_images"],
        "train_missing_images": train_stats.get("missing_images", 0),
        "val_missing_images": val_stats.get("missing_images", 0),
        "train_ambiguous_images": train_stats.get("ambiguous_images", 0),
        "val_ambiguous_images": val_stats.get("ambiguous_images", 0),
        "train_report_path": train_stats.get("report_path"),
        "val_report_path": val_stats.get("report_path"),
    }


def prepare_deepfashion2(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    train_stats = convert_deepfashion2_to_yolo(
        images_dir=args.raw_root / "train" / "image",
        annos_dir=args.raw_root / "train" / "annos",
        output_dir=output_dir,
        split="train",
        link_mode=args.link_mode,
        sample_fraction=args.train_fraction,
        sample_seed=args.sample_seed,
    )
    val_stats = convert_deepfashion2_to_yolo(
        images_dir=args.raw_root / "validation" / "image",
        annos_dir=args.raw_root / "validation" / "annos",
        output_dir=output_dir,
        split="val",
        link_mode=args.link_mode,
        sample_fraction=args.val_fraction,
        sample_seed=args.sample_seed,
    )
    return {
        "num_classes": train_stats["num_classes"],
        "train_images": train_stats["written_images"],
        "val_images": val_stats["written_images"],
        "train_missing_images": train_stats.get("missing_images", 0),
        "val_missing_images": val_stats.get("missing_images", 0),
    }


def prepare_custom_coco(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    required = {
        "--train-images-dir": args.train_images_dir,
        "--train-annotations": args.train_annotations,
        "--val-images-dir": args.val_images_dir,
        "--val-annotations": args.val_annotations,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        die(f"custom-coco requires: {', '.join(missing)}")

    train_stats = convert_coco_to_yolo(
        images_dir=args.train_images_dir,
        annotations_path=args.train_annotations,
        output_dir=output_dir,
        split="train",
        link_mode=args.link_mode,
        sample_fraction=args.train_fraction,
        sample_seed=args.sample_seed,
    )
    val_stats = convert_coco_to_yolo(
        images_dir=args.val_images_dir,
        annotations_path=args.val_annotations,
        output_dir=output_dir,
        split="val",
        link_mode=args.link_mode,
        sample_fraction=args.val_fraction,
        sample_seed=args.sample_seed,
    )
    return {
        "num_classes": train_stats["num_classes"],
        "train_images": train_stats["written_images"],
        "val_images": val_stats["written_images"],
        "train_missing_images": train_stats.get("missing_images", 0),
        "val_missing_images": val_stats.get("missing_images", 0),
        "train_ambiguous_images": train_stats.get("ambiguous_images", 0),
        "val_ambiguous_images": val_stats.get("ambiguous_images", 0),
        "train_report_path": train_stats.get("report_path"),
        "val_report_path": val_stats.get("report_path"),
    }


def main() -> None:
    args = parse_args()

    args.train_fraction = clamp_fraction(args.train_fraction, "--train-fraction")
    args.val_fraction = clamp_fraction(args.val_fraction, "--val-fraction")

    if args.dataset in {"fashionpedia", "deepfashion2"} and args.raw_root is None:
        die("--raw-root is required for dataset modes: fashionpedia, deepfashion2")

    dataset_key = args.custom_name if args.dataset == "custom-coco" else args.dataset
    output_dir = (args.workdir / dataset_key).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    warn_on_slow_paths(args.workdir, "workdir")
    if args.raw_root is not None:
        warn_on_slow_paths(args.raw_root, "raw-root")

    log(f"[INFO] Preparing dataset: {args.dataset} -> {output_dir}")
    log(
        "[INFO] Sampling: "
        f"train_fraction={args.train_fraction} val_fraction={args.val_fraction} sample_seed={args.sample_seed} "
        f"link_mode={args.link_mode}"
    )

    if args.dataset == "fashionpedia":
        prep_stats = prepare_fashionpedia(args, output_dir)
    elif args.dataset == "deepfashion2":
        prep_stats = prepare_deepfashion2(args, output_dir)
    else:
        prep_stats = prepare_custom_coco(args, output_dir)

    if prep_stats.get("train_images", 0) == 0 or prep_stats.get("val_images", 0) == 0:
        die("Prepared dataset has zero train/val images. Check raw dataset structure and annotation paths.")

    class_names = load_class_names(output_dir / "classes.txt")
    data_yaml = write_dataset_yaml(output_dir, class_names, dataset_key)

    log(
        "[INFO] Prepared "
        f"dataset={dataset_key} classes={prep_stats.get('num_classes')} "
        f"train_images={prep_stats.get('train_images')} val_images={prep_stats.get('val_images')}"
    )

    if prep_stats.get("train_missing_images", 0) or prep_stats.get("val_missing_images", 0):
        log(
            "[WARN] Missing images "
            f"train={prep_stats.get('train_missing_images', 0)} "
            f"val={prep_stats.get('val_missing_images', 0)}"
        )
    if prep_stats.get("train_ambiguous_images", 0) or prep_stats.get("val_ambiguous_images", 0):
        log(
            "[WARN] Ambiguous image basenames "
            f"train={prep_stats.get('train_ambiguous_images', 0)} "
            f"val={prep_stats.get('val_ambiguous_images', 0)}"
        )
    if prep_stats.get("train_report_path") or prep_stats.get("val_report_path"):
        log(f"[INFO] conversion_report_train: {prep_stats.get('train_report_path')}")
        log(f"[INFO] conversion_report_val:   {prep_stats.get('val_report_path')}")

    log(f"[INFO] data.yaml: {data_yaml}")
    log("[INFO] Done.")


if __name__ == "__main__":
    main()
    