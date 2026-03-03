#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import yaml

from convert_coco_to_yolo import convert_coco_to_yolo
from convert_deepfashion2_to_yolo import convert_deepfashion2_to_yolo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end YOLO training pipeline for Fashionpedia or DeepFashion2."
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
    parser.add_argument(
        "--model",
        type=str,
        help="Model checkpoint/config for training (required unless --prepare-only).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="yolo-train")
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--custom-name", type=str, default="custom", help="Dataset folder name for custom-coco mode.")
    parser.add_argument("--train-images-dir", type=Path, help="custom-coco: train images directory.")
    parser.add_argument("--train-annotations", type=Path, help="custom-coco: train COCO annotations JSON.")
    parser.add_argument("--val-images-dir", type=Path, help="custom-coco: val images directory.")
    parser.add_argument("--val-annotations", type=Path, help="custom-coco: val COCO annotations JSON.")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare YOLO dataset and data.yaml without starting training.",
    )
    return parser.parse_args()


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


def load_class_names(classes_txt: Path) -> List[str]:
    return [line.strip() for line in classes_txt.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare_fashionpedia(args: argparse.Namespace, output_dir: Path) -> Dict[str, int]:
    train_stats = convert_coco_to_yolo(
        images_dir=args.raw_root / "train" / "images",
        annotations_path=args.raw_root / "train" / "annotations.json",
        output_dir=output_dir,
        split="train",
        link_mode=args.link_mode,
    )
    val_stats = convert_coco_to_yolo(
        images_dir=args.raw_root / "val" / "images",
        annotations_path=args.raw_root / "val" / "annotations.json",
        output_dir=output_dir,
        split="val",
        link_mode=args.link_mode,
    )
    return {
        "num_classes": train_stats["num_classes"],
        "train_images": train_stats["written_images"],
        "val_images": val_stats["written_images"],
        "train_missing_images": train_stats.get("missing_images", 0),
        "val_missing_images": val_stats.get("missing_images", 0),
        "train_ambiguous_images": train_stats.get("ambiguous_images", 0),
        "val_ambiguous_images": val_stats.get("ambiguous_images", 0),
    }


def prepare_deepfashion2(args: argparse.Namespace, output_dir: Path) -> Dict[str, int]:
    train_stats = convert_deepfashion2_to_yolo(
        images_dir=args.raw_root / "train" / "image",
        annos_dir=args.raw_root / "train" / "annos",
        output_dir=output_dir,
        split="train",
        link_mode=args.link_mode,
    )
    val_stats = convert_deepfashion2_to_yolo(
        images_dir=args.raw_root / "validation" / "image",
        annos_dir=args.raw_root / "validation" / "annos",
        output_dir=output_dir,
        split="val",
        link_mode=args.link_mode,
    )
    return {
        "num_classes": train_stats["num_classes"],
        "train_images": train_stats["written_images"],
        "val_images": val_stats["written_images"],
        "train_missing_images": train_stats.get("missing_images", 0),
        "val_missing_images": val_stats.get("missing_images", 0),
    }


def prepare_custom_coco(args: argparse.Namespace, output_dir: Path) -> Dict[str, int]:
    required = {
        "--train-images-dir": args.train_images_dir,
        "--train-annotations": args.train_annotations,
        "--val-images-dir": args.val_images_dir,
        "--val-annotations": args.val_annotations,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"custom-coco requires: {', '.join(missing)}")

    train_stats = convert_coco_to_yolo(
        images_dir=args.train_images_dir,
        annotations_path=args.train_annotations,
        output_dir=output_dir,
        split="train",
        link_mode=args.link_mode,
    )
    val_stats = convert_coco_to_yolo(
        images_dir=args.val_images_dir,
        annotations_path=args.val_annotations,
        output_dir=output_dir,
        split="val",
        link_mode=args.link_mode,
    )
    return {
        "num_classes": train_stats["num_classes"],
        "train_images": train_stats["written_images"],
        "val_images": val_stats["written_images"],
        "train_missing_images": train_stats.get("missing_images", 0),
        "val_missing_images": val_stats.get("missing_images", 0),
        "train_ambiguous_images": train_stats.get("ambiguous_images", 0),
        "val_ambiguous_images": val_stats.get("ambiguous_images", 0),
    }


def main() -> None:
    args = parse_args()

    if args.dataset in {"fashionpedia", "deepfashion2"} and args.raw_root is None:
        raise ValueError("--raw-root is required for dataset modes: fashionpedia, deepfashion2")
    if not args.prepare_only and not args.model:
        raise ValueError("--model is required unless --prepare-only is set")

    dataset_key = args.custom_name if args.dataset == "custom-coco" else args.dataset
    output_dir = args.workdir / dataset_key
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "fashionpedia":
        prep_stats = prepare_fashionpedia(args, output_dir)
    elif args.dataset == "deepfashion2":
        prep_stats = prepare_deepfashion2(args, output_dir)
    else:
        prep_stats = prepare_custom_coco(args, output_dir)

    if prep_stats["train_images"] == 0 or prep_stats["val_images"] == 0:
        raise RuntimeError(
            "Prepared dataset has zero train/val images. Check raw dataset structure and annotation paths."
        )

    class_names = load_class_names(output_dir / "classes.txt")
    data_yaml = write_dataset_yaml(output_dir, class_names, dataset_key)

    print(
        f"Prepared dataset={dataset_key} classes={prep_stats['num_classes']} "
        f"train_images={prep_stats['train_images']} val_images={prep_stats['val_images']}"
    )
    if prep_stats.get("train_missing_images", 0) or prep_stats.get("val_missing_images", 0):
        print(
            "Warning: missing images "
            f"train={prep_stats.get('train_missing_images', 0)} "
            f"val={prep_stats.get('val_missing_images', 0)}"
        )
    if prep_stats.get("train_ambiguous_images", 0) or prep_stats.get("val_ambiguous_images", 0):
        print(
            "Warning: ambiguous image basenames "
            f"train={prep_stats.get('train_ambiguous_images', 0)} "
            f"val={prep_stats.get('val_ambiguous_images', 0)}"
        )
    print(f"data.yaml: {data_yaml}")

    if args.prepare_only:
        print("Preparation-only mode enabled. Skipping training.")
        return

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
