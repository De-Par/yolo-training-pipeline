#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.common import run_cli_with_progress, stdout_logger
from core.datasets.convert_dataset_to_yolo import ConvertDatasetOptions, convert_dataset_to_yolo, supported_input_formats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a raw detection dataset into a YOLO-styled dataset.")
    parser.add_argument("--dataset-name", type=str, required=True, help="Logical dataset name for the converted output directory.")
    parser.add_argument(
        "--input-format",
        choices=supported_input_formats(),
        required=True,
        help="Raw annotation schema adapter to use.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/converted"), help="Where converted YOLO-style datasets are written.")
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink", help="How to place images into the converted dataset.")
    parser.add_argument("--train-fraction", type=float, default=1.0, help="Fraction of train annotations/images to keep during conversion.")
    parser.add_argument("--val-fraction", type=float, default=1.0, help="Fraction of val annotations/images to keep during conversion.")
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed used when sampling train/val fractions below 1.0.")
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=False, help="Clean the converted dataset directory before regeneration.")
    parser.add_argument("--train-images-dir", type=Path, required=True, help="Directory containing raw train images.")
    parser.add_argument("--train-annotations", type=Path, required=True, help="Raw train annotations file or directory, depending on --input-format.")
    parser.add_argument("--val-images-dir", type=Path, required=True, help="Directory containing raw val images.")
    parser.add_argument("--val-annotations", type=Path, required=True, help="Raw val annotations file or directory, depending on --input-format.")
    parser.add_argument("--class-names-file", type=Path, default=None, help="Required for per-image-json input format.")
    parser.add_argument("--object-prefix", type=str, default="item", help="Object key prefix for per-image-json format.")
    parser.add_argument("--category-id-key", type=str, default="category_id", help="Field name used to read class ids in per-image-json annotations.")
    parser.add_argument("--bbox-key", type=str, default="bounding_box", help="Field name used to read bounding boxes in per-image-json annotations.")
    parser.add_argument("--bbox-format", choices=["xyxy", "xywh"], default="xyxy", help="Bounding box format stored in per-image-json annotations.")
    parser.add_argument("--image-width-key", type=str, default="width", help="Field name used to read image width in per-image-json annotations. Falls back to the actual image size when missing.")
    parser.add_argument("--image-height-key", type=str, default="height", help="Field name used to read image height in per-image-json annotations. Falls back to the actual image size when missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = ConvertDatasetOptions(
        dataset_name=args.dataset_name,
        input_format=args.input_format,
        output_root=args.output_root,
        link_mode=args.link_mode,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        sample_seed=args.sample_seed,
        clean=args.clean,
        train_images_dir=args.train_images_dir,
        train_annotations=args.train_annotations,
        val_images_dir=args.val_images_dir,
        val_annotations=args.val_annotations,
        class_names_file=args.class_names_file,
        object_prefix=args.object_prefix,
        category_id_key=args.category_id_key,
        bbox_key=args.bbox_key,
        bbox_format=args.bbox_format,
        image_width_key=args.image_width_key,
        image_height_key=args.image_height_key,
    )
    def _run(progress_callback):
        return convert_dataset_to_yolo(
            options,
            logger=stdout_logger,
            progress_callback=progress_callback,
        )

    run_cli_with_progress(
        desc="convert dataset",
        unit="item",
        action=_run,
    )


if __name__ == "__main__":
    main()
