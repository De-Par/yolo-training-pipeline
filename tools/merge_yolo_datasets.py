#!/usr/bin/env python3
from __future__ import annotations

from tools._runtime import bootstrap_project_root
bootstrap_project_root(__file__, levels=1)

import argparse
import shutil
import yaml

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple
from core.common import format_info, run_cli_with_progress

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ALL_SPLITS = ("train", "val", "test")

Stats = Dict[str, int]
SplitDirs = Dict[str, Tuple[Path, Path]]
ProgressCallback = Callable[[str, int, int, str], None]


def _empty_stats() -> Stats:
    return {
        "images": 0,
        "labels": 0,
        "backgrounds": 0,
    }


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")

    return data


def parse_names(data: dict, yaml_path: Path) -> List[str]:
    names = data.get("names")
    if names is None:
        raise ValueError(f"'names' is missing in {yaml_path}")

    if isinstance(names, list):
        return [str(x) for x in names]

    if isinstance(names, dict):
        try:
            items = sorted((int(k), str(v)) for k, v in names.items())
        except Exception as e:
            raise ValueError(f"Failed to parse dict 'names' in {yaml_path}: {e}") from e
        return [value for _, value in items]

    raise ValueError(f"'names' must be list or dict in {yaml_path}")


def parse_nc(data: dict, names: List[str], yaml_path: Path) -> int:
    nc = data.get("nc")
    if nc is None:
        return len(names)

    nc = int(nc)
    if nc != len(names):
        raise ValueError(
            f"'nc' does not match len(names) in {yaml_path}: "
            f"nc={nc}, len(names)={len(names)}"
        )

    return nc


def ensure_same_classes(data1: dict, data2: dict, yaml1: Path, yaml2: Path) -> List[str]:
    names1 = parse_names(data1, yaml1)
    names2 = parse_names(data2, yaml2)

    nc1 = parse_nc(data1, names1, yaml1)
    nc2 = parse_nc(data2, names2, yaml2)

    if nc1 != nc2:
        raise ValueError(
            f"Class count mismatch:\n"
            f"  {yaml1}: nc={nc1}\n"
            f"  {yaml2}: nc={nc2}"
        )

    if names1 != names2:
        raise ValueError(
            "Class names or their order do not match:\n"
            f"  {yaml1}: {names1}\n"
            f"  {yaml2}: {names2}"
        )

    return names1


def resolve_dataset_root(data: dict, yaml_path: Path) -> Path:
    base = data.get("path")
    if base is None:
        return yaml_path.resolve().parent

    base_path = Path(str(base))
    if base_path.is_absolute():
        return base_path.resolve()

    return (yaml_path.resolve().parent / base_path).resolve()


def resolve_split_image_dir(dataset_root: Path, data: dict, split: str) -> Path | None:
    value = data.get(split)
    if value is None:
        return None

    split_path = Path(str(value))
    if split_path.is_absolute():
        return split_path.resolve()

    return (dataset_root / split_path).resolve()


def label_dir_from_image_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)

    try:
        idx = parts.index("images")
    except ValueError as e:
        raise ValueError(
            f"Could not infer labels directory from image directory: {image_dir}. "
            f"Expected path containing 'images'."
        ) from e

    parts[idx] = "labels"
    return Path(*parts)


def ensure_declared_split_dirs_exist(data: dict, yaml_path: Path) -> SplitDirs:
    dataset_root = resolve_dataset_root(data, yaml_path)
    result: SplitDirs = {}

    for split in ALL_SPLITS:
        image_dir = resolve_split_image_dir(dataset_root, data, split)
        if image_dir is None:
            continue

        label_dir = label_dir_from_image_dir(image_dir)

        if not image_dir.exists() or not image_dir.is_dir():
            raise FileNotFoundError(
                f"Split '{split}' is declared in {yaml_path}, "
                f"but image dir does not exist: {image_dir}"
            )

        if not label_dir.exists() or not label_dir.is_dir():
            raise FileNotFoundError(
                f"Split '{split}' is declared in {yaml_path}, "
                f"but label dir does not exist: {label_dir}"
            )

        result[split] = (image_dir, label_dir)

    return result


def iter_images(directory: Path) -> Iterable[Path]:
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def count_images(directory: Path) -> int:
    count = 0
    for _ in iter_images(directory):
        count += 1
    return count


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")

    shutil.copy2(src, dst)


def make_prefixed_relpath(prefix: str, rel_path: Path) -> Path:
    if not rel_path.parts:
        raise ValueError("Empty relative path")

    return rel_path.with_name(f"{prefix}_{rel_path.stem}{rel_path.suffix}")


def corresponding_label_path(
    image_path: Path,
    image_split_dir: Path,
    label_split_dir: Path,
) -> Path:
    rel = image_path.relative_to(image_split_dir)
    return (label_split_dir / rel).with_suffix(".txt")


def add_stats(total: Stats, part: Stats) -> None:
    total["images"] += part["images"]
    total["labels"] += part["labels"]
    total["backgrounds"] += part["backgrounds"]


def write_output_yaml_and_classes(
    *,
    out_root: Path,
    names: List[str],
    present_splits: List[str],
    yaml_name: str,
) -> tuple[Path, Path]:
    names_dict = {i: name for i, name in enumerate(names)}

    payload: dict = {
        "path": str(out_root.resolve()),
        "names": names_dict,
        "nc": len(names),
    }

    for split in present_splits:
        payload[split] = f"images/{split}"

    out_yaml = out_root / yaml_name
    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True, default_flow_style=False, indent=2)
    
    out_classes = out_root / "classes.txt"
    with out_classes.open("w", encoding="utf-8") as f:
        f.write("\n".join(names) + "\n")

    return out_yaml, out_classes


def build_merge_context(
    *,
    yaml1: Path,
    yaml2: Path,
    out_root: Path,
    prefix1: str,
    prefix2: str,
    yaml_name: str,
) -> dict:
    if prefix1 == prefix2:
        raise ValueError("--prefix1 and --prefix2 must be different")

    if not yaml1.exists():
        raise FileNotFoundError(f"yaml1 not found: {yaml1}")

    if not yaml2.exists():
        raise FileNotFoundError(f"yaml2 not found: {yaml2}")

    data1 = load_yaml(yaml1)
    data2 = load_yaml(yaml2)

    names = ensure_same_classes(data1, data2, yaml1, yaml2)

    splits1 = ensure_declared_split_dirs_exist(data1, yaml1)
    splits2 = ensure_declared_split_dirs_exist(data2, yaml2)

    present_splits = [split for split in ALL_SPLITS if split in splits1 or split in splits2]
    if not present_splits:
        raise ValueError("Neither dataset declares train/val/test splits")

    scan_targets: List[Tuple[str, str, Path]] = []
    for split in ALL_SPLITS:
        if split in splits1:
            image_dir, _ = splits1[split]
            scan_targets.append(("dataset1", split, image_dir))

        if split in splits2:
            image_dir, _ = splits2[split]
            scan_targets.append(("dataset2", split, image_dir))

    if not scan_targets:
        raise ValueError("No declared image directories found")

    return {
        "yaml1": yaml1,
        "yaml2": yaml2,
        "out_root": out_root,
        "prefix1": prefix1,
        "prefix2": prefix2,
        "yaml_name": yaml_name,
        "names": names,
        "splits1": splits1,
        "splits2": splits2,
        "present_splits": present_splits,
        "scan_targets": scan_targets,
    }


def scan_merge_context(
    *,
    ctx: dict,
    progress_callback: ProgressCallback,
) -> dict:
    scan_targets: List[Tuple[str, str, Path]] = ctx["scan_targets"]
    scan_total = len(scan_targets)

    total_images = 0
    progress_callback("scan", 0, scan_total, "counting images")

    for idx, (dataset_name, split, image_dir) in enumerate(scan_targets, start=1):
        num_images = count_images(image_dir)
        total_images += num_images
        progress_callback(
            "scan",
            idx,
            scan_total,
            f"{dataset_name}/{split}: {num_images} images",
        )

    if total_images == 0:
        raise ValueError("No images found in declared splits")

    updated = dict(ctx)
    updated["total_images"] = total_images
    return updated


def prepare_output_dir(
    *,
    ctx: dict,
    progress_callback: ProgressCallback,
) -> dict:
    out_root: Path = ctx["out_root"]
    present_splits: List[str] = ctx["present_splits"]

    total_steps = 2 + len(present_splits)
    current = 0

    progress_callback("prepare", current, total_steps, "preparing output directory")

    if out_root.exists():
        shutil.rmtree(out_root)
    current += 1
    progress_callback("prepare", current, total_steps, "removed previous output")

    out_root.mkdir(parents=True, exist_ok=True)
    current += 1
    progress_callback("prepare", current, total_steps, "created output root")

    for split in present_splits:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        current += 1
        progress_callback("prepare", current, total_steps, f"created split dirs: {split}")

    return ctx


def merge_dataset_split(
    *,
    split: str,
    src_image_dir: Path,
    src_label_dir: Path,
    out_root: Path,
    prefix: str,
    progress_callback: ProgressCallback,
    current: int,
    total: int,
) -> tuple[Stats, int]:
    stats = _empty_stats()

    out_image_dir = out_root / "images" / split
    out_label_dir = out_root / "labels" / split

    for image_path in iter_images(src_image_dir):
        rel = image_path.relative_to(src_image_dir)
        out_rel = make_prefixed_relpath(prefix, rel)

        dst_image = out_image_dir / out_rel
        copy_file(image_path, dst_image)
        stats["images"] += 1

        src_label = corresponding_label_path(image_path, src_image_dir, src_label_dir)
        if src_label.exists():
            dst_label = (out_label_dir / out_rel).with_suffix(".txt")
            copy_file(src_label, dst_label)
            stats["labels"] += 1
        else:
            stats["backgrounds"] += 1

        current += 1
        progress_callback("copy", current, total, f"{split}: {image_path.name}")

    return stats, current


def execute_merge(
    *,
    ctx: dict,
    progress_callback: ProgressCallback,
) -> dict:
    out_root: Path = ctx["out_root"]
    present_splits: List[str] = ctx["present_splits"]
    splits1: SplitDirs = ctx["splits1"]
    splits2: SplitDirs = ctx["splits2"]
    total_images: int = ctx["total_images"]

    dataset1_stats = {split: _empty_stats() for split in present_splits}
    dataset2_stats = {split: _empty_stats() for split in present_splits}
    total_stats = _empty_stats()

    current = 0
    progress_callback("copy", current, total_images, "starting merge")

    for split in present_splits:
        if split in splits1:
            src_image_dir, src_label_dir = splits1[split]
            stats, current = merge_dataset_split(
                split=split,
                src_image_dir=src_image_dir,
                src_label_dir=src_label_dir,
                out_root=out_root,
                prefix=ctx["prefix1"],
                progress_callback=progress_callback,
                current=current,
                total=total_images,
            )
            dataset1_stats[split] = stats
            add_stats(total_stats, stats)

        if split in splits2:
            src_image_dir, src_label_dir = splits2[split]
            stats, current = merge_dataset_split(
                split=split,
                src_image_dir=src_image_dir,
                src_label_dir=src_label_dir,
                out_root=out_root,
                prefix=ctx["prefix2"],
                progress_callback=progress_callback,
                current=current,
                total=total_images,
            )
            dataset2_stats[split] = stats
            add_stats(total_stats, stats)

    out_yaml, out_classes = write_output_yaml_and_classes(
        out_root=out_root,
        names=ctx["names"],
        present_splits=present_splits,
        yaml_name=ctx["yaml_name"],
    )

    return {
        "yaml1": ctx["yaml1"],
        "yaml2": ctx["yaml2"],
        "out_root": out_root,
        "out_yaml": out_yaml,
        "out_classes": out_classes,
        "classes": ctx["names"],
        "num_classes": len(ctx["names"]),
        "present_splits": present_splits,
        "dataset1": dataset1_stats,
        "dataset2": dataset2_stats,
        "total": total_stats,
        "total_images": total_images,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge two YOLO datasets with identical classes and split structure.")
    parser.add_argument("--yaml1", type=Path, required=True, help="Path to first dataset data.yaml.")
    parser.add_argument("--yaml2", type=Path, required=True, help="Path to second dataset data.yaml.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for merged dataset.")
    parser.add_argument("--prefix1", type=str, required=True, help="Filename prefix for dataset 1.")
    parser.add_argument("--prefix2", type=str, required=True, help="Filename prefix for dataset 2.")
    parser.add_argument("--yaml-name", type=str, default="dataset.yaml", help="Output YAML filename.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ctx = build_merge_context(
        yaml1=args.yaml1.resolve(),
        yaml2=args.yaml2.resolve(),
        out_root=args.out.resolve(),
        prefix1=args.prefix1,
        prefix2=args.prefix2,
        yaml_name=args.yaml_name,
    )

    ctx = run_cli_with_progress(
        desc="scan datasets",
        unit="split",
        action=lambda progress_callback: scan_merge_context(
            ctx=ctx,
            progress_callback=progress_callback,
        ),
    )

    ctx = run_cli_with_progress(
        desc="prepare output",
        unit="step",
        action=lambda progress_callback: prepare_output_dir(
            ctx=ctx,
            progress_callback=progress_callback,
        ),
    )

    report = run_cli_with_progress(
        desc="merge datasets",
        unit="img",
        action=lambda progress_callback: execute_merge(
            ctx=ctx,
            progress_callback=progress_callback,
        ),
    )

    print(format_info(f"Merged dataset root: {report['out_root']}"))
    print(format_info(f"Wrote YAML: {report['out_yaml']}"))
    print(format_info(f"Wrote classes: {report['out_classes']}"))
    print(format_info(f"Num classes: {report['num_classes']}"))
    print(format_info(f"Present splits: {', '.join(report['present_splits'])}"))
    print(
        format_info(
            "Total: "
            f"images={report['total']['images']} "
            f"labels={report['total']['labels']} "
            f"backgrounds={report['total']['backgrounds']}"
        )
    )

    print(f"\n{format_info('Dataset 1 stats')}:")
    for split in report["present_splits"]:
        row = report["dataset1"][split]
        print(
            f"  {split:<5} "
            f"images={row['images']:<6d} "
            f"labels={row['labels']:<6d} "
            f"backgrounds={row['backgrounds']:<6d}"
        )

    print(f"\n{format_info('Dataset 2 stats')}:")
    for split in report["present_splits"]:
        row = report["dataset2"][split]
        print(
            f"  {split:<5} "
            f"images={row['images']:<6d} "
            f"labels={row['labels']:<6d} "
            f"backgrounds={row['backgrounds']:<6d}"
        )


if __name__ == "__main__":
    main()
    