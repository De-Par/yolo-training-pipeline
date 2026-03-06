from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import yaml

from core.common import PipelineError, ProgressCallback, format_info
from core.datasets.common import count_images, iter_image_files, load_class_names, write_classes_txt, write_dataset_yaml
from core.datasets.filtering import normalize_name, parse_class_selectors
from core.datasets.pipeline_utils import clamp_fraction, require_existing, slow_path_warning

LogFn = Callable[[str], None]


@dataclass(slots=True)
class PrepareYoloDatasetOptions:
    dataset_dir: Path
    recipe_path: Path


@dataclass(slots=True)
class PrepareRecipe:
    dataset_name: Optional[str]
    sample_seed: int
    train_fraction: float
    val_fraction: float
    empty_policy: str
    keep_tokens: List[str]
    drop_tokens: List[str]
    remap_rules: List[dict[str, Any]]
    raw: Dict[str, Any]


def _noop(_: str) -> None:
    return None


def _sample_paths(paths: list[Path], fraction: float, seed: int) -> set[Path]:
    if fraction >= 1.0 or not paths:
        return set(paths)
    import random

    selected_count = max(1, int(round(len(paths) * fraction)))
    randomizer = random.Random(seed)
    selected_indices = set(randomizer.sample(range(len(paths)), selected_count))
    return {path for index, path in enumerate(paths) if index in selected_indices}


def _coerce_selector_tokens(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _load_recipe(recipe_path: Path) -> PrepareRecipe:
    recipe_path = require_existing(recipe_path, "--recipe")
    payload = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise PipelineError(f"Prepare recipe must contain a YAML mapping: {recipe_path}")

    sampling = payload.get("sampling") or {}
    if sampling is None:
        sampling = {}
    if not isinstance(sampling, dict):
        raise PipelineError("Recipe field 'sampling' must be a mapping")

    classes = payload.get("classes") or {}
    if classes is None:
        classes = {}
    if not isinstance(classes, dict):
        raise PipelineError("Recipe field 'classes' must be a mapping")

    remap_rules = classes.get("remap") or []
    if not isinstance(remap_rules, list):
        raise PipelineError("Recipe field 'classes.remap' must be a list")

    recipe = PrepareRecipe(
        dataset_name=(str(payload.get("dataset_name")).strip() if payload.get("dataset_name") is not None else None),
        sample_seed=int(payload.get("sample_seed", 42)),
        train_fraction=clamp_fraction(float(sampling.get("train_fraction", 1.0)), "sampling.train_fraction"),
        val_fraction=clamp_fraction(float(sampling.get("val_fraction", 1.0)), "sampling.val_fraction"),
        empty_policy=str(payload.get("empty_policy", "drop")).strip().lower(),
        keep_tokens=_coerce_selector_tokens(classes.get("keep")),
        drop_tokens=_coerce_selector_tokens(classes.get("drop")),
        remap_rules=remap_rules,
        raw=payload,
    )
    if recipe.empty_policy not in {"drop", "keep"}:
        raise PipelineError("Recipe field 'empty_policy' must be 'drop' or 'keep'")
    return recipe


def _recipe_requests_mutation(recipe: PrepareRecipe, *, dataset_dir: Path) -> bool:
    if recipe.train_fraction != 1.0 or recipe.val_fraction != 1.0:
        return True
    if recipe.keep_tokens or recipe.drop_tokens or recipe.remap_rules:
        return True
    if recipe.dataset_name and recipe.dataset_name != dataset_dir.name:
        return True
    return False


def _remove_unselected_images(
    dataset_dir: Path,
    split: str,
    selected_images: set[Path],
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int]:
    images_dir = require_existing(dataset_dir / "images" / split, f"images/{split}")
    labels_dir = require_existing(dataset_dir / "labels" / split, f"labels/{split}")
    all_images = sorted(iter_image_files(images_dir), key=lambda path: path.name)
    selected_names = {path.name for path in selected_images}
    removed_images = 0
    removed_labels = 0
    total_images = len(all_images)
    if progress_callback is not None:
        progress_callback(f"prepare:sampling:{split}:init", 0, total_images, f"prepare:sampling:{split}")

    for index, image_path in enumerate(all_images, start=1):
        if image_path.name in selected_names:
            if progress_callback is not None:
                progress_callback(
                    f"prepare:sampling:{split}",
                    index,
                    total_images,
                    f"prepare:sampling:{split}: {image_path.name}",
                )
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        image_path.unlink(missing_ok=True)
        removed_images += 1
        if label_path.exists():
            label_path.unlink(missing_ok=True)
            removed_labels += 1
        if progress_callback is not None:
            progress_callback(
                f"prepare:sampling:{split}",
                index,
                total_images,
                f"prepare:sampling:{split}: {image_path.name}",
            )

    return {
        "source_images": len(all_images),
        "kept_images": len(selected_images),
        "removed_images": removed_images,
        "removed_labels": removed_labels,
    }


def _apply_sampling(
    dataset_dir: Path,
    train_fraction: float,
    val_fraction: float,
    sample_seed: int,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    out: Dict[str, Any] = {}
    for index, (split, fraction) in enumerate((("train", train_fraction), ("val", val_fraction))):
        images_dir = require_existing(dataset_dir / "images" / split, f"images/{split}")
        all_images = sorted(iter_image_files(images_dir), key=lambda path: path.name)
        selected = _sample_paths(all_images, fraction, sample_seed + index)
        out[split] = {
            "fraction": fraction,
            "seed": sample_seed + index,
            **_remove_unselected_images(dataset_dir, split, selected, progress_callback=progress_callback),
        }
    return out


def _build_remap_plan(class_names: List[str], recipe: PrepareRecipe) -> tuple[List[str], Dict[int, int], Dict[str, Any]]:
    keep_ids: Optional[Set[int]] = None
    drop_ids: Set[int] = set()
    if recipe.keep_tokens:
        keep_ids = parse_class_selectors(recipe.keep_tokens, class_names)
    if recipe.drop_tokens:
        drop_ids = parse_class_selectors(recipe.drop_tokens, class_names)

    final_keep = set(range(len(class_names))) if keep_ids is None else set(keep_ids)
    final_keep -= drop_ids
    if not final_keep:
        raise PipelineError("After applying recipe keep/drop selectors, no classes remain.")

    assigned_targets: Dict[int, str] = {class_id: class_names[class_id] for class_id in sorted(final_keep)}
    remap_sources_seen: Set[int] = set()
    remap_summary: List[Dict[str, Any]] = []
    for rule in recipe.remap_rules:
        if not isinstance(rule, dict):
            raise PipelineError("Each item in recipe classes.remap must be a mapping")
        target_name = str(rule.get("name", "")).strip()
        if not target_name:
            raise PipelineError("Each remap rule must define a non-empty 'name'")
        source_tokens = _coerce_selector_tokens(rule.get("from"))
        if not source_tokens:
            raise PipelineError(f"Remap rule for '{target_name}' must define non-empty 'from'")
        source_ids = parse_class_selectors(source_tokens, class_names)
        if not source_ids:
            raise PipelineError(f"Remap rule for '{target_name}' resolved to zero classes")
        if not source_ids.issubset(final_keep):
            invalid = sorted(source_ids - final_keep)
            raise PipelineError(f"Remap rule for '{target_name}' references dropped classes: {invalid}")
        overlap = remap_sources_seen & source_ids
        if overlap:
            raise PipelineError(f"Multiple remap rules target the same source classes: {sorted(overlap)}")
        remap_sources_seen.update(source_ids)
        for source_id in sorted(source_ids):
            assigned_targets[source_id] = target_name
        remap_summary.append({"name": target_name, "source_ids": sorted(source_ids)})

    new_class_names: List[str] = []
    target_to_new_id: Dict[str, int] = {}
    old_to_new: Dict[int, int] = {}
    for old_id in sorted(final_keep):
        target_name = assigned_targets[old_id]
        target_key = normalize_name(target_name)
        if target_key not in target_to_new_id:
            target_to_new_id[target_key] = len(new_class_names)
            new_class_names.append(target_name)
        old_to_new[old_id] = target_to_new_id[target_key]

    summary = {
        "kept_old_ids": sorted(final_keep),
        "dropped_old_ids": sorted(set(range(len(class_names))) - final_keep),
        "old_to_new": old_to_new,
        "new_class_names": new_class_names,
        "remap_rules": remap_summary,
        "empty_policy": recipe.empty_policy,
    }
    return new_class_names, old_to_new, summary


def _apply_remap_in_place(
    dataset_dir: Path,
    old_to_new: Dict[int, int],
    empty_policy: str,
    progress_callback: ProgressCallback | None = None,
) -> Dict[str, Any]:
    if empty_policy not in {"drop", "keep"}:
        raise PipelineError(f"Unsupported empty_policy='{empty_policy}'")

    drop_empty = empty_policy == "drop"
    summary: Dict[str, Any] = {"dropped_empty_train": 0, "dropped_empty_val": 0}
    for split in ["train", "val"]:
        labels_dir = require_existing(dataset_dir / "labels" / split, f"labels/{split}")
        images_dir = require_existing(dataset_dir / "images" / split, f"images/{split}")
        image_map = {path.stem: path for path in iter_image_files(images_dir)}
        dropped_empty = 0
        rewritten_labels = 0
        label_paths = sorted(labels_dir.glob("*.txt"))
        total_labels = len(label_paths)
        if progress_callback is not None:
            progress_callback(f"prepare:remap:{split}:init", 0, total_labels, f"prepare:remap:{split}")

        for index, label_path in enumerate(label_paths, start=1):
            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                if drop_empty and label_path.stem in image_map:
                    image_map[label_path.stem].unlink(missing_ok=True)
                    label_path.unlink(missing_ok=True)
                    dropped_empty += 1
                if progress_callback is not None:
                    progress_callback(
                        f"prepare:remap:{split}",
                        index,
                        total_labels,
                        f"prepare:remap:{split}: {label_path.name}",
                    )
                continue

            new_lines: List[str] = []
            for line in text.splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    old_class_id = int(float(parts[0]))
                except Exception:
                    continue
                if old_class_id not in old_to_new:
                    continue
                parts[0] = str(old_to_new[old_class_id])
                new_lines.append(" ".join(parts))

            if not new_lines:
                if drop_empty and label_path.stem in image_map:
                    image_map[label_path.stem].unlink(missing_ok=True)
                    label_path.unlink(missing_ok=True)
                    dropped_empty += 1
                else:
                    label_path.write_text("", encoding="utf-8")
                    rewritten_labels += 1
            else:
                label_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                rewritten_labels += 1
            if progress_callback is not None:
                progress_callback(
                    f"prepare:remap:{split}",
                    index,
                    total_labels,
                    f"prepare:remap:{split}: {label_path.name}",
                )

        summary[f"dropped_empty_{split}"] = dropped_empty
        summary[f"rewritten_labels_{split}"] = rewritten_labels
    return summary


def _remove_stale_dataset_yamls(dataset_dir: Path, recipe_path: Path) -> None:
    recipe_resolved = recipe_path.resolve()
    for path in dataset_dir.glob("*.yaml"):
        if path.resolve() == recipe_resolved:
            continue
        path.unlink(missing_ok=True)


def prepare_yolo_dataset(
    options: PrepareYoloDatasetOptions,
    logger: Optional[LogFn] = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    log = logger or _noop
    dataset_dir = require_existing(options.dataset_dir, "--dataset-dir").resolve()
    recipe = _load_recipe(options.recipe_path)

    for message in slow_path_warning(dataset_dir, "dataset-dir"):
        log(message)

    classes_path = require_existing(dataset_dir / "classes.txt", "classes.txt")
    class_names = load_class_names(classes_path)
    if not class_names:
        raise PipelineError(f"No class names found in {classes_path}")
    if not _recipe_requests_mutation(recipe, dataset_dir=dataset_dir):
        raise PipelineError(
            "Prepare recipe does not request any dataset changes.",
            hint="Set train_fraction/val_fraction below 1.0, add keep/drop/remap rules, or change dataset_name.",
        )

    log(format_info(f"Preparing YOLO dataset in place: {dataset_dir}"))
    log(format_info(f"Using recipe file: {options.recipe_path.resolve()}"))

    sampling_summary = _apply_sampling(
        dataset_dir,
        recipe.train_fraction,
        recipe.val_fraction,
        recipe.sample_seed,
        progress_callback=progress_callback,
    )
    if count_images(dataset_dir / "images" / "train") == 0 or count_images(dataset_dir / "images" / "val") == 0:
        raise PipelineError("After sampling, train/val became empty. Relax sampling fractions or reconvert the dataset.")

    new_class_names, old_to_new, remap_plan = _build_remap_plan(class_names, recipe)
    rewrite_summary = _apply_remap_in_place(
        dataset_dir,
        old_to_new,
        recipe.empty_policy,
        progress_callback=progress_callback,
    )
    if count_images(dataset_dir / "images" / "train") == 0 or count_images(dataset_dir / "images" / "val") == 0:
        raise PipelineError(
            "After class filtering/remapping, train/val became empty. Relax the recipe or reconvert the dataset."
        )

    dataset_name = recipe.dataset_name or dataset_dir.name
    if progress_callback is not None:
        progress_callback("prepare:finalize:init", 0, 4, "prepare:finalize")
    _remove_stale_dataset_yamls(dataset_dir, options.recipe_path)
    if progress_callback is not None:
        progress_callback("prepare:finalize", 1, 4, "prepare:finalize: remove stale yaml")
    classes_txt = write_classes_txt(dataset_dir, new_class_names)
    if progress_callback is not None:
        progress_callback("prepare:finalize", 2, 4, "prepare:finalize: write classes.txt")
    data_yaml = write_dataset_yaml(dataset_dir, new_class_names, dataset_name)
    if progress_callback is not None:
        progress_callback("prepare:finalize", 3, 4, "prepare:finalize: write data.yaml")

    prepare_report = {
        "dataset_dir": str(dataset_dir),
        "dataset_name": dataset_name,
        "recipe_path": str(options.recipe_path.resolve()),
        "sampling": sampling_summary,
        "class_transform": remap_plan,
        "rewrite_summary": rewrite_summary,
        "num_classes": len(new_class_names),
        "data_yaml": str(data_yaml.resolve()),
        "mutated_in_place": True,
    }
    prepare_report_path = dataset_dir / "prepare_report.json"
    prepare_report_path.write_text(json.dumps(prepare_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if progress_callback is not None:
        progress_callback("prepare:finalize", 4, 4, "prepare:finalize: write report")

    log(format_info(f"Wrote classes.txt: {classes_txt}"))
    log(format_info(f"Wrote data.yaml: {data_yaml}"))
    log(format_info(f"Wrote prepare report: {prepare_report_path}"))
    log(
        format_info(
            f"Final dataset state: classes={len(new_class_names)}, "
            f"train_images={count_images(dataset_dir / 'images' / 'train')}, "
            f"val_images={count_images(dataset_dir / 'images' / 'val')}"
        )
    )
    log(format_info("Completed"))

    return {
        "dataset_dir": dataset_dir,
        "class_names": new_class_names,
        "data_yaml": data_yaml,
        "prepare_report_path": prepare_report_path,
        "sampling": sampling_summary,
        "class_transform": remap_plan,
        "rewrite_summary": rewrite_summary,
    }
