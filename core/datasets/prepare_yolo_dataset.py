from __future__ import annotations

import hashlib
import json
import random

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import yaml

from core.common import PipelineError, ProgressCallback, format_info, format_warning
from core.datasets.common import (
    DATASET_SPLIT_ORDER,
    count_images,
    detect_dataset_splits,
    iter_image_files,
    load_class_names,
    remove_tree,
    write_classes_txt,
    write_dataset_yaml,
)
from core.datasets.filtering import normalize_name, parse_class_selectors
from core.datasets.pipeline_utils import clamp_fraction, clamp_fraction_allow_zero, require_existing, slow_path_warning

LogFn = Callable[[str], None]
COMBINED_SPLIT_MODES = {"resplit_combined_random", "resplit_combined_by_instances"}


@dataclass(slots=True)
class PrepareYoloDatasetOptions:
    dataset_dir: Path
    recipe_path: Path


@dataclass(slots=True)
class PrepareSplitConfig:
    mode: str
    seed: int
    train_fraction: float
    val_fraction: float
    test_fraction: float
    min_val_instances_per_class: int
    min_test_instances_per_class: int
    per_class_min_val_instances: Dict[str, int]
    per_class_min_test_instances: Dict[str, int]


@dataclass(slots=True)
class PrepareRecipe:
    dataset_name: Optional[str]
    empty_policy: str
    split: PrepareSplitConfig
    keep_tokens: List[str]
    drop_tokens: List[str]
    remap_rules: List[dict[str, Any]]
    raw: Dict[str, Any]


@dataclass(slots=True)
class DatasetImageItem:
    source_split: str
    image_path: Path
    label_path: Optional[Path]
    name: str
    stem: str
    relative_key: str
    class_counts: Dict[int, int]


@dataclass(slots=True)
class CollectedDatasetItems:
    items: List[DatasetImageItem]
    dropped_orphan_labels: int


def _noop(_: str) -> None:
    return None


def _coerce_selector_tokens(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _coerce_selector_map(value: Any, field_name: str) -> Dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PipelineError(f"Recipe field '{field_name}' must be a mapping")
    out: Dict[str, int] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip()
        if not key:
            raise PipelineError(f"Recipe field '{field_name}' contains an empty selector key")
        try:
            count = int(raw_value)
        except Exception as exc:
            raise PipelineError(f"Recipe field '{field_name}.{key}' must be an integer") from exc
        if count < 0:
            raise PipelineError(f"Recipe field '{field_name}.{key}' must be >= 0")
        out[key] = count
    return out


def _load_split_config(payload: Dict[str, Any]) -> PrepareSplitConfig:
    split_payload = payload.get("split")
    legacy_sampling = payload.get("sampling")

    if split_payload is None:
        split_payload = {}
    if not isinstance(split_payload, dict):
        raise PipelineError("Recipe field 'split' must be a mapping")

    if legacy_sampling is not None:
        if not isinstance(legacy_sampling, dict):
            raise PipelineError("Recipe field 'sampling' must be a mapping")
        if split_payload:
            raise PipelineError("Use either 'split' or legacy 'sampling' in the recipe, not both")
        split_payload = {
            "mode": "sample_existing",
            "seed": payload.get("sample_seed", 42),
            "train_fraction": legacy_sampling.get("train_fraction", 1.0),
            "val_fraction": legacy_sampling.get("val_fraction", 1.0),
            "test_fraction": legacy_sampling.get("test_fraction", 1.0),
        }

    mode = str(split_payload.get("mode", "keep_existing")).strip().lower()
    if mode not in {"keep_existing", "sample_existing", *COMBINED_SPLIT_MODES}:
        raise PipelineError(
            f"Unsupported split.mode='{mode}'",
            hint="Use one of: keep_existing, sample_existing, resplit_combined_random, resplit_combined_by_instances.",
        )

    test_fraction_default = 1.0 if mode == "sample_existing" else 0.0

    split = PrepareSplitConfig(
        mode=mode,
        seed=int(split_payload.get("seed", payload.get("sample_seed", 42))),
        train_fraction=clamp_fraction_allow_zero(float(split_payload.get("train_fraction", 1.0)), "split.train_fraction"),
        val_fraction=clamp_fraction_allow_zero(float(split_payload.get("val_fraction", 1.0)), "split.val_fraction"),
        test_fraction=clamp_fraction_allow_zero(
            float(split_payload.get("test_fraction", test_fraction_default)),
            "split.test_fraction",
        ),
        min_val_instances_per_class=int(split_payload.get("min_val_instances_per_class", 0)),
        min_test_instances_per_class=int(split_payload.get("min_test_instances_per_class", 0)),
        per_class_min_val_instances=_coerce_selector_map(
            split_payload.get("per_class_min_val_instances"),
            "split.per_class_min_val_instances",
        ),
        per_class_min_test_instances=_coerce_selector_map(
            split_payload.get("per_class_min_test_instances"),
            "split.per_class_min_test_instances",
        ),
    )

    if split.min_val_instances_per_class < 0 or split.min_test_instances_per_class < 0:
        raise PipelineError("split.min_val_instances_per_class and split.min_test_instances_per_class must be >= 0")

    if split.mode in COMBINED_SPLIT_MODES:
        total_fraction = split.train_fraction + split.val_fraction + split.test_fraction
        if split.train_fraction <= 0.0 or split.val_fraction <= 0.0:
            raise PipelineError(
                "Combined resplit modes require train_fraction > 0 and val_fraction > 0.",
                hint="Set split.train_fraction and split.val_fraction to positive values that sum with test_fraction to 1.0.",
            )
        if abs(total_fraction - 1.0) > 1e-6:
            raise PipelineError(
                f"Combined resplit fractions must sum to 1.0, got {total_fraction:.6f}.",
                hint="Adjust split.train_fraction, split.val_fraction, and split.test_fraction so their sum is exactly 1.0.",
            )
    return split


def _load_recipe(recipe_path: Path) -> PrepareRecipe:
    recipe_path = require_existing(recipe_path, "--recipe")
    payload = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise PipelineError(f"Prepare recipe must contain a YAML mapping: {recipe_path}")

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
        empty_policy=str(payload.get("empty_policy", "drop")).strip().lower(),
        split=_load_split_config(payload),
        keep_tokens=_coerce_selector_tokens(classes.get("keep")),
        drop_tokens=_coerce_selector_tokens(classes.get("drop")),
        remap_rules=remap_rules,
        raw=payload,
    )
    if recipe.empty_policy not in {"drop", "keep"}:
        raise PipelineError("Recipe field 'empty_policy' must be 'drop' or 'keep'")
    return recipe


def _recipe_requests_mutation(recipe: PrepareRecipe, *, dataset_dir: Path) -> bool:
    split = recipe.split
    if split.mode in COMBINED_SPLIT_MODES:
        return True
    if split.mode == "sample_existing" and (
        split.train_fraction != 1.0 or split.val_fraction != 1.0 or split.test_fraction != 1.0
    ):
        return True
    if recipe.keep_tokens or recipe.drop_tokens or recipe.remap_rules:
        return True
    if recipe.dataset_name and recipe.dataset_name != dataset_dir.name:
        return True
    return False


def _sample_paths(paths: Sequence[Path], fraction: float, seed: int) -> Set[Path]:
    if fraction <= 0.0:
        return set()
    if fraction >= 1.0 or not paths:
        return set(paths)
    selected_count = max(1, int(round(len(paths) * fraction)))
    randomizer = random.Random(seed)
    selected_indices = set(randomizer.sample(range(len(paths)), selected_count))
    return {path for index, path in enumerate(paths) if index in selected_indices}


def _parse_label_counts(label_path: Optional[Path]) -> Dict[int, int]:
    if label_path is None or not label_path.exists():
        return {}
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    counts: Dict[int, int] = {}
    for line in text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            class_id = int(float(parts[0]))
        except Exception:
            continue
        counts[class_id] = counts.get(class_id, 0) + 1
    return counts


def _collect_dataset_items(dataset_dir: Path, splits: Sequence[str]) -> CollectedDatasetItems:
    items: List[DatasetImageItem] = []
    dropped_orphan_labels = 0

    for split in splits:
        images_dir = require_existing(dataset_dir / "images" / split, f"images/{split}")
        labels_dir = dataset_dir / "labels" / split
        image_paths = sorted(iter_image_files(images_dir), key=lambda path: path.name)
        image_stems = {path.stem for path in image_paths}
        label_stems = {path.stem for path in labels_dir.glob("*.txt")} if labels_dir.exists() else set()
        dropped_orphan_labels += len(label_stems - image_stems)

        for image_path in image_paths:
            label_path = labels_dir / f"{image_path.stem}.txt" if labels_dir.exists() else None
            if label_path is not None and not label_path.exists():
                label_path = None
            relative_key = str(image_path.relative_to(images_dir).with_suffix(""))
            items.append(
                DatasetImageItem(
                    source_split=split,
                    image_path=image_path,
                    label_path=label_path,
                    name=image_path.name,
                    stem=image_path.stem,
                    relative_key=relative_key,
                    class_counts=_parse_label_counts(label_path),
                )
            )
    return CollectedDatasetItems(items=items, dropped_orphan_labels=dropped_orphan_labels)


def _resolve_existing_splits(dataset_dir: Path) -> List[str]:
    splits = detect_dataset_splits(dataset_dir)
    if "train" not in splits or "val" not in splits:
        raise PipelineError(
            f"Dataset directory must contain at least train and val splits: {dataset_dir}",
            hint="Run yolo-convert-dataset first or inspect the images/ and labels/ directory layout.",
        )
    return splits


def _remove_unselected_images(
    dataset_dir: Path,
    split: str,
    selected_images: Set[Path],
    progress_callback: ProgressCallback | None = None,
) -> Dict[str, int]:
    images_dir = require_existing(dataset_dir / "images" / split, f"images/{split}")
    labels_dir = require_existing(dataset_dir / "labels" / split, f"labels/{split}")
    all_images = sorted(iter_image_files(images_dir), key=lambda path: path.name)
    selected_names = {path.name for path in selected_images}
    removed_images = 0
    removed_labels = 0
    total_images = len(all_images)
    if progress_callback is not None:
        progress_callback(f"prepare:split:{split}:init", 0, total_images, f"prepare:split:{split}")

    for index, image_path in enumerate(all_images, start=1):
        if image_path.name in selected_names:
            if progress_callback is not None:
                progress_callback(f"prepare:split:{split}", index, total_images, f"prepare:split:{split}: {image_path.name}")
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        image_path.unlink(missing_ok=True)
        removed_images += 1
        if label_path.exists():
            label_path.unlink(missing_ok=True)
            removed_labels += 1
        if progress_callback is not None:
            progress_callback(f"prepare:split:{split}", index, total_images, f"prepare:split:{split}: {image_path.name}")

    return {
        "source_images": len(all_images),
        "kept_images": len(selected_images),
        "removed_images": removed_images,
        "removed_labels": removed_labels,
    }


def _sample_existing_splits(
    dataset_dir: Path,
    split_config: PrepareSplitConfig,
    splits: Sequence[str],
    progress_callback: ProgressCallback | None = None,
) -> Dict[str, Any]:
    fractions = {
        "train": split_config.train_fraction,
        "val": split_config.val_fraction,
        "test": split_config.test_fraction,
    }
    summary: Dict[str, Any] = {"mode": split_config.mode, "seed": split_config.seed, "splits": {}}
    for offset, split in enumerate(splits):
        fraction = fractions.get(split, 1.0)
        images_dir = require_existing(dataset_dir / "images" / split, f"images/{split}")
        all_images = sorted(iter_image_files(images_dir), key=lambda path: path.name)
        selected = _sample_paths(all_images, fraction, split_config.seed + offset)
        summary["splits"][split] = {
            "fraction": fraction,
            "seed": split_config.seed + offset,
            **_remove_unselected_images(dataset_dir, split, selected, progress_callback=progress_callback),
        }
    return summary


def _resolve_target_split_names(split_config: PrepareSplitConfig) -> List[str]:
    splits = ["train", "val"]
    if split_config.test_fraction > 0.0:
        splits.append("test")
    return splits


def _compute_target_split_sizes(total_items: int, split_config: PrepareSplitConfig) -> Dict[str, int]:
    fractions = {
        "train": split_config.train_fraction,
        "val": split_config.val_fraction,
        "test": split_config.test_fraction,
    }
    active_splits = _resolve_target_split_names(split_config)
    raw = {split: fractions[split] * total_items for split in active_splits}
    counts = {split: int(raw[split]) for split in active_splits}
    remainder = total_items - sum(counts.values())
    order = sorted(active_splits, key=lambda split: (-(raw[split] - counts[split]), split != "train", split))
    for split in order[:remainder]:
        counts[split] += 1
    return counts


def _resolve_per_class_minima(
    class_names: List[str],
    class_totals: Dict[int, int],
    default_minimum: int,
    per_class_overrides: Dict[str, int],
    *,
    field_name: str,
) -> Dict[int, int]:
    minima: Dict[int, int] = {
        class_id: default_minimum for class_id, total in class_totals.items() if total > 0 and default_minimum > 0
    }
    for selector, value in per_class_overrides.items():
        class_ids = parse_class_selectors([selector], class_names)
        for class_id in class_ids:
            if class_totals.get(class_id, 0) <= 0:
                continue
            minima[class_id] = max(minima.get(class_id, 0), value)
    impossible = [
        {"class_id": class_id, "class_name": class_names[class_id], "requested": minimum, "available": class_totals.get(class_id, 0)}
        for class_id, minimum in sorted(minima.items())
        if minimum > class_totals.get(class_id, 0)
    ]
    if impossible:
        preview = ", ".join(
            f"{item['class_name']} (requested={item['requested']}, available={item['available']})" for item in impossible[:6]
        )
        raise PipelineError(
            f"Unsatisfiable split constraints in {field_name}: {preview}",
            hint="Lower the per-class minimum instance targets or merge/drop very rare classes before resplitting.",
        )
    return {class_id: minimum for class_id, minimum in minima.items() if minimum > 0}


def _class_totals_from_items(items: Sequence[DatasetImageItem]) -> Dict[int, int]:
    totals: Dict[int, int] = {}
    for item in items:
        for class_id, count in item.class_counts.items():
            totals[class_id] = totals.get(class_id, 0) + count
    return totals


def _greedy_pick_item(items: Sequence[DatasetImageItem], deficits: Dict[int, int]) -> int | None:
    best_index: int | None = None
    best_gain = 0
    best_support = -1
    for index, item in enumerate(items):
        gain = 0
        support = 0
        for class_id, deficit in deficits.items():
            contribution = min(deficit, item.class_counts.get(class_id, 0))
            if contribution > 0:
                gain += contribution
                support += 1
        if gain > best_gain or (gain == best_gain and gain > 0 and support > best_support):
            best_index = index
            best_gain = gain
            best_support = support
    return best_index if best_gain > 0 else None


def _allocate_combined_by_instances(
    items: List[DatasetImageItem],
    class_names: List[str],
    split_config: PrepareSplitConfig,
) -> tuple[Dict[str, List[DatasetImageItem]], Dict[str, Any]]:
    randomizer = random.Random(split_config.seed)
    shuffled = list(items)
    randomizer.shuffle(shuffled)
    assignments: Dict[str, List[DatasetImageItem]] = {split: [] for split in _resolve_target_split_names(split_config)}
    targets = _compute_target_split_sizes(len(shuffled), split_config)
    class_totals = _class_totals_from_items(shuffled)
    constraint_targets = {
        "val": _resolve_per_class_minima(
            class_names,
            class_totals,
            split_config.min_val_instances_per_class,
            split_config.per_class_min_val_instances,
            field_name="split.per_class_min_val_instances",
        ),
        "test": _resolve_per_class_minima(
            class_names,
            class_totals,
            split_config.min_test_instances_per_class,
            split_config.per_class_min_test_instances,
            field_name="split.per_class_min_test_instances",
        ),
        "train": {},
    }

    remaining = shuffled
    unmet_summary: Dict[str, Any] = {}
    for split in [name for name in ("test", "val") if name in assignments and constraint_targets.get(name)]:
        deficits = dict(constraint_targets[split])
        while deficits:
            picked_index = _greedy_pick_item(remaining, deficits)
            if picked_index is None:
                preview = ", ".join(
                    f"{class_names[class_id]}: {value}" for class_id, value in list(deficits.items())[:6]
                )
                raise PipelineError(
                    f"Unable to satisfy {split} instance constraints: {preview}",
                    hint="Lower the requested minimum instances, relax the split fractions, or merge/drop rare classes first.",
                )
            picked = remaining.pop(picked_index)
            assignments[split].append(picked)
            updated: Dict[int, int] = {}
            for class_id, deficit in deficits.items():
                left = deficit - picked.class_counts.get(class_id, 0)
                if left > 0:
                    updated[class_id] = left
            deficits = updated
        unmet_summary[split] = {"min_instances_satisfied": True, "assigned_images": len(assignments[split])}

    randomizer.shuffle(remaining)
    for split in [name for name in ("val", "test") if name in assignments]:
        needed = max(0, targets[split] - len(assignments[split]))
        assignments[split].extend(remaining[:needed])
        del remaining[:needed]
    assignments["train"].extend(remaining)

    return assignments, {
        "mode": split_config.mode,
        "seed": split_config.seed,
        "target_images": targets,
        "constraint_state": unmet_summary,
    }


def _allocate_combined_random(
    items: List[DatasetImageItem],
    split_config: PrepareSplitConfig,
) -> tuple[Dict[str, List[DatasetImageItem]], Dict[str, Any]]:
    randomizer = random.Random(split_config.seed)
    shuffled = list(items)
    randomizer.shuffle(shuffled)
    targets = _compute_target_split_sizes(len(shuffled), split_config)
    assignments: Dict[str, List[DatasetImageItem]] = {split: [] for split in _resolve_target_split_names(split_config)}

    cursor = 0
    for split in [name for name in ("val", "test") if name in assignments]:
        next_cursor = cursor + targets[split]
        assignments[split].extend(shuffled[cursor:next_cursor])
        cursor = next_cursor
    assignments["train"].extend(shuffled[cursor:])
    return assignments, {"mode": split_config.mode, "seed": split_config.seed, "target_images": targets}


def _sanitize_name_token(value: str) -> str:
    token = normalize_name(value).replace('-', '_')
    return token or 'item'


def _build_hashed_name(item: DatasetImageItem, used_stems: Set[str], used_names: Set[str]) -> tuple[str, str]:
    suffix = item.image_path.suffix.lower() or item.image_path.suffix
    payload = f"{item.source_split}::{item.relative_key}::{item.name}".encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()
    candidate_stem = digest[:20]
    candidate_name = f"{candidate_stem}{suffix}"
    if candidate_stem not in used_stems and candidate_name not in used_names:
        return candidate_stem, candidate_name

    counter = 2
    while True:
        salted = hashlib.sha1(payload + f"::{counter}".encode("utf-8")).hexdigest()
        candidate_stem = salted[:20]
        candidate_name = f"{candidate_stem}{suffix}"
        if candidate_stem not in used_stems and candidate_name not in used_names:
            return candidate_stem, candidate_name
        counter += 1


def _rewrite_dataset_splits(
    dataset_dir: Path,
    source_splits: Sequence[str],
    assignments: Dict[str, List[DatasetImageItem]],
    progress_callback: ProgressCallback | None = None,
) -> Dict[str, Any]:
    temp_root = dataset_dir / ".prepare_split_tmp"
    remove_tree(temp_root)
    images_temp_root = temp_root / "images"
    labels_temp_root = temp_root / "labels"
    images_temp_root.mkdir(parents=True, exist_ok=True)
    labels_temp_root.mkdir(parents=True, exist_ok=True)

    active_splits = [split for split in DATASET_SPLIT_ORDER if split in assignments]
    for split in active_splits:
        (images_temp_root / split).mkdir(parents=True, exist_ok=True)
        (labels_temp_root / split).mkdir(parents=True, exist_ok=True)

    total_items = sum(len(items) for items in assignments.values())
    if progress_callback is not None:
        progress_callback("prepare:split:combined:init", 0, total_items, "prepare:split:combined")

    used_stems: Set[str] = set()
    used_names: Set[str] = set()
    renamed_items = 0
    moved = 0
    rename_samples: List[Dict[str, str]] = []
    for split in active_splits:
        for item in assignments[split]:
            target_stem, target_name = _build_hashed_name(item, used_stems, used_names)
            used_stems.add(target_stem)
            used_names.add(target_name)

            target_image = images_temp_root / split / target_name
            item.image_path.rename(target_image)
            if item.label_path is not None and item.label_path.exists():
                target_label = labels_temp_root / split / f"{target_stem}.txt"
                item.label_path.rename(target_label)

            if target_name != item.name:
                renamed_items += 1
                if len(rename_samples) < 10:
                    rename_samples.append({
                        "source_split": item.source_split,
                        "old_name": item.name,
                        "new_name": target_name,
                    })

            moved += 1
            if progress_callback is not None:
                progress_callback(
                    "prepare:split:combined",
                    moved,
                    total_items,
                    f"prepare:split:combined: {target_name}",
                )

    for split in source_splits:
        remove_tree(dataset_dir / "images" / split)
        remove_tree(dataset_dir / "labels" / split)

    (dataset_dir / "images").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels").mkdir(parents=True, exist_ok=True)
    for split in active_splits:
        (images_temp_root / split).rename(dataset_dir / "images" / split)
        (labels_temp_root / split).rename(dataset_dir / "labels" / split)
    remove_tree(temp_root)

    return {
        "active_splits": active_splits,
        "images_per_split": {split: len(assignments[split]) for split in active_splits},
        "renamed_items": renamed_items,
        "rename_samples": rename_samples,
    }


def _apply_split_strategy(
    dataset_dir: Path,
    split_config: PrepareSplitConfig,
    class_names: List[str],
    progress_callback: ProgressCallback | None = None,
) -> Dict[str, Any]:
    source_splits = _resolve_existing_splits(dataset_dir)

    if split_config.mode == "keep_existing":
        return {
            "mode": split_config.mode,
            "seed": split_config.seed,
            "source_splits": source_splits,
            "active_splits": source_splits,
            "images_per_split": {split: count_images(dataset_dir / "images" / split) for split in source_splits},
        }

    if split_config.mode == "sample_existing":
        sampling_summary = _sample_existing_splits(
            dataset_dir,
            split_config,
            source_splits,
            progress_callback=progress_callback,
        )
        sampling_summary["source_splits"] = source_splits
        sampling_summary["active_splits"] = source_splits
        sampling_summary["images_per_split"] = {
            split: count_images(dataset_dir / "images" / split) for split in source_splits
        }
        return sampling_summary

    collected = _collect_dataset_items(dataset_dir, source_splits)
    if not collected.items:
        raise PipelineError(
            f"No images found across existing dataset splits in {dataset_dir}",
            hint="Inspect images/train and images/val before running a combined resplit.",
        )

    if split_config.mode == "resplit_combined_random":
        assignments, split_summary = _allocate_combined_random(collected.items, split_config)
    else:
        assignments, split_summary = _allocate_combined_by_instances(collected.items, class_names, split_config)

    rewritten = _rewrite_dataset_splits(
        dataset_dir,
        source_splits,
        assignments,
        progress_callback=progress_callback,
    )
    split_summary.update(rewritten)
    split_summary["source_splits"] = source_splits
    split_summary["dropped_orphan_labels"] = collected.dropped_orphan_labels
    return split_summary


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
    splits: Sequence[str],
    old_to_new: Dict[int, int],
    empty_policy: str,
    progress_callback: ProgressCallback | None = None,
) -> Dict[str, Any]:
    if empty_policy not in {"drop", "keep"}:
        raise PipelineError(f"Unsupported empty_policy='{empty_policy}'")

    drop_empty = empty_policy == "drop"
    summary: Dict[str, Any] = {}
    for split in splits:
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


def _ensure_required_splits_non_empty(dataset_dir: Path, splits: Sequence[str], *, stage_name: str) -> None:
    for required_split in ("train", "val"):
        if required_split not in splits:
            raise PipelineError(
                f"After {stage_name}, required split '{required_split}' is missing.",
                hint="Keep both train and val in the split recipe. test is optional, train and val are required.",
            )
        if count_images(dataset_dir / "images" / required_split) == 0:
            raise PipelineError(
                f"After {stage_name}, split '{required_split}' became empty.",
                hint="Relax the split fractions, lower per-class constraints, or reconvert the dataset.",
            )


def prepare_yolo_dataset(
    options: PrepareYoloDatasetOptions,
    logger: Optional[LogFn] = None,
    progress_callback: ProgressCallback | None = None,
) -> Dict[str, Any]:
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
            hint="Change split.mode/fractions, add keep/drop/remap rules, or change dataset_name.",
        )

    log(format_info(f"Preparing YOLO dataset in place: {dataset_dir}"))
    log(format_info(f"Using recipe file: {options.recipe_path.resolve()}"))

    split_summary = _apply_split_strategy(
        dataset_dir,
        recipe.split,
        class_names,
        progress_callback=progress_callback,
    )
    active_splits = split_summary["active_splits"]
    _ensure_required_splits_non_empty(dataset_dir, active_splits, stage_name="split preparation")
    if split_summary.get("dropped_orphan_labels", 0) > 0:
        log(
            format_warning(
                f"Dropped orphan labels during combined resplit: {split_summary['dropped_orphan_labels']}"
            )
        )
    if split_summary.get("renamed_items", 0) > 0:
        log(
            format_info(
                f"Renamed image/label pairs during combined resplit: {split_summary['renamed_items']}"
            )
        )

    new_class_names, old_to_new, remap_plan = _build_remap_plan(class_names, recipe)
    rewrite_summary = _apply_remap_in_place(
        dataset_dir,
        active_splits,
        old_to_new,
        recipe.empty_policy,
        progress_callback=progress_callback,
    )
    _ensure_required_splits_non_empty(dataset_dir, active_splits, stage_name="class filtering/remapping")

    dataset_name = recipe.dataset_name or dataset_dir.name
    if progress_callback is not None:
        progress_callback("prepare:finalize:init", 0, 4, "prepare:finalize")
    _remove_stale_dataset_yamls(dataset_dir, options.recipe_path)
    if progress_callback is not None:
        progress_callback("prepare:finalize", 1, 4, "prepare:finalize: remove stale yaml")
    classes_txt = write_classes_txt(dataset_dir, new_class_names)
    if progress_callback is not None:
        progress_callback("prepare:finalize", 2, 4, "prepare:finalize: write classes.txt")
    data_yaml = write_dataset_yaml(dataset_dir, new_class_names, dataset_name, split_names=active_splits)
    if progress_callback is not None:
        progress_callback("prepare:finalize", 3, 4, "prepare:finalize: write data.yaml")

    prepare_report = {
        "dataset_dir": str(dataset_dir),
        "dataset_name": dataset_name,
        "recipe_path": str(options.recipe_path.resolve()),
        "split": split_summary,
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

    final_split_counts = {split: count_images(dataset_dir / "images" / split) for split in active_splits}
    log(format_info(f"Wrote classes.txt: {classes_txt}"))
    log(format_info(f"Wrote data.yaml: {data_yaml}"))
    log(format_info(f"Wrote prepare report: {prepare_report_path}"))
    log(
        format_info(
            "Final dataset state: "
            + f"classes={len(new_class_names)}, "
            + ", ".join(f"{split}_images={final_split_counts[split]}" for split in active_splits)
        )
    )
    log(format_info("Completed"))

    return {
        "dataset_dir": dataset_dir,
        "class_names": new_class_names,
        "data_yaml": data_yaml,
        "prepare_report_path": prepare_report_path,
        "split": split_summary,
        "class_transform": remap_plan,
        "rewrite_summary": rewrite_summary,
    }
