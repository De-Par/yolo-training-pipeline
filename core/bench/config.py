from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from core.common import PipelineError

from .models import BenchmarkConfig
from .utils import load_yaml

_SUPPORTED_IMGSZ_MODES = {"square", "rect", "dynamic"}
_SUPPORTED_MODEL_SUFFIXES = {".pt": "pt", ".onnx": "onnx"}
_SUPPORTED_HW_KINDS = {"cpu", "gpu"}


def _expand_core_token(token: Any) -> List[int]:
    if isinstance(token, int):
        if token < 0:
            raise PipelineError('CPU core ids must be non-negative integers.')
        return [token]

    value = str(token).strip()
    if not value:
        raise PipelineError('CPU core ids must not be empty.')
    if value.isdigit():
        return [int(value)]
    if '-' in value:
        start_s, end_s = value.split('-', 1)
        if not start_s.isdigit() or not end_s.isdigit():
            raise PipelineError(f"Invalid CPU core range: {value}", hint='Use ranges like 0-7 or explicit ids like 3.')
        start = int(start_s)
        end = int(end_s)
        if end < start:
            raise PipelineError(f"Invalid CPU core range: {value}", hint='Use ascending inclusive ranges like 0-7.')
        return list(range(start, end + 1))
    raise PipelineError(f"Invalid CPU core token: {value}", hint='Use integers or inclusive ranges like 0-7.')


def _parse_core_list(raw_cores: Any) -> List[int]:
    if raw_cores is None:
        return []
    if not isinstance(raw_cores, list):
        raise PipelineError('hardware.points[*].cores must be a list of core ids or ranges.')
    expanded: List[int] = []
    for token in raw_cores:
        expanded.extend(_expand_core_token(token))
    return sorted(set(expanded))


def validate_benchmark_config(path: Path) -> BenchmarkConfig:
    cfg = load_yaml(path)
    for section in ("hardware", "imgsz", "dataset", "benchmark", "quality", "output"):
        if section not in cfg or not isinstance(cfg[section], dict):
            raise PipelineError(f"Benchmark config requires a '{section}' mapping: {path}")

    if 'model' not in cfg:
        raise PipelineError(f"Benchmark config requires a 'model' field: {path}")

    model_raw = cfg['model']
    if not isinstance(model_raw, str) or not model_raw.strip():
        raise PipelineError(
            'Benchmark config requires model to point to an existing .pt or .onnx file.',
            hint='Set model: path/to/model.pt or model: path/to/model.onnx.',
        )

    hw_kind = str(cfg["hardware"].get("kind", "")).lower().strip()
    if hw_kind not in _SUPPORTED_HW_KINDS:
        raise PipelineError(
            f"Unsupported hardware.kind='{hw_kind}'",
            hint="Use hardware.kind: cpu or gpu.",
        )
    points = cfg["hardware"].get("points")
    if not isinstance(points, list) or not points:
        raise PipelineError("hardware.points must be a non-empty list")
    for index, point in enumerate(points):
        if not isinstance(point, dict):
            raise PipelineError(f"hardware.points[{index}] must be a mapping")
        if not str(point.get("label", "")).strip():
            raise PipelineError(f"hardware.points[{index}].label must be non-empty")
        cores = point.get("cores")
        if cores is not None:
            try:
                point['cores'] = _parse_core_list(cores)
            except PipelineError as exc:
                raise PipelineError(
                    f"hardware.points[{index}].cores is invalid",
                    hint=getattr(exc, 'hint', None) or 'Use a list like [0, 1, 2] or ranges like ["0-3", 6].',
                    details=str(exc),
                ) from exc

    imgsz_mode = str(cfg["imgsz"].get("mode", "")).lower().strip()
    if imgsz_mode not in _SUPPORTED_IMGSZ_MODES:
        raise PipelineError(
            f"Unsupported imgsz.mode='{imgsz_mode}'",
            hint="Use imgsz.mode: square, rect, or dynamic.",
        )
    if imgsz_mode == "square":
        value = cfg["imgsz"].get("value")
        if not isinstance(value, int) or value <= 0:
            raise PipelineError("imgsz.value must be a positive integer for square mode")
    elif imgsz_mode == "rect":
        value = cfg["imgsz"].get("value")
        if not isinstance(value, list) or len(value) != 2 or not all(isinstance(x, int) and x > 0 for x in value):
            raise PipelineError("imgsz.value must be [H, W] with positive integers for rect mode")
    if int(cfg["imgsz"].get("stride", 32)) <= 0:
        raise PipelineError("imgsz.stride must be > 0")

    model_path = Path(str(model_raw)).expanduser().resolve()
    model_format = _SUPPORTED_MODEL_SUFFIXES.get(model_path.suffix.lower())
    if model_format is None:
        raise PipelineError(
            f"Unsupported model file suffix: {model_path.suffix or '<none>'}",
            hint="Use a .pt checkpoint or an .onnx model file.",
        )
    if not model_path.exists():
        raise PipelineError(
            f"Model file does not exist: {model_path}",
            hint="Point model to an existing .pt or .onnx file.",
        )

    dataset_cfg = cfg["dataset"]
    source_cfg = dataset_cfg.get("source")
    if not isinstance(source_cfg, dict):
        raise PipelineError(
            "dataset.source must be a mapping",
            hint="Define one default benchmark source in dataset.source and use dataset.speed or dataset.quality only for explicit overrides.",
        )
    for name in ("speed", "quality"):
        if name in dataset_cfg and not isinstance(dataset_cfg[name], dict):
            raise PipelineError(f"dataset.{name} must be a mapping when provided")

    has_speed_override = "speed" in dataset_cfg
    has_quality_override = "quality" in dataset_cfg

    def _merge_dataset_source(
        base_cfg: Dict[str, Any],
        override_cfg: Dict[str, Any],
        *,
        reset_yaml_split_to: str | None = None,
    ) -> Dict[str, Any]:
        merged = dict(base_cfg)
        if any(key in override_cfg for key in ("data_yaml",)):
            merged.pop("images_dir", None)
            merged.pop("annotations_dir", None)
            merged.pop("labels_dir", None)
        if any(key in override_cfg for key in ("images_dir", "annotations_dir", "labels_dir")):
            merged.pop("data_yaml", None)
        merged.update(override_cfg)
        if reset_yaml_split_to is not None and override_cfg.get("data_yaml") and "split" not in override_cfg:
            merged["split"] = reset_yaml_split_to
        if merged.get("labels_dir") and not merged.get("annotations_dir"):
            merged["annotations_dir"] = merged["labels_dir"]
        merged.pop("labels_dir", None)
        return merged

    def _validate_dataset_source(source_name: str, source_cfg: Dict[str, Any], *, require_annotations: bool) -> None:
        data_yaml = source_cfg.get("data_yaml")
        images_dir = source_cfg.get("images_dir")
        annotations_dir = source_cfg.get("annotations_dir")
        if data_yaml and images_dir:
            raise PipelineError(
                f"dataset.{source_name} cannot define both data_yaml and images_dir",
                hint=f"Use dataset.source for the default benchmark source, then override either split only or the entire source in dataset.{source_name}.",
            )
        if not data_yaml and not images_dir:
            raise PipelineError(
                f"dataset.{source_name}.data_yaml or dataset.{source_name}.images_dir is required after source resolution",
                hint=f"Define dataset.source first, or fully override dataset.{source_name}.",
            )
        if require_annotations and images_dir and not annotations_dir:
            raise PipelineError(
                f"dataset.{source_name}.annotations_dir is required when dataset.{source_name}.images_dir is used",
                hint=f"Point dataset.{source_name}.annotations_dir to the YOLO label directory matching dataset.{source_name}.images_dir.",
            )
        if images_dir and not Path(str(images_dir)).expanduser().resolve().exists():
            raise PipelineError(f"dataset.{source_name}.images_dir does not exist: {images_dir}")
        if annotations_dir and not Path(str(annotations_dir)).expanduser().resolve().exists():
            raise PipelineError(f"dataset.{source_name}.annotations_dir does not exist: {annotations_dir}")
        if data_yaml and not Path(str(data_yaml)).expanduser().resolve().exists():
            raise PipelineError(f"dataset.{source_name}.data_yaml does not exist: {data_yaml}")

    dataset_cfg["speed"] = _merge_dataset_source(
        source_cfg,
        dataset_cfg.get("speed", {}),
        reset_yaml_split_to='test' if has_speed_override else None,
    )
    dataset_cfg["quality"] = _merge_dataset_source(
        source_cfg,
        dataset_cfg.get("quality", {}),
        reset_yaml_split_to='test' if has_quality_override else None,
    )

    _validate_dataset_source("speed", dataset_cfg["speed"], require_annotations=False)
    _validate_dataset_source("quality", dataset_cfg["quality"], require_annotations=True)

    if int(cfg["benchmark"].get("batch", 1)) <= 0:
        raise PipelineError("benchmark.batch must be > 0")
    if int(cfg["benchmark"].get("warmup_iters", 5)) < 0:
        raise PipelineError("benchmark.warmup_iters must be >= 0")
    if int(cfg["benchmark"].get("max_images", 0) or 0) < 0:
        raise PipelineError("benchmark.max_images must be >= 0")

    output_dir = Path(str(cfg["output"].get("dir", "runs/bench"))).expanduser().resolve()
    cfg["output"]["dir"] = str(output_dir)
    cfg["model"] = str(model_path)
    cfg["model_format"] = model_format
    for name in ("source", "speed", "quality"):
        current = cfg["dataset"].get(name)
        if not isinstance(current, dict):
            continue
        if current.get("data_yaml"):
            current["data_yaml"] = str(Path(str(current["data_yaml"])).expanduser().resolve())
        if current.get("images_dir"):
            current["images_dir"] = str(Path(str(current["images_dir"])).expanduser().resolve())
        annotations_dir = current.get("annotations_dir") or current.get("labels_dir")
        if annotations_dir:
            current["annotations_dir"] = str(Path(str(annotations_dir)).expanduser().resolve())
        current.pop("labels_dir", None)

    return BenchmarkConfig(raw=cfg, path=path.resolve())
