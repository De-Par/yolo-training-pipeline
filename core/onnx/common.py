from __future__ import annotations

import shutil

from pathlib import Path
from typing import Any

from core.common import PipelineError


__all__ = [
    "build_name",
    "build_onnx_artifact_name",
    "ensure_dir",
    "ensure_target_provider_available",
    "get_graph_level",
    "get_providers",
    "move_if_needed",
    "parse_hw",
    "parse_imgsz",
    "require_onnx",
    "require_onnxruntime",
]


def _parse_positive_int(token: str, *, option_name: str) -> int:
    try:
        value = int(token)
    except ValueError as exc:
        raise PipelineError(
            f"{option_name} must contain integer values.",
            hint=f"Use one integer or two integers for {option_name}, for example: {option_name} 640 or {option_name} 640 640.",
        ) from exc
    if value <= 0:
        raise PipelineError(
            f"{option_name} values must be positive integers.",
            hint=f"Use one integer or two positive integers for {option_name}.",
        )
    return value


def parse_imgsz(values: list[str]) -> int | tuple[int, int]:
    if len(values) == 1:
        return _parse_positive_int(values[0], option_name="--imgsz")
    if len(values) == 2:
        return (
            _parse_positive_int(values[0], option_name="--imgsz"),
            _parse_positive_int(values[1], option_name="--imgsz"),
        )
    raise PipelineError("--imgsz must contain 1 or 2 integer values.")


def parse_hw(values: list[str]) -> tuple[int, int]:
    imgsz = parse_imgsz(values)
    if isinstance(imgsz, int):
        return imgsz, imgsz
    return imgsz


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def move_if_needed(src: Path, dst: Path | None) -> Path:
    src = src.expanduser().resolve()
    if dst is None:
        return src
    dst = dst.expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src != dst:
        shutil.move(str(src), str(dst))
    return dst


def require_onnx():
    try:
        import onnx  # type: ignore
    except ImportError as exc:
        raise PipelineError(
            "onnx is not installed.",
            hint="Install project dependencies again with: source scripts/setup_env.sh base, cpu, or cuda.",
        ) from exc
    return onnx


def require_onnxruntime():
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError as exc:
        raise PipelineError(
            "onnxruntime is not installed.",
            hint="Install an ONNX Runtime profile with: source scripts/setup_env.sh cpu or source scripts/setup_env.sh cuda.",
        ) from exc
    return ort


def get_providers(target: str) -> list[str]:
    if target == "cpu":
        return ["CPUExecutionProvider"]
    if target == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    raise PipelineError(f"Unsupported target: {target}", hint="Use --target cpu or --target cuda.")


def ensure_target_provider_available(target: str) -> None:
    ort = require_onnxruntime()
    wanted = get_providers(target)[0]
    available = set(ort.get_available_providers())
    if wanted not in available:
        raise PipelineError(
            f"{wanted} is not available in this ONNX Runtime build.",
            hint="Install the matching runtime profile or switch --target to a provider available in the current environment.",
            details={"available_providers": sorted(available)},
        )


def get_graph_level(name: str) -> Any:
    ort = require_onnxruntime()
    mapping = {
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise PipelineError(
            f"Unsupported graph optimization level: {name}",
            hint="Use --graph-level basic, extended, or all.",
        ) from exc


def build_name(stem: str, *parts: str, suffix: str = ".onnx") -> str:
    chunks = [stem] + [part for part in parts if part]
    return ".".join(chunks) + suffix


def build_onnx_artifact_name(
    stem: str,
    *,
    stage: str,
    precision: str | None = None,
    target: str | None = None,
    graph_level: str | None = None,
    variant: str | None = None,
    tag: str | None = None,
    suffix: str = ".onnx",
) -> str:
    chunks = [stem, stage]
    if target:
        chunks.append(target)
    if graph_level:
        chunks.append(graph_level)
    if precision:
        chunks.append(precision)
    if variant:
        chunks.append(variant)
    if tag:
        chunks.append(tag)
    return ".".join(chunks) + suffix
