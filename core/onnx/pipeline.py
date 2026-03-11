from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from core.common import ProgressCallback

from .exporter import ExportConfig, export_yolo_to_onnx
from .optimizer import OptimizeConfig, optimize_onnx


__all__ = ["PipelineConfig", "run_export_and_optimize"]


@dataclass(slots=True)
class PipelineConfig:
    export: ExportConfig
    optimize: OptimizeConfig


def run_export_and_optimize(cfg: PipelineConfig, *, progress_callback: ProgressCallback | None = None) -> dict[str, Path]:
    exported = export_yolo_to_onnx(cfg.export, progress_callback=progress_callback)
    optimize_cfg = replace(cfg.optimize, input_model=exported)
    artifacts = optimize_onnx(optimize_cfg, progress_callback=progress_callback)
    artifacts["exported"] = exported
    return artifacts
