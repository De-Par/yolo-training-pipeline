from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.common import PipelineError, ProgressCallback

from .common import move_if_needed


__all__ = ["ExportConfig", "export_yolo_to_onnx"]


@dataclass(slots=True)
class ExportConfig:
    weights_path: Path
    output_path: Path | None = None
    imgsz: int | tuple[int, int] = 768
    batch: int = 1
    device: str = "cpu"
    dynamic: bool = False
    simplify: bool = True
    opset: int | None = None

    def validate(self) -> None:
        self.weights_path = self.weights_path.expanduser().resolve()
        if self.output_path is not None:
            self.output_path = self.output_path.expanduser().resolve()
        if not self.weights_path.exists():
            raise PipelineError(
                f"Weights not found: {self.weights_path}",
                hint="Point --weights to an existing YOLO .pt checkpoint before exporting to ONNX.",
            )
        if self.batch <= 0:
            raise PipelineError("Export batch size must be a positive integer.")
        if self.opset is not None and self.opset <= 0:
            raise PipelineError("--opset must be a positive integer when provided.")


def export_yolo_to_onnx(cfg: ExportConfig, *, progress_callback: ProgressCallback | None = None) -> Path:
    cfg.validate()

    if progress_callback is not None:
        progress_callback("onnx:export:init", 0, 3, "onnx:export: validate")
        progress_callback("onnx:export", 1, 3, "onnx:export: load model")

    from ultralytics import YOLO

    model = YOLO(str(cfg.weights_path))
    if progress_callback is not None:
        progress_callback("onnx:export", 2, 3, "onnx:export: export")

    exported_path = model.export(
        format="onnx",
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        dynamic=cfg.dynamic,
        simplify=cfg.simplify,
        opset=cfg.opset,
    )
    output_path = move_if_needed(Path(exported_path).resolve(), cfg.output_path)

    if progress_callback is not None:
        progress_callback("onnx:export", 3, 3, f"onnx:export: wrote {output_path.name}")

    return output_path
