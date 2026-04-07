from __future__ import annotations

import cv2
import numpy as np

from pathlib import Path
from typing import Iterator
from core.common import PipelineError

try:
    import onnxruntime as ort  # type: ignore
    from onnxruntime.quantization import CalibrationDataReader  # type: ignore
except ImportError:
    ort = None

    class CalibrationDataReader:  # type: ignore[override]
        pass

__all__ = ["ImageCalibrationDataReader", "collect_image_paths", "letterbox", "preprocess_image"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int],
    color: tuple[int, int, int] = (114, 114, 114),
    scaleup: bool = True,
) -> np.ndarray:
    shape = image.shape[:2]
    new_h, new_w = new_shape
    r = min(new_h / shape[0], new_w / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    resized_w = int(round(shape[1] * r))
    resized_h = int(round(shape[0] * r))
    pad_w = new_w - resized_w
    pad_h = new_h - resized_h
    if shape[::-1] != (resized_w, resized_h):
        image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    top = int(round(pad_h / 2 - 0.1))
    bottom = int(round(pad_h / 2 + 0.1))
    left = int(round(pad_w / 2 - 0.1))
    right = int(round(pad_w / 2 + 0.1))
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def preprocess_image(image_path: Path, input_hw: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise PipelineError(f"Failed to read calibration image: {image_path}")
    image = letterbox(image, input_hw)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return np.expand_dims(image, axis=0)


def collect_image_paths(image_dir: Path, limit: int) -> list[Path]:
    if limit <= 0:
        raise PipelineError("--calib-size must be a positive integer.")
    paths = [p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    paths = sorted(paths)
    if not paths:
        raise PipelineError(
            f"No calibration images found in: {image_dir}",
            hint="Point --calib-dir to a directory containing representative JPEG or PNG calibration images.",
        )
    return paths[:limit]


class ImageCalibrationDataReader(CalibrationDataReader):
    def __init__(self, model_path: Path, image_paths: list[Path], input_hw: tuple[int, int]) -> None:
        if ort is None:
            raise PipelineError(
                "onnxruntime is required for calibration.",
                hint="Install an ONNX Runtime profile with: source scripts/setup_env.sh cpu or source scripts/setup_env.sh cuda.",
            )
        self.image_paths = image_paths
        self.input_hw = input_hw
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = session.get_inputs()[0].name
        self._iterator: Iterator[dict[str, np.ndarray]] | None = None

    def _iter_data(self) -> Iterator[dict[str, np.ndarray]]:
        for image_path in self.image_paths:
            yield {self.input_name: preprocess_image(image_path, self.input_hw)}

    def get_next(self):
        if self._iterator is None:
            self._iterator = self._iter_data()
        return next(self._iterator, None)

    def rewind(self):
        self._iterator = None
        