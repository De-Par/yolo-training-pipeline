from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(slots=True)
class DatasetStats:
    image_paths: List[Path]
    avg_orig_h: float
    avg_orig_w: float
    avg_effective_h: float
    avg_effective_w: float
    class_counts: Dict[int, int]


@dataclass(slots=True)
class SpeedPointResult:
    label: str
    mean_ms: float
    std_ms: float
    fps: float
    num_images: int


@dataclass(slots=True)
class BenchmarkConfig:
    raw: Dict[str, Any]
    path: Path
    