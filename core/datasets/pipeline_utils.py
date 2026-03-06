from __future__ import annotations

import os

from pathlib import Path
from typing import Optional

from core.common import PipelineError, format_warning


def clamp_fraction(value: float, name: str) -> float:
    if not (0.0 < value <= 1.0):
        raise PipelineError(
            f"{name} must be in (0, 1], got {value}",
            hint=f"Use a value like 1.0 for the full split or 0.1 for a 10% sample.",
        )
    return value


def clamp_fraction_allow_zero(value: float, name: str) -> float:
    if not (0.0 <= value <= 1.0):
        raise PipelineError(
            f"{name} must be in [0, 1], got {value}",
            hint="Use 0.0 to disable the split, 1.0 for the full split, or a fraction like 0.1 for a 10% split.",
        )
    return value


def is_wsl() -> bool:
    try:
        return "WSL_DISTRO_NAME" in os.environ or "microsoft" in Path("/proc/version").read_text().lower()
    except Exception:
        return False


def slow_path_warning(path: Path, label: str) -> list[str]:
    try:
        resolved = str(path.resolve())
    except Exception:
        resolved = str(path)
    if is_wsl() and (resolved.startswith("/mnt/c/") or resolved.startswith("/mnt/C/")):
        return [
            format_warning(f"{label} is on /mnt/c (NTFS), which is often slow in WSL: {resolved}"),
            format_warning("Prefer a Linux filesystem (~/...) or an SSD-backed mount for better performance."),
        ]
    return []


def require_existing(path: Optional[Path], label: str) -> Path:
    if path is None:
        raise PipelineError(
            f"{label} is required",
            hint=f"Provide --{label.replace(' ', '-').replace('_', '-').lower()} or set it in the config/recipe if supported.",
        )
    if not path.exists():
        raise PipelineError(
            f"{label} does not exist: {path}",
            hint="Check the path, current working directory, and whether the previous pipeline stage completed successfully.",
        )
    return path
