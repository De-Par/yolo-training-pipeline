from __future__ import annotations

import sys

from pathlib import Path


def bootstrap_project_root(file_path: str, *, levels: int) -> Path:
    project_root = Path(file_path).resolve().parents[levels]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return project_root
