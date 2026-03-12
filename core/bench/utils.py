from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from core.common import PipelineError


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise PipelineError(f"Benchmark config must be a YAML mapping: {path}")
    return payload


def get_any(data: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in data:
            return data.get(key)
    return None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
