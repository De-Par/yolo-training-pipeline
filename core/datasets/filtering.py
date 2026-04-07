from __future__ import annotations

from typing import Dict, List, Optional, Set
from core.common import PipelineError


def normalize_name(value: str) -> str:
    return " ".join(value.strip().split()).casefold()


def parse_class_selectors(tokens: Optional[List[str]], class_names: List[str]) -> Set[int]:
    if not tokens:
        return set()

    name_to_id: Dict[str, int] = {}
    for index, name in enumerate(class_names):
        name_to_id.setdefault(normalize_name(name), index)

    out: Set[int] = set()
    for token in tokens:
        current = token.strip()
        if not current:
            continue
        if "-" in current:
            left, right = current.split("-", 1)
            if left.strip().isdigit() and right.strip().isdigit():
                start = int(left.strip())
                end = int(right.strip())
                if start > end:
                    start, end = end, start
                if start < 0 or end >= len(class_names):
                    raise PipelineError(f"Class id range out of bounds: {current} (0..{len(class_names) - 1})")
                out.update(range(start, end + 1))
                continue
        if current.isdigit():
            class_id = int(current)
            if not (0 <= class_id < len(class_names)):
                raise PipelineError(f"Class id out of bounds: {class_id} (0..{len(class_names) - 1})")
            out.add(class_id)
            continue

        key = normalize_name(current)
        if key not in name_to_id:
            raise PipelineError(f"Unknown class name selector: '{current}'")
        out.add(name_to_id[key])
    return out
