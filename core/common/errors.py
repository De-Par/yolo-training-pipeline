from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from core.common.logging import format_detail, format_error, format_hint


class PipelineError(RuntimeError):
    """Domain-level error for dataset/training pipeline operations"""
    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        details: str | Sequence[str] | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details = details


def _normalize_detail_lines(details: str | Sequence[str] | Mapping[str, Any] | None) -> list[str]:
    if details is None:
        return []
    if isinstance(details, str):
        text = details.strip()
        return [text] if text else []
    if isinstance(details, Mapping):
        lines: list[str] = []
        for key, value in details.items():
            lines.append(f"{key}: {value}")
        return lines
    lines = []
    for item in details:
        text = str(item).strip()
        if text:
            lines.append(text)
    return lines


def format_pipeline_error(exc: PipelineError) -> str:
    lines = [format_error(exc.message)]
    if exc.hint:
        lines.append(format_hint(exc.hint))
    for detail in _normalize_detail_lines(exc.details):
        lines.append(format_detail(detail))
    return "\n".join(lines)
