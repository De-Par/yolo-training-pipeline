from __future__ import annotations

import sys

from collections.abc import Callable
from typing import Dict, Optional

from tqdm import tqdm


ProgressCallback = Callable[[str, int, int, str], None]


def noop_progress_callback(stage: str, current: int, total: int, message: str) -> None:
    del stage, current, total, message


class NullProgressReporter:
    """Progress reporter with the same API that intentionally does nothing."""

    callback: ProgressCallback = staticmethod(noop_progress_callback)

    def close(self) -> None:
        return None

    def __enter__(self) -> "NullProgressReporter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class TqdmProgressReporter:
    """Reusable staged progress reporter compatible with core progress callbacks."""

    def __init__(
        self,
        desc: str,
        *,
        unit: str = "item",
        leave: bool = False,
        enabled: bool | None = None,
    ) -> None:
        self._bar = tqdm(
            total=0,
            desc=desc,
            unit=unit,
            dynamic_ncols=True,
            leave=leave,
            disable=(not sys.stderr.isatty()) if enabled is None else (not enabled),
        )
        self._last_by_stage: Dict[str, int] = {}

    def callback(self, stage: str, current: int, total: int, message: str) -> None:
        if stage.endswith(":init"):
            base_stage = stage.removesuffix(":init")
            self._last_by_stage[base_stage] = 0
            self._bar.reset(total=total)
            if message:
                self._bar.set_description(message)
            self._bar.refresh()
            return

        last_value = self._last_by_stage.get(stage, 0)
        delta = max(0, current - last_value)
        if delta:
            self._bar.update(delta)
            self._last_by_stage[stage] = current
        if message:
            self._bar.set_description(message)

    def close(self) -> None:
        self._bar.close()

    def __enter__(self) -> "TqdmProgressReporter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def create_progress_reporter(
    *,
    desc: str,
    unit: str = "item",
    leave: bool = False,
    enabled: Optional[bool] = None,
) -> TqdmProgressReporter | NullProgressReporter:
    if enabled is False:
        return NullProgressReporter()
    return TqdmProgressReporter(desc=desc, unit=unit, leave=leave, enabled=enabled)
