from __future__ import annotations

import sys

from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, Dict, Optional, TextIO

from tqdm import tqdm


ProgressCallback = Callable[[str, int, int, str], None]
_CURRENT_PROGRESS_REPORTER: ContextVar[Any | None] = ContextVar("current_progress_reporter", default=None)


def noop_progress_callback(stage: str, current: int, total: int, message: str) -> None:
    del stage, current, total, message


class NullProgressReporter:
    """Progress reporter with the same API that intentionally does nothing."""

    callback: ProgressCallback = staticmethod(noop_progress_callback)

    def write(self, message: str, *, file: TextIO | None = None, flush: bool = True) -> None:
        stream = file or sys.stdout
        print(message, file=stream, flush=flush)

    def close(self) -> None:
        return None

    def __enter__(self) -> "NullProgressReporter":
        self._token = _CURRENT_PROGRESS_REPORTER.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _CURRENT_PROGRESS_REPORTER.reset(self._token)
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
        self._desc = desc
        self._unit = unit
        self._leave = leave
        self._disabled = (not sys.stderr.isatty()) if enabled is None else (not enabled)
        self._bar: tqdm | None = None
        self._last_by_stage: Dict[str, int] = {}
        self._token = None

    def _ensure_bar(self, *, total: int = 0, desc: str | None = None) -> None:
        if self._disabled:
            return
        if self._bar is None:
            self._bar = tqdm(
                total=max(0, total),
                desc=desc or self._desc,
                unit=self._unit,
                dynamic_ncols=True,
                leave=self._leave,
                disable=False,
            )
            return
        if total:
            self._bar.reset(total=total)
        if desc:
            self._bar.set_description(desc)

    def callback(self, stage: str, current: int, total: int, message: str) -> None:
        if stage.endswith(":init"):
            base_stage = stage.removesuffix(":init")
            self._last_by_stage[base_stage] = 0
            self._ensure_bar(total=total, desc=message or self._desc)
            if self._bar is not None:
                self._bar.refresh()
            return

        self._ensure_bar(total=total if self._bar is None else 0, desc=message or None)
        last_value = self._last_by_stage.get(stage, 0)
        delta = max(0, current - last_value)
        if delta and self._bar is not None:
            self._bar.update(delta)
            self._last_by_stage[stage] = current
        if message and self._bar is not None:
            self._bar.set_description(message)

    def write(self, message: str, *, file: TextIO | None = None, flush: bool = True) -> None:
        stream = file or sys.stdout
        if self._bar is None or self._disabled:
            print(message, file=stream, flush=flush)
            return
        tqdm.write(message, file=stream, end="\n")
        if flush and hasattr(stream, "flush"):
            stream.flush()

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()

    def __enter__(self) -> "TqdmProgressReporter":
        self._token = _CURRENT_PROGRESS_REPORTER.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _CURRENT_PROGRESS_REPORTER.reset(self._token)
        self.close()


def write_console_line(message: str, *, file: TextIO | None = None, flush: bool = True) -> None:
    reporter = _CURRENT_PROGRESS_REPORTER.get()
    if reporter is None:
        stream = file or sys.stdout
        print(message, file=stream, flush=flush)
        return
    reporter.write(message, file=file, flush=flush)


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
