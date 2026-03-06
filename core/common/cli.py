from __future__ import annotations

import sys

from collections.abc import Callable
from typing import TypeVar

from core.common.errors import PipelineError, format_pipeline_error
from core.common.progress import ProgressCallback, create_progress_reporter, write_console_line


T = TypeVar("T")
CliAction = Callable[[], T]
CliProgressAction = Callable[[ProgressCallback], T]


def stdout_logger(message: str) -> None:
    write_console_line(message, flush=True)


def exit_with_pipeline_error(exc: PipelineError) -> "NoReturn":
    print(format_pipeline_error(exc), file=sys.stderr, flush=True)
    raise SystemExit(1) from exc


def run_cli(action: CliAction[T]) -> T:
    try:
        return action()
    except PipelineError as exc:
        exit_with_pipeline_error(exc)


def run_cli_with_progress(
    *,
    desc: str,
    unit: str,
    action: CliProgressAction[T],
    enabled: bool | None = None,
) -> T:
    def _runner() -> T:
        with create_progress_reporter(desc=desc, unit=unit, enabled=enabled) as progress:
            return action(progress.callback)

    return run_cli(_runner)
