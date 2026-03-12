#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from pathlib import Path

from _runtime import bootstrap_project_root

bootstrap_project_root(__file__, levels=1)

from core.bench import run_yolo_benchmark_report
from core.common import run_cli, stdout_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLO speed and quality from a YAML config and render a single PNG report.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the benchmark YAML config.")
    parser.add_argument("--internal-worker-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-result-path", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.internal_worker_index is not None:
        result = run_yolo_benchmark_report(args.config, worker_index=args.internal_worker_index, script_path=Path(__file__).resolve())
        payload = json.dumps(result, ensure_ascii=False)
        if args.internal_result_path is not None:
            args.internal_result_path.write_text(payload, encoding='utf-8')
        else:
            print(payload)
        return

    def _run():
        result = run_yolo_benchmark_report(
            args.config,
            progress_callback=None,
            script_path=Path(__file__).resolve(),
        )
        stdout_logger(f"[INFO] Wrote benchmark report PNG: {result['report_png']}")
        stdout_logger(f"[INFO] Wrote benchmark report JSON: {result['report_json']}")
        stdout_logger(f"[INFO] Wrote benchmark speed CSV: {result['speed_csv']}")
        if result.get('quality_run_dir'):
            stdout_logger(f"[INFO] Quality evaluation run: {result['quality_run_dir']}")
        return result

    run_cli(_run)


if __name__ == "__main__":
    main()
