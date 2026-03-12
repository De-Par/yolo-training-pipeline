#!/usr/bin/env python3
from __future__ import annotations

import argparse

from pathlib import Path

from _runtime import bootstrap_project_root

bootstrap_project_root(__file__, levels=1)

from core.common import format_info, run_cli_with_progress
from core.datasets.stats import (
    collect_yolo_dataset_stats,
    render_class_table,
    render_dataset_summary,
    write_dataset_stats_json,
    write_dataset_stats_plot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print statistics and plots for a YOLO-styled dataset.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to a YOLO-styled dataset directory.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path. Defaults to <dataset-dir>/dataset_stats.json")
    parser.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="Optional base PNG output path. Writes one mosaic PNG per detected split, for example <stem>_train.png and <stem>_test.png.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = run_cli_with_progress(
        desc="collect stats",
        unit="step",
        action=lambda progress_callback: collect_yolo_dataset_stats(
            args.dataset_dir,
            progress_callback=progress_callback,
        ),
    )

    print(render_dataset_summary(stats), flush=True)
    print()
    print(render_class_table(stats), flush=True)

    output_json = args.output_json or (args.dataset_dir.resolve() / "dataset_stats.json")
    output_png = args.output_png or (args.dataset_dir.resolve() / "dataset_stats.png")
    write_dataset_stats_json(output_json, stats)
    output_pngs = run_cli_with_progress(
        desc="render plots",
        unit="plot",
        action=lambda progress_callback: write_dataset_stats_plot(
            output_png,
            stats,
            progress_callback=progress_callback,
        ),
    )
    print()
    print(format_info(f"Wrote dataset_stats.json: {output_json}"), flush=True)
    for split in stats["splits"]:
        print(format_info(f"Wrote dataset_stats_{split}.png: {output_pngs[split]}"), flush=True)


if __name__ == "__main__":
    main()
