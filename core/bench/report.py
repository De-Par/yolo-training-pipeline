from __future__ import annotations

import logging

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional
from core.common import ProgressCallback, ensure_local_mplconfigdir
from core.bench.config import validate_benchmark_config
from core.bench.data import (
    collect_dataset_stats,
    ensure_quality_eval_dataset,
    ensure_speed_source,
    list_images_from_split,
    parse_label_counts,
    resolve_class_names,
)
from core.bench.render import render_report
from core.bench.runtime import build_speed_results, run_quality_eval, run_worker

LOGGER = logging.getLogger(__name__)


def run_yolo_benchmark_report(
    config_path: Path,
    *,
    worker_index: Optional[int] = None,
    progress_callback: ProgressCallback | None = None,
    script_path: Optional[Path] = None,
) -> Dict[str, Any]:
    ensure_local_mplconfigdir()
    benchmark_config = validate_benchmark_config(config_path.resolve())
    cfg = benchmark_config.raw
    script_path = (script_path or Path(__file__).resolve()).resolve()

    if worker_index is not None:
        return run_worker(cfg, worker_index)

    if progress_callback is not None:
        progress_callback('benchmark:run:init', 0, 4, 'benchmark:run: collect dataset')

    speed_image_paths, speed_labels_dir, speed_image_root, speed_source_desc = ensure_speed_source(cfg)
    dataset_stats = collect_dataset_stats(speed_image_paths, cfg, speed_labels_dir, speed_image_root)
    LOGGER.info(
        "Resolved speed source | source=%s | image_root=%s | labels_dir=%s | num_images=%d",
        speed_source_desc,
        speed_image_root,
        speed_labels_dir,
        len(speed_image_paths),
    )

    quality_data_yaml, quality_split, quality_labels_dir, quality_image_root, quality_tmp_dir, quality_source_desc = ensure_quality_eval_dataset(cfg)
    try:
        quality_image_paths = list_images_from_split(quality_data_yaml, quality_split)
        dataset_stats.class_counts = parse_label_counts(quality_image_paths, labels_dir=quality_labels_dir, image_root=quality_image_root)
        class_names_for_counts = resolve_class_names(cfg, quality_data_yaml, quality_labels_dir)

        LOGGER.info(
            "Resolved quality source | source=%s | data_yaml=%s | split=%s | image_root=%s | labels_dir=%s | num_images=%d",
            quality_source_desc,
            quality_data_yaml,
            quality_split,
            quality_image_root,
            quality_labels_dir,
            len(quality_image_paths),
        )
        LOGGER.info(
            "Resolved quality class names (%d): %s",
            len(class_names_for_counts),
            class_names_for_counts,
        )
    finally:
        if quality_tmp_dir is not None:
            quality_tmp_dir.cleanup()

    if progress_callback is not None:
        progress_callback('benchmark:run', 1, 4, 'benchmark:run: quality eval')
    quality = run_quality_eval(cfg)

    if progress_callback is not None:
        progress_callback('benchmark:run', 2, 4, 'benchmark:run: speed points')
    speed_results = build_speed_results(cfg, script_path, benchmark_config.path)

    if progress_callback is not None:
        progress_callback('benchmark:run', 3, 4, 'benchmark:run: render report')
    artifacts = render_report(cfg, dataset_stats, speed_results, quality, class_names_for_counts, speed_source_desc, quality_source_desc)

    summary = {
        **artifacts,
        'quality_run_dir': quality.get('run_dir'),
        'global_metrics': quality['global'],
        'speed_points': [asdict(result) for result in speed_results],
    }

    if progress_callback is not None:
        progress_callback('benchmark:run', 4, 4, 'benchmark:run: done')
        
    return summary
