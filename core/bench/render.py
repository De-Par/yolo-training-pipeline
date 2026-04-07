from __future__ import annotations

import csv
import json
import numpy as np

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Sequence
from core.bench.models import DatasetStats, SpeedPointResult
from core.bench.utils import json_safe


def render_report(
    cfg: Dict[str, Any],
    dataset_stats: DatasetStats,
    speed_results: Sequence[SpeedPointResult],
    quality: Dict[str, Any],
    class_names_for_counts: Sequence[str],
    speed_source_desc: str,
    quality_source_desc: str,
) -> Dict[str, str]:
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    output_dir = Path(cfg['output']['dir']).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / str(cfg['output'].get('report_png', 'benchmark_report.png'))
    out_json = output_dir / str(cfg['output'].get('report_json', 'benchmark_report.json'))
    out_csv = output_dir / str(cfg['output'].get('speed_csv', 'benchmark_speed.csv'))

    quality_rows = list(quality.get('per_class_rows', []))
    quality_rows.sort(key=lambda item: (-float(item['map50_95']), str(item['class_name'])))
    global_metrics = quality['global']

    fig_h = max(12, 6 + 0.28 * max(0, len(quality_rows)))
    fig = plt.figure(figsize=(18, fig_h), constrained_layout=True)
    fig.patch.set_facecolor('#f7f7f5')
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.0, 2.4], width_ratios=[1.0, 1.0])

    labels = [result.label for result in speed_results]
    x = np.arange(len(speed_results))
    latency_values = np.array([result.mean_ms for result in speed_results], dtype=float)
    latency_err = np.array([result.std_ms for result in speed_results], dtype=float)
    fps_values = np.array([result.fps for result in speed_results], dtype=float)

    ax_ms = fig.add_subplot(gs[0, 0])
    ms_bars = ax_ms.bar(x, latency_values, color='#2a6f97', alpha=0.9)
    ax_ms.errorbar(x, latency_values, yerr=latency_err, fmt='none', ecolor='#c1121f', capsize=4, linewidth=1.4)
    ax_ms.set_title('Inference latency', fontsize=14, weight='bold')
    ax_ms.set_ylabel('ms / image')
    ax_ms.set_xticks(x)
    ax_ms.set_xticklabels(labels, rotation=15, ha='right')
    ax_ms.grid(True, axis='y', alpha=0.25)
    for bar, value in zip(ms_bars, latency_values):
        ax_ms.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.1f}', ha='center', va='bottom', fontsize=9)

    ax_fps = fig.add_subplot(gs[0, 1])
    fps_bars = ax_fps.bar(x, fps_values, color='#90be6d', alpha=0.95)
    ax_fps.set_title('Throughput', fontsize=14, weight='bold')
    ax_fps.set_ylabel('FPS')
    ax_fps.set_xticks(x)
    ax_fps.set_xticklabels(labels, rotation=15, ha='right')
    ax_fps.grid(True, axis='y', alpha=0.25)
    for bar, value in zip(fps_bars, fps_values):
        ax_fps.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom', fontsize=9)

    ax_quality = fig.add_subplot(gs[1, 0])
    if quality_rows:
        q_names = [row['class_name'] for row in quality_rows]
        q_values = np.array([float(row['map50_95']) for row in quality_rows], dtype=float)
        q_counts = [int(row['instances']) for row in quality_rows]
        y_pos = np.arange(len(quality_rows))
        q_bars = ax_quality.barh(y_pos, q_values, color='#bc4749', alpha=0.9)
        ax_quality.set_yticks(y_pos)
        ax_quality.set_yticklabels(q_names)
        ax_quality.invert_yaxis()
        ax_quality.set_xlim(0.0, max(0.05, float(np.max(q_values)) * 1.15 if len(q_values) else 0.05))
        ax_quality.set_xlabel('mAP50-95')
        ax_quality.set_title('Per-class quality (mAP50-95, instances)', fontsize=14, weight='bold')
        ax_quality.grid(True, axis='x', alpha=0.25)
        for bar, value, count in zip(q_bars, q_values, q_counts):
            ax_quality.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f'{value:.3f}  ({count} inst)', va='center', ha='left', fontsize=8)
    else:
        ax_quality.axis('off')
        ax_quality.text(0.02, 0.95, 'Per-class quality unavailable', va='top', ha='left')

    ax_summary = fig.add_subplot(gs[1, 1])
    ax_summary.axis('off')
    class_preview = []
    for class_id, count in sorted(dataset_stats.class_counts.items()):
        class_name = class_names_for_counts[class_id] if 0 <= class_id < len(class_names_for_counts) else str(class_id)
        class_preview.append(f'{class_name}: {count}')
    counts_text = '\n'.join(class_preview[:10]) + ('\n...' if len(class_preview) > 10 else '')
    summary_lines = [
        'Benchmark summary',
        '',
        f"Model: {Path(cfg['model']).resolve().name}",
        f"Backend: {cfg['model_format']}",
        f"Hardware: {cfg['hardware']['kind']}   device={cfg['hardware'].get('device', 'cpu')}",
        f"Batch: {cfg['benchmark'].get('batch', 1)}",
        f"Warmup iters: {cfg['benchmark'].get('warmup_iters', 5)}",
        f"Max images / point: {cfg['benchmark'].get('max_images', 0)}",
        f"Speed source: {speed_source_desc}",
        f"Quality source: {quality_source_desc}",
        '',
        f"Average original size: {dataset_stats.avg_orig_h:.1f} x {dataset_stats.avg_orig_w:.1f}",
        f"Average effective size: {dataset_stats.avg_effective_h:.1f} x {dataset_stats.avg_effective_w:.1f}",
        f"Timed images available: {len(dataset_stats.image_paths)}",
        '',
        f"All classes mAP50-95: {global_metrics['map50_95']:.4f}",
        f"All classes mAP50: {global_metrics['map50']:.4f}",
        f"All classes mAP75: {global_metrics['map75']:.4f}",
    ]
    if global_metrics.get('precision') is not None:
        summary_lines.append(f"Precision: {global_metrics['precision']:.4f}")
    if global_metrics.get('recall') is not None:
        summary_lines.append(f"Recall: {global_metrics['recall']:.4f}")
    summary_lines.extend(['', 'Per-class bars: mAP50-95, then instance count in parentheses.', 'Instances by class (preview):', counts_text if counts_text else 'n/a'])
    ax_summary.text(0.02, 0.98, '\n'.join(summary_lines), va='top', ha='left', family='monospace', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='#d0d0d0'))

    fig.savefig(out_png, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    with out_csv.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['label', 'mean_ms', 'std_ms', 'fps', 'num_images'])
        writer.writeheader()
        for result in speed_results:
            writer.writerow({'label': result.label, 'mean_ms': result.mean_ms, 'std_ms': result.std_ms, 'fps': result.fps, 'num_images': result.num_images})

    payload = {
        'report_png': str(out_png),
        'report_json': str(out_json),
        'speed_csv': str(out_csv),
        'speed_points': [asdict(result) for result in speed_results],
        'dataset': {
            'avg_orig_h': dataset_stats.avg_orig_h,
            'avg_orig_w': dataset_stats.avg_orig_w,
            'avg_effective_h': dataset_stats.avg_effective_h,
            'avg_effective_w': dataset_stats.avg_effective_w,
            'class_counts': dataset_stats.class_counts,
            'num_images': len(dataset_stats.image_paths),
        },
        'quality': quality,
    }
    out_json.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding='utf-8')
    return {'report_png': str(out_png), 'report_json': str(out_json), 'speed_csv': str(out_csv)}
