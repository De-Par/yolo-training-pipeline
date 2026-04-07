from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import numpy as np

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from tqdm import tqdm
from core.common import PipelineError
from core.bench.data import (
    build_quality_rows,
    ensure_quality_eval_dataset,
    ensure_speed_source,
    list_images_from_split,
    parse_label_counts,
    preprocess_image_to_nchw,
    resolve_class_names,
    sample_benchmark_images,
    sample_quality_images,
)
from core.bench.models import SpeedPointResult
from core.bench.utils import get_any

LOGGER = logging.getLogger(__name__)


def maybe_set_affinity(cores: Optional[Sequence[int]]) -> None:
    if not cores:
        return
    core_set = {int(x) for x in cores}
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, core_set)
        return
    try:
        import psutil  # type: ignore
        psutil.Process().cpu_affinity(sorted(core_set))
    except Exception:
        return


def run_shell(cmd: Optional[str]) -> None:
    if not cmd:
        return
    subprocess.run(cmd, shell=True, check=True)


def make_benchmark_bar(*, total: int, desc: str) -> tqdm:
    bar = tqdm(
        total=max(0, total),
        desc=desc,
        unit='step',
        dynamic_ncols=True,
        leave=True,
        disable=not sys.stderr.isatty(),
    )
    bar.set_postfix_str('warmup', refresh=False)
    return bar


def build_batches(images: Sequence[Path], cfg: Dict[str, Any], *, batch_size: int, force_float32: bool) -> List[np.ndarray]:
    batches: List[np.ndarray] = []
    for index in range(0, len(images), batch_size):
        chunk = images[index:index + batch_size]
        arrays = []
        for image_path in chunk:
            arr, _orig, _eff = preprocess_image_to_nchw(image_path, cfg)
            if force_float32:
                arr = arr.astype(np.float32, copy=False)
            arrays.append(arr)
        batches.append(np.stack(arrays, axis=0))
    return batches


def run_benchmark_batches(
    *,
    point_label: str,
    warmup_iters: int,
    warmup_batch: np.ndarray,
    batches: Sequence[np.ndarray],
    forward_once: Any,
) -> List[float]:
    total_steps = warmup_iters + len(batches)
    latencies_ms: List[float] = []
    with make_benchmark_bar(total=total_steps, desc=point_label) as bar:
        for warmup_index in range(warmup_iters):
            forward_once(warmup_batch)
            bar.update(1)
            bar.set_postfix_str(f'warmup {warmup_index + 1}/{warmup_iters}', refresh=False)

        bench_total = len(batches)
        bar.set_postfix_str(f'bench 0/{bench_total}', refresh=False)
        for bench_index, batch_np in enumerate(batches, start=1):
            t0 = time.perf_counter()
            forward_once(batch_np)
            t1 = time.perf_counter()
            dt_ms = (t1 - t0) * 1000.0 / batch_np.shape[0]
            latencies_ms.extend([dt_ms] * batch_np.shape[0])
            bar.update(1)
            bar.set_postfix_str(f'bench {bench_index}/{bench_total}', refresh=False)
    return latencies_ms


def materialize_backend_output(output: Any) -> None:
    if output is None:
        return
    if isinstance(output, np.ndarray):
        _ = output.shape
        return
    if isinstance(output, dict):
        for value in output.values():
            materialize_backend_output(value)
        return
    if isinstance(output, (list, tuple)):
        for value in output:
            materialize_backend_output(value)
        return
    shape = getattr(output, 'shape', None)
    if shape is not None:
        _ = tuple(shape)


def pick_device_string(cfg: Dict[str, Any]) -> str:
    if cfg['hardware']['kind'] == 'cpu':
        return 'cpu'
    return str(cfg['hardware'].get('device', 'cuda:0'))


def benchmark_pt(images: Sequence[Path], cfg: Dict[str, Any], point_cfg: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    from ultralytics import YOLO

    device = pick_device_string(cfg)
    if cfg['hardware']['kind'] == 'cpu':
        threads = int(point_cfg.get('threads') or len(point_cfg.get('cores', [])) or 1)
        torch.set_num_threads(threads)
        if hasattr(torch, 'set_num_interop_threads'):
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass
        maybe_set_affinity(point_cfg.get('cores'))

    model = YOLO(str(cfg['model']), task=str(cfg['task']))
    net = model.model
    net.eval()

    use_half = bool(cfg['benchmark'].get('half', False)) and device != 'cpu'
    if device != 'cpu':
        net.to(device)
        if use_half:
            net.half()

    batch_size = int(cfg['benchmark'].get('batch', 1))
    warmup_iters = int(cfg['benchmark'].get('warmup_iters', 5))
    point_label = str(point_cfg.get('label', 'point'))

    batches = build_batches(images, cfg, batch_size=batch_size, force_float32=False)
    if not batches:
        raise PipelineError('Benchmark speed source resolved to zero images.')

    warmup_np, _orig, _eff = preprocess_image_to_nchw(images[0], cfg)
    warmup_batch = np.expand_dims(warmup_np, axis=0)

    def _forward_once(batch_np: np.ndarray) -> None:
        x = torch.from_numpy(batch_np)
        x = x.to(device)
        x = x.half() if use_half else x.float()
        with torch.inference_mode():
            output = net(x)
        if device != 'cpu':
            torch.cuda.synchronize()
        materialize_backend_output(output)

    latencies_ms = run_benchmark_batches(
        point_label=point_label,
        warmup_iters=warmup_iters,
        warmup_batch=warmup_batch,
        batches=batches,
        forward_once=_forward_once,
    )

    mean_ms = float(np.mean(latencies_ms))
    std_ms = float(np.std(latencies_ms, ddof=0))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return {
        'label': str(point_cfg['label']),
        'mean_ms': mean_ms,
        'std_ms': std_ms,
        'fps': fps,
        'num_images': len(latencies_ms),
    }


def benchmark_onnx(images: Sequence[Path], cfg: Dict[str, Any], point_cfg: Dict[str, Any]) -> Dict[str, Any]:
    import onnxruntime as ort

    so = ort.SessionOptions()
    if cfg['hardware']['kind'] == 'cpu':
        maybe_set_affinity(point_cfg.get('cores'))
        threads = int(point_cfg.get('threads') or len(point_cfg.get('cores', [])) or 1)
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        providers = ['CPUExecutionProvider']
    else:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    session = ort.InferenceSession(str(cfg['model']), sess_options=so, providers=providers)
    LOGGER.info(
        "ONNX Runtime session | model=%s | requested_providers=%s | actual_providers=%s",
        cfg["model"],
        providers,
        session.get_providers(),
    )

    input_name = session.get_inputs()[0].name
    batch_size = int(cfg['benchmark'].get('batch', 1))
    warmup_iters = int(cfg['benchmark'].get('warmup_iters', 5))
    point_label = str(point_cfg.get('label', 'point'))

    batches = build_batches(images, cfg, batch_size=batch_size, force_float32=True)
    if not batches:
        raise PipelineError('Benchmark speed source resolved to zero images.')

    warmup_np, _orig, _eff = preprocess_image_to_nchw(images[0], cfg)
    warmup_batch = np.expand_dims(warmup_np.astype(np.float32, copy=False), axis=0)

    def _forward_once(batch_np: np.ndarray) -> None:
        output = session.run(None, {input_name: batch_np})
        materialize_backend_output(output)

    latencies_ms = run_benchmark_batches(
        point_label=point_label,
        warmup_iters=warmup_iters,
        warmup_batch=warmup_batch,
        batches=batches,
        forward_once=_forward_once,
    )

    mean_ms = float(np.mean(latencies_ms))
    std_ms = float(np.std(latencies_ms, ddof=0))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return {
        'label': str(point_cfg['label']),
        'mean_ms': mean_ms,
        'std_ms': std_ms,
        'fps': fps,
        'num_images': len(latencies_ms),
        'providers': providers,
    }


def run_worker(cfg: Dict[str, Any], point_index: int) -> Dict[str, Any]:
    points = cfg['hardware']['points']
    point_cfg = points[point_index]
    run_shell(point_cfg.get('pre_cmd'))

    try:
        image_paths, _labels_dir, _image_root, _source_desc = ensure_speed_source(cfg)
        images = sample_benchmark_images(image_paths, cfg)
        model_format = str(cfg['model_format']).lower()

        if model_format == 'pt':
            return benchmark_pt(images, cfg, point_cfg)
        return benchmark_onnx(images, cfg, point_cfg)

    finally:
        run_shell(point_cfg.get('post_cmd'))


def run_quality_eval(cfg: Dict[str, Any]) -> Dict[str, Any]:
    from ultralytics import YOLO

    if str(cfg["task"]).lower() != "detect":
        raise PipelineError(
            f"Unsupported task for bench quality eval: {cfg['task']}",
            hint="The current bench quality implementation supports only detect task."
        )

    data_yaml, split, labels_dir, image_root, tmp_dir, _source_desc = ensure_quality_eval_dataset(cfg)
    try:
        LOGGER.info(
            "Quality eval dataset | data_yaml=%s | split=%s | image_root=%s | labels_dir=%s",
            data_yaml,
            split,
            image_root,
            labels_dir,
        )

        device = pick_device_string(cfg)
        imgsz_mode = str(cfg['imgsz']['mode']).lower()

        if imgsz_mode == 'square':
            imgsz = int(cfg['imgsz']['value'])
            rect = False
        elif imgsz_mode == 'rect':
            imgsz = tuple(int(x) for x in cfg['imgsz']['value'])
            rect = False
        else:
            imgsz = int(cfg['imgsz'].get('val_imgsz_fallback', 640))
            rect = True

        save_json = bool(cfg['quality'].get('save_json', False))
        plots = bool(cfg['quality'].get('plots', False))
        persist_quality_artifacts = save_json or plots
        quality_run_tmp: tempfile.TemporaryDirectory[str] | None = None

        if persist_quality_artifacts:
            quality_project = Path(cfg['output']['dir']).resolve()
            quality_name = 'quality_artifacts'
            run_dir = quality_project / quality_name
        else:
            quality_run_tmp = tempfile.TemporaryDirectory(prefix='yolo_benchmark_quality_')
            quality_project = Path(quality_run_tmp.name)
            quality_name = 'quality_eval'
            run_dir = None

        model = YOLO(str(cfg['model']), task=str(cfg['task']))
        try:
            metrics = model.val(
                data=str(data_yaml),
                split=split,
                imgsz=imgsz,
                rect=rect,
                batch=int(cfg['quality'].get('batch', 1)),
                device=device,
                workers=0,
                conf=0.001,
                iou=0.7,
                verbose=False,
                save_json=save_json,
                plots=plots,
                project=str(quality_project),
                name=quality_name,
            )

            results_dict = getattr(metrics, 'results_dict', {}) or {}
            class_names = resolve_class_names(cfg, data_yaml, labels_dir)
            LOGGER.info(
                "Quality eval class names (%d): %s",
                len(class_names),
                class_names,
            )

            per_class_maps = [float(x) for x in list(metrics.box.maps)[: len(class_names)]]
            quality_rows = build_quality_rows(
                class_names,
                parse_label_counts(list_images_from_split(data_yaml, split), labels_dir=labels_dir, image_root=image_root),
                per_class_maps,
            )

            precision = get_any(results_dict, ['metrics/precision(B)', 'metrics/precision'])
            recall = get_any(results_dict, ['metrics/recall(B)', 'metrics/recall'])
            return {
                'global': {
                    'precision': None if precision is None else float(precision),
                    'recall': None if recall is None else float(recall),
                    'map50': float(metrics.box.map50),
                    'map75': float(metrics.box.map75),
                    'map50_95': float(metrics.box.map),
                },
                'per_class_rows': quality_rows,
                'run_dir': None if run_dir is None else str(run_dir),
            }
        finally:
            if quality_run_tmp is not None:
                quality_run_tmp.cleanup()
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()


def build_speed_results(cfg: Dict[str, Any], script_path: Path, config_path: Path) -> List[SpeedPointResult]:
    results: List[SpeedPointResult] = []
    for index, _point in enumerate(cfg['hardware']['points']):
        with tempfile.NamedTemporaryFile(prefix='yolo_bench_worker_', suffix='.json', delete=False) as handle:
            result_path = Path(handle.name)
        try:
            cmd = [
                sys.executable,
                str(script_path),
                '--config',
                str(config_path.resolve()),
                '--internal-worker-index',
                str(index),
                '--internal-result-path',
                str(result_path),
            ]
            subprocess.run(cmd, check=True, text=True, stdout=None, stderr=None)
            data = json.loads(result_path.read_text(encoding='utf-8'))
        finally:
            result_path.unlink(missing_ok=True)
            
        results.append(
            SpeedPointResult(
                label=str(data['label']),
                mean_ms=float(data['mean_ms']),
                std_ms=float(data['std_ms']),
                fps=float(data['fps']),
                num_images=int(data.get('num_images', 0) or 0),
            )
        )
    return results
