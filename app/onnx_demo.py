#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from tqdm import tqdm

from core.common import format_info, format_warning, run_cli
from core.onnx.infer import collect_input_images, create_onnx_detector, draw_detections, infer_image


def _print_image_report(index: int, total: int, result) -> None:
    separator = '-' * 72
    print()
    print(separator)
    print(format_info(f'Image [{index}/{total}]'))
    print(format_info(f'Path: {result.image_path}'))
    print(format_info(f'Original size: {result.orig_hw[1]}x{result.orig_hw[0]}'))
    print(format_info(f'Model input: {result.input_hw[1]}x{result.input_hw[0]}'))
    print(format_info(f'Output format: {result.output_format}'))
    print(format_info(f'NMS applied: {"yes" if result.nms_applied else "no"}'))
    print(format_info(f'Detections: {len(result.detections)}'))
    print(format_info(f'Inference: {result.inference_ms:.2f} ms'))
    if result.detections:
        print(format_info('Top detections:'))
        for det in result.detections[:8]:
            print(format_info(f'  - {det.class_name}: score={det.score:.2f}, box={det.xyxy}'))
    else:
        print(format_info('Top detections: none'))
    print(separator)


def _parse_imgsz(values: list[str] | None) -> tuple[int, int] | None:
    if not values:
        return None
    if len(values) == 1:
        size = int(values[0])
        if size <= 0:
            raise ValueError
        return size, size
    if len(values) == 2:
        h, w = int(values[0]), int(values[1])
        if h <= 0 or w <= 0:
            raise ValueError
        return h, w
    raise ValueError


def _read_image_for_render(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f'Failed to read image: {image_path}')
    return image_bgr


def _timing_summary(times_ms: list[float]) -> dict[str, float]:
    if not times_ms:
        return {
            'count': 0.0,
            'sum_ms': 0.0,
            'mean_ms': 0.0,
            'median_ms': 0.0,
            'p90_ms': 0.0,
            'p95_ms': 0.0,
            'p99_ms': 0.0,
            'min_ms': 0.0,
            'max_ms': 0.0,
        }

    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        'count': float(arr.size),
        'sum_ms': float(arr.sum()),
        'mean_ms': float(arr.mean()),
        'median_ms': float(np.median(arr)),
        'p90_ms': float(np.percentile(arr, 90)),
        'p95_ms': float(np.percentile(arr, 95)),
        'p99_ms': float(np.percentile(arr, 99)),
        'min_ms': float(arr.min()),
        'max_ms': float(arr.max()),
    }


def _print_run_summary(processed: int, total: int, times_ms: list[float], wall_seconds: float, interrupted: bool) -> None:
    summary = _timing_summary(times_ms)
    pure_inference_seconds = summary['sum_ms'] / 1000.0
    inference_fps = processed / pure_inference_seconds if pure_inference_seconds > 0 else 0.0
    wall_fps = processed / wall_seconds if wall_seconds > 0 else 0.0

    separator = '=' * 72
    print()
    print(separator)
    print(format_info('Run summary'))
    print(format_info(f'Processed: {processed}/{total}'))
    if interrupted:
        print(format_warning('Run interrupted by user.'))

    print(format_info(f'Wall time: {wall_seconds:.3f} s'))
    print(format_info(f'Pure inference total: {summary["sum_ms"]:.2f} ms'))
    print(format_info(f'Pure inference mean: {summary["mean_ms"]:.2f} ms'))
    print(format_info(f'Pure inference median: {summary["median_ms"]:.2f} ms'))
    print(format_info(f'Pure inference p90: {summary["p90_ms"]:.2f} ms'))
    print(format_info(f'Pure inference p95: {summary["p95_ms"]:.2f} ms'))
    print(format_info(f'Pure inference p99: {summary["p99_ms"]:.2f} ms'))
    print(format_info(f'Pure inference min/max: {summary["min_ms"]:.2f} / {summary["max_ms"]:.2f} ms'))
    print(format_info(f'Pure inference FPS: {inference_fps:.2f}'))
    print(format_info(f'End-to-end FPS (demo loop): {wall_fps:.2f}'))
    print(separator)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run an ONNX detection demo on one image or a directory of images.'
    )
    parser.add_argument('--model', type=Path, required=True, help='Path to the ONNX model file.')
    parser.add_argument('--source', type=Path, required=True, help='Path to one image or a directory with images.')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='ONNX Runtime device target.')
    parser.add_argument('--class-names-file', type=Path, default=None, help='Optional class names file, one class per line.')
    parser.add_argument('--imgsz', nargs='+', default=None, help='Optional input size override: N or H W. If omitted, the ONNX input shape is used.')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detections.')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold for non-end-to-end outputs.')
    parser.add_argument('--max-det', type=int, default=300, help='Maximum number of detections after filtering.')
    parser.add_argument('--window-name', type=str, default='YOLO ONNX Demo', help='OpenCV window title.')
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Enable manual browsing mode with W/S/Q controls and detailed per-image terminal logs.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        input_hw = _parse_imgsz(args.imgsz)
    except ValueError:
        raise SystemExit('Invalid --imgsz. Use one integer or two positive integers: --imgsz 640 or --imgsz 640 640.')

    if args.max_det <= 0:
        raise SystemExit('--max-det must be a positive integer.')
    if not (0.0 <= args.conf <= 1.0):
        raise SystemExit('--conf must be between 0 and 1.')
    if not (0.0 <= args.iou <= 1.0):
        raise SystemExit('--iou must be between 0 and 1.')

    def _run() -> None:
        detector = create_onnx_detector(
            args.model,
            device=args.device,
            class_names_file=args.class_names_file,
            input_hw=input_hw,
        )
        images = collect_input_images(args.source)

        print(format_info(f'Model: {detector.model_path}'))
        print(format_info(f'Providers: {detector.providers}'))
        if detector.class_names:
            print(format_info(f'Classes: {len(detector.class_names)}'))
        else:
            print(format_warning('Class names are not available. The demo will show numeric class ids.'))
        print(format_info(f'Images loaded: {len(images)}'))
        print(format_info(f'Mode: {"interactive" if args.interactive else "auto"}'))

        if args.interactive:
            print(format_info('Controls: W next, S previous, Q or Esc quit.'))
        else:
            print(format_info('Controls: Q or Esc quit early.'))

        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

        if args.interactive:
            index = 0
            while True:
                image_path = images[index]
                result = infer_image(
                    detector,
                    image_path,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                )
                image_bgr = _read_image_for_render(image_path)
                canvas = draw_detections(result, image_bgr)
                cv2.imshow(args.window_name, canvas)

                _print_image_report(index + 1, len(images), result)

                key = cv2.waitKey(0) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                if key in (ord('w'), ord('W')):
                    if index < len(images) - 1:
                        index += 1
                    else:
                        print(format_warning('Already at the last image.'))
                    continue
                if key in (ord('s'), ord('S')):
                    if index > 0:
                        index -= 1
                    else:
                        print(format_warning('Already at the first image.'))
                    continue
                print(format_warning('Unsupported key. Use W, S, Q, or Esc.'))

            cv2.destroyAllWindows()
            return None

        times_ms: list[float] = []
        processed = 0
        interrupted = False
        wall_t0 = time.perf_counter()

        progress = tqdm(images, desc='Infer', unit='img')

        try:
            for image_path in progress:
                result = infer_image(
                    detector,
                    image_path,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                )
                times_ms.append(result.inference_ms)
                processed += 1

                image_bgr = _read_image_for_render(image_path)
                canvas = draw_detections(result, image_bgr)
                cv2.imshow(args.window_name, canvas)

                progress.set_postfix(ms=f'{result.inference_ms:.2f}', det=len(result.detections))

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    interrupted = True
                    break

        except KeyboardInterrupt:
            interrupted = True
            print()
            print(format_warning('Interrupted by user (Ctrl+C). Showing summary for processed images...'))

        finally:
            wall_t1 = time.perf_counter()
            progress.close()
            cv2.destroyAllWindows()

        _print_run_summary(
            processed=processed,
            total=len(images),
            times_ms=times_ms,
            wall_seconds=wall_t1 - wall_t0,
            interrupted=interrupted,
        )
        return None

    run_cli(_run)


if __name__ == '__main__':
    main()
    