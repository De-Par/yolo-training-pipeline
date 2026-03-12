#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run an interactive ONNX detection demo on one image or a directory of images.')
    parser.add_argument('--model', type=Path, required=True, help='Path to the ONNX model file.')
    parser.add_argument('--source', type=Path, required=True, help='Path to one image or a directory with images.')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='ONNX Runtime device target.')
    parser.add_argument('--class-names-file', type=Path, default=None, help='Optional class names file, one class per line.')
    parser.add_argument('--imgsz', nargs='+', default=None, help='Optional input size override: N or H W. If omitted, the ONNX input shape is used.')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detections.')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold.')
    parser.add_argument('--max-det', type=int, default=300, help='Maximum number of detections after NMS.')
    parser.add_argument('--window-name', type=str, default='YOLO ONNX Demo', help='OpenCV window title.')
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
        print(format_info('Controls: W next, S previous, Q or Esc quit.'))

        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        index = 0

        while True:
            image_path = images[index]
            result = infer_image(detector, image_path, conf=args.conf, iou=args.iou, max_det=args.max_det)
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f'Failed to read image: {image_path}')
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

    run_cli(_run)


if __name__ == '__main__':
    main()
