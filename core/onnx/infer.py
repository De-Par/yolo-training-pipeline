from __future__ import annotations

import ast
import time
import cv2
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from core.common import PipelineError
from .common import ensure_target_provider_available, get_providers, require_onnxruntime

__all__ = [
    'Detection',
    'InferenceResult',
    'OnnxDetector',
    'collect_input_images',
    'create_onnx_detector',
    'draw_detections',
    'infer_image',
    'inspect_raw_outputs',
]

_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
_YOLO_MAX_STRIDE = 32
_FINAL_DETECTIONS_MAX_ROWS = 300


@dataclass(slots=True)
class Detection:
    class_id: int
    class_name: str
    score: float
    xyxy: tuple[int, int, int, int]


@dataclass(slots=True)
class InferenceResult:
    image_path: Path
    orig_hw: tuple[int, int]
    input_hw: tuple[int, int]
    inference_ms: float
    detections: list[Detection]
    output_format: str
    nms_applied: bool


@dataclass(slots=True)
class OnnxDetector:
    model_path: Path
    session: Any
    input_name: str
    providers: list[str]
    class_names: list[str]
    input_hw: tuple[int, int] | None
    dynamic_hw: bool


@dataclass(slots=True)
class _DecodedPredictions:
    boxes_xyxy: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    requires_nms: bool
    format_name: str


def collect_input_images(source: Path) -> list[Path]:
    source = source.expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in _IMAGE_EXTS:
            raise PipelineError(f'Unsupported image file: {source}')
        return [source]
    if source.is_dir():
        images = [path for path in sorted(source.rglob('*')) if path.suffix.lower() in _IMAGE_EXTS]
        if not images:
            raise PipelineError(f'No images found in directory: {source}')
        return images
    raise PipelineError(
        f'Input source does not exist: {source}',
        hint='Point --source to an image file or a directory with images.',
    )


def _load_class_names_from_file(path: Path | None) -> list[str]:
    if path is None:
        return []
    path = path.expanduser().resolve()
    if not path.exists():
        raise PipelineError(f'Class names file does not exist: {path}')
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _load_class_names_from_metadata(session: Any) -> list[str]:
    try:
        metadata = session.get_modelmeta().custom_metadata_map or {}
    except Exception:
        metadata = {}
    raw = metadata.get('names') or metadata.get('classes')
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return []
    if isinstance(parsed, dict):
        items = sorted(parsed.items(), key=lambda item: int(item[0]))
        return [str(value) for _key, value in items]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def _resolve_input_hw(session: Any, override_hw: tuple[int, int] | None) -> tuple[tuple[int, int] | None, bool]:
    if override_hw is not None:
        return override_hw, False

    shape = list(session.get_inputs()[0].shape)
    if len(shape) != 4:
        raise PipelineError(
            f'Unsupported ONNX input rank: {shape}',
            hint='This demo expects a detection model with NCHW input.',
        )

    h_raw = shape[2]
    w_raw = shape[3]
    dynamic_hw = not isinstance(h_raw, int) or not isinstance(w_raw, int)

    if dynamic_hw:
        return None, True

    return (int(h_raw), int(w_raw)), False


def create_onnx_detector(
    model_path: Path,
    *,
    device: str = 'cpu',
    class_names_file: Path | None = None,
    input_hw: tuple[int, int] | None = None,
) -> OnnxDetector:
    model_path = model_path.expanduser().resolve()
    if not model_path.exists():
        raise PipelineError(f'ONNX model does not exist: {model_path}')

    target = 'cuda' if str(device).lower().startswith('cuda') else 'cpu'
    ensure_target_provider_available(target)

    ort = require_onnxruntime()
    providers = get_providers(target)

    try:
        session = ort.InferenceSession(str(model_path), providers=providers)
    except Exception as exc:
        message = str(exc)
        hint = 'Re-export the ONNX model from a working checkpoint or validate the file with onnx.load/check_model.'
        if 'INVALID_PROTOBUF' in message or 'Protobuf parsing failed' in message:
            raise PipelineError(
                f'Failed to load ONNX model: {model_path}',
                hint='The file is not a valid ONNX protobuf. Re-export the model or replace the corrupted file.',
                details=message,
            ) from exc
        raise PipelineError(
            f'Failed to create ONNX Runtime session for model: {model_path}',
            hint=hint,
            details=message,
        ) from exc

    resolved_input_hw, dynamic_hw = _resolve_input_hw(session, input_hw)
    class_names = _load_class_names_from_file(class_names_file) or _load_class_names_from_metadata(session)

    return OnnxDetector(
        model_path=model_path,
        session=session,
        input_name=session.get_inputs()[0].name,
        providers=list(session.get_providers()),
        class_names=class_names,
        input_hw=resolved_input_hw,
        dynamic_hw=dynamic_hw,
    )


def _round_hw_to_stride(hw: tuple[int, int], stride: int = _YOLO_MAX_STRIDE) -> tuple[int, int]:
    h, w = hw
    return (
        ((h + stride - 1) // stride) * stride,
        ((w + stride - 1) // stride) * stride,
    )


def _resolve_runtime_hw(detector: OnnxDetector, image_hw: tuple[int, int]) -> tuple[int, int]:
    if detector.input_hw is not None:
        return _round_hw_to_stride(detector.input_hw)
    if detector.dynamic_hw:
        return _round_hw_to_stride(image_hw)
    return image_hw


def _letterbox(image_bgr: np.ndarray, target_hw: tuple[int, int]) -> tuple[np.ndarray, float, tuple[float, float]]:
    target_h, target_w = target_hw
    orig_h, orig_w = image_bgr.shape[:2]

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

    pad_x = (target_w - new_w) / 2.0
    pad_y = (target_h - new_h) / 2.0

    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    canvas[top:top + new_h, left:left + new_w] = resized

    return canvas, scale, (pad_x, pad_y)


def _to_nchw(image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    array = image_rgb.astype(np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0)


def _pick_detection_output(outputs: Sequence[np.ndarray]) -> np.ndarray:
    candidates: list[np.ndarray] = []

    for output in outputs:
        arr = np.asarray(output)
        if arr.ndim == 3:
            candidates.append(arr)
        elif arr.ndim == 2:
            candidates.append(arr[None, ...])

    if not candidates:
        raise PipelineError(
            'Could not find a 2D/3D detection output in ONNX results.',
            hint='This demo expects a detect ONNX model exported from the current YOLO pipeline.',
        )

    arr = candidates[0]

    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)

    if arr.ndim != 2:
        raise PipelineError(f'Unsupported detection output shape: {arr.shape}')

    if arr.shape[0] <= 256 and arr.shape[1] > arr.shape[0]:
        arr = arr.T

    if arr.shape[1] < 6:
        raise PipelineError(
            f'Unsupported detection output shape: {arr.shape}',
            hint='Expected rows shaped as [x, y, w, h, class_scores...] or [x1, y1, x2, y2, score, class_id].',
        )

    return arr.astype(np.float32, copy=False)


def _looks_like_final_detections(raw: np.ndarray, class_count: int) -> bool:
    if raw.ndim != 2 or raw.shape[1] != 6:
        return False

    if raw.shape[0] <= _FINAL_DETECTIONS_MAX_ROWS:
        return True

    scores = raw[:, 4]
    class_values = raw[:, 5]

    valid = np.isfinite(scores) & np.isfinite(class_values)
    if not np.any(valid):
        return False

    scores = scores[valid]
    class_values = class_values[valid]

    if scores.size == 0 or class_values.size == 0:
        return False

    score_lo = float(np.quantile(scores, 0.05))
    score_hi = float(np.quantile(scores, 0.95))
    if score_lo < -1e-3 or score_hi > 1.0 + 1e-3:
        return False

    integerish_fraction = float(np.mean(np.abs(class_values - np.round(class_values)) < 1e-3))
    if integerish_fraction < 0.9:
        return False

    if class_count > 0:
        if float(class_values.min()) < -1e-3:
            return False
        if float(class_values.max()) > class_count - 1 + 1e-3:
            return False

    return True


def _decode_predictions(raw: np.ndarray, class_names: Sequence[str]) -> _DecodedPredictions:
    class_count = len(class_names)

    if _looks_like_final_detections(raw, class_count):
        return _DecodedPredictions(
            boxes_xyxy=raw[:, :4].astype(np.float32, copy=False),
            scores=raw[:, 4].astype(np.float32, copy=False),
            class_ids=np.round(raw[:, 5]).astype(np.int32, copy=False),
            requires_nms=False,
            format_name='xyxy_score_class',
        )

    xywh = raw[:, :4]
    remaining = raw[:, 4:]

    if class_count > 0 and remaining.shape[1] == class_count + 1:
        objectness = remaining[:, 0]
        class_scores = remaining[:, 1:]
        class_ids = class_scores.argmax(axis=1)
        scores = objectness * class_scores[np.arange(class_scores.shape[0]), class_ids]
        format_name = 'xywh_obj_cls'
    else:
        class_ids = remaining.argmax(axis=1)
        scores = remaining[np.arange(remaining.shape[0]), class_ids]
        format_name = 'xywh_cls'

    boxes_xyxy = _xywh_to_xyxy(xywh)

    return _DecodedPredictions(
        boxes_xyxy=boxes_xyxy.astype(np.float32, copy=False),
        scores=scores.astype(np.float32, copy=False),
        class_ids=class_ids.astype(np.int32, copy=False),
        requires_nms=True,
        format_name=format_name,
    )


def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    xyxy = np.empty_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0
    return xyxy


def _scale_boxes(
    boxes_xyxy: np.ndarray,
    *,
    scale: float,
    pad: tuple[float, float],
    orig_hw: tuple[int, int],
) -> np.ndarray:
    pad_x, pad_y = pad
    orig_h, orig_w = orig_hw

    boxes = boxes_xyxy.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h - 1)

    return boxes


def _run_classwise_nms(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    *,
    conf: float,
    iou: float,
    max_det: int,
) -> list[int]:
    keep: list[int] = []

    for class_id in sorted(set(int(value) for value in class_ids.tolist())):
        indices = np.where((class_ids == class_id) & (scores >= conf))[0]
        if indices.size == 0:
            continue

        class_boxes = boxes_xyxy[indices]
        class_scores = scores[indices]
        nms_boxes = [
            [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]
            for x1, y1, x2, y2 in class_boxes
        ]

        selected = cv2.dnn.NMSBoxes(nms_boxes, class_scores.tolist(), conf, iou)
        if len(selected) == 0:
            continue

        selected_flat = np.array(selected).reshape(-1)
        keep.extend(indices[selected_flat].tolist())

    keep.sort(key=lambda index: float(scores[index]), reverse=True)
    return keep[:max_det]


def _select_confident_detections(scores: np.ndarray, *, conf: float, max_det: int) -> list[int]:
    indices = np.where(scores >= conf)[0]
    if indices.size == 0:
        return []

    ordered = indices[np.argsort(scores[indices])[::-1]]
    return ordered[:max_det].tolist()


def infer_image(
    detector: OnnxDetector,
    image_path: Path,
    *,
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 300,
) -> InferenceResult:
    image_path = image_path.expanduser().resolve()

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise PipelineError(f'Failed to read image: {image_path}')

    orig_h, orig_w = image_bgr.shape[:2]
    target_hw = _resolve_runtime_hw(detector, (orig_h, orig_w))
    prepared_bgr, scale, pad = _letterbox(image_bgr, target_hw)
    input_tensor = _to_nchw(prepared_bgr)

    t0 = time.perf_counter()
    outputs = detector.session.run(None, {detector.input_name: input_tensor})
    t1 = time.perf_counter()
    inference_ms = (t1 - t0) * 1000.0

    raw = _pick_detection_output(outputs)
    decoded = _decode_predictions(raw, detector.class_names)

    boxes_xyxy = _scale_boxes(decoded.boxes_xyxy, scale=scale, pad=pad, orig_hw=(orig_h, orig_w))
    if decoded.requires_nms:
        keep = _run_classwise_nms(
            boxes_xyxy,
            decoded.scores,
            decoded.class_ids,
            conf=conf,
            iou=iou,
            max_det=max_det,
        )
    else:
        keep = _select_confident_detections(decoded.scores, conf=conf, max_det=max_det)

    detections: list[Detection] = []
    for index in keep:
        class_id = int(decoded.class_ids[index])
        class_name = detector.class_names[class_id] if 0 <= class_id < len(detector.class_names) else str(class_id)
        x1, y1, x2, y2 = boxes_xyxy[index]
        detections.append(
            Detection(
                class_id=class_id,
                class_name=class_name,
                score=float(decoded.scores[index]),
                xyxy=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
            )
        )

    return InferenceResult(
        image_path=image_path,
        orig_hw=(orig_h, orig_w),
        input_hw=target_hw,
        inference_ms=inference_ms,
        detections=detections,
        output_format=decoded.format_name,
        nms_applied=decoded.requires_nms,
    )


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = [
        (56, 56, 255),
        (151, 157, 255),
        (31, 112, 255),
        (29, 178, 255),
        (49, 210, 207),
        (10, 249, 72),
        (23, 204, 146),
        (134, 219, 61),
        (52, 147, 26),
        (187, 212, 0),
        (168, 153, 44),
        (255, 194, 0),
        (147, 69, 52),
        (255, 115, 100),
        (236, 24, 0),
        (255, 56, 132),
        (133, 0, 82),
        (255, 56, 203),
        (200, 149, 255),
        (199, 55, 255),
    ]
    return palette[class_id % len(palette)]


def draw_detections(result: InferenceResult, image_bgr: np.ndarray) -> np.ndarray:
    canvas = image_bgr.copy()
    line_width = max(2, round((canvas.shape[0] + canvas.shape[1]) / 600))
    font_scale = max(0.5, line_width / 3.0)

    for det in result.detections:
        color = _color_for_class(det.class_id)
        x1, y1, x2, y2 = det.xyxy

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=line_width, lineType=cv2.LINE_AA)

        label = f'{det.class_name} {det.score:.2f}'
        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            max(1, line_width - 1),
        )

        text_top = max(0, y1 - text_h - baseline - 6)
        cv2.rectangle(
            canvas,
            (x1, text_top),
            (x1 + text_w + 6, text_top + text_h + baseline + 6),
            color,
            thickness=-1,
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 3, text_top + text_h + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(1, line_width - 1),
            cv2.LINE_AA,
        )

    return canvas


def _summarize_numeric_array(arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr)

    summary: dict[str, Any] = {
        'shape': tuple(int(dim) for dim in arr.shape),
        'dtype': str(arr.dtype),
        'size': int(arr.size),
    }

    if arr.size == 0:
        return summary

    if not np.issubdtype(arr.dtype, np.number):
        return summary

    arr_f32 = arr.astype(np.float32, copy=False)

    summary.update(
        {
            'min': float(arr_f32.min()),
            'max': float(arr_f32.max()),
            'mean': float(arr_f32.mean()),
            'std': float(arr_f32.std()),
            'nonzero': int(np.count_nonzero(arr_f32)),
            'gt_1e_6': int((arr_f32 > 1e-6).sum()),
            'gt_1e_5': int((arr_f32 > 1e-5).sum()),
            'gt_1e_4': int((arr_f32 > 1e-4).sum()),
            'gt_1e_3': int((arr_f32 > 1e-3).sum()),
            'gt_1e_2': int((arr_f32 > 1e-2).sum()),
            'lt_minus_1e_6': int((arr_f32 < -1e-6).sum()),
        }
    )
    return summary


def _summarize_decoded_scores(scores: np.ndarray) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float32)

    if scores.size == 0:
        return {
            'size': 0,
            'score_min': None,
            'score_max': None,
            'score_mean': None,
            'score_gt_0_001': 0,
            'score_gt_0_01': 0,
            'score_gt_0_1': 0,
            'score_gt_0_25': 0,
        }

    return {
        'size': int(scores.size),
        'score_min': float(scores.min()),
        'score_max': float(scores.max()),
        'score_mean': float(scores.mean()),
        'score_gt_0_001': int((scores > 0.001).sum()),
        'score_gt_0_01': int((scores > 0.01).sum()),
        'score_gt_0_1': int((scores > 0.1).sum()),
        'score_gt_0_25': int((scores > 0.25).sum()),
    }


def _summarize_primary_raw_slices(raw: np.ndarray, format_name: str) -> dict[str, Any]:
    raw = np.asarray(raw)
    summary: dict[str, Any] = {}

    if raw.ndim != 2 or raw.shape[1] < 6:
        return summary

    summary['boxes_raw'] = _summarize_numeric_array(raw[:, :4])

    if format_name == 'xywh_obj_cls':
        summary['objectness_raw'] = _summarize_numeric_array(raw[:, 4])
        summary['classes_raw'] = _summarize_numeric_array(raw[:, 5:])
    elif format_name == 'xywh_cls':
        summary['classes_raw'] = _summarize_numeric_array(raw[:, 4:])
    elif format_name == 'xyxy_score_class':
        summary['scores_raw'] = _summarize_numeric_array(raw[:, 4])
        summary['class_ids_raw'] = _summarize_numeric_array(raw[:, 5])

    return summary


def inspect_raw_outputs(
    detector: OnnxDetector,
    image_path: Path,
) -> dict[str, Any]:
    image_path = image_path.expanduser().resolve()

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise PipelineError(f'Failed to read image: {image_path}')

    orig_h, orig_w = image_bgr.shape[:2]
    target_hw = _resolve_runtime_hw(detector, (orig_h, orig_w))
    prepared_bgr, scale, pad = _letterbox(image_bgr, target_hw)
    input_tensor = _to_nchw(prepared_bgr)

    outputs = detector.session.run(None, {detector.input_name: input_tensor})
    output_metas = detector.session.get_outputs()

    summary: dict[str, Any] = {
        'image_path': str(image_path),
        'orig_hw': (orig_h, orig_w),
        'input_hw': target_hw,
        'providers': list(detector.providers),
        'outputs': [],
    }

    for index, output in enumerate(outputs):
        arr = np.asarray(output)
        meta_name = output_metas[index].name if index < len(output_metas) else f'output_{index}'

        item: dict[str, Any] = {
            'index': index,
            'name': meta_name,
            'shape': tuple(int(dim) for dim in arr.shape),
            'dtype': str(arr.dtype),
            'size': int(arr.size),
        }

        if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
            arr_f32 = arr.astype(np.float32, copy=False)
            item['min'] = float(arr_f32.min())
            item['max'] = float(arr_f32.max())
            item['mean'] = float(arr_f32.mean())
            item['std'] = float(arr_f32.std())
            item['nonzero'] = int(np.count_nonzero(arr_f32))

        summary['outputs'].append(item)

    try:
        raw = _pick_detection_output(outputs)
        decoded = _decode_predictions(raw, detector.class_names)

        primary: dict[str, Any] = {
            'shape': tuple(int(dim) for dim in raw.shape),
            'format_name': decoded.format_name,
            'requires_nms': bool(decoded.requires_nms),
            'decoded_scores': _summarize_decoded_scores(decoded.scores),
            'raw_slices': _summarize_primary_raw_slices(raw, decoded.format_name),
        }

        if decoded.class_ids.size:
            unique_ids, counts = np.unique(decoded.class_ids, return_counts=True)
            pairs = sorted(
                zip(unique_ids.tolist(), counts.tolist(), strict=False),
                key=lambda item: item[1],
                reverse=True,
            )
            primary['top_class_ids'] = [
                {'class_id': int(class_id), 'count': int(count)}
                for class_id, count in pairs[:10]
            ]

        summary['primary_detection_output'] = primary
    except Exception as exc:
        summary['primary_detection_output_error'] = str(exc)

    summary['letterbox'] = {
        'scale': float(scale),
        'pad_x': float(pad[0]),
        'pad_y': float(pad[1]),
    }

    return summary
