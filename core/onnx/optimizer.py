from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.common import PipelineError, ProgressCallback

from .common import (
    build_onnx_artifact_name,
    ensure_dir,
    ensure_target_provider_available,
    get_graph_level,
    get_providers,
    require_onnx,
    require_onnxruntime,
)


__all__ = ["OptimizeConfig", "optimize_onnx"]


@dataclass(slots=True)
class OptimizeConfig:
    input_model: Path
    output_dir: Path
    target: str = "cpu"
    graph_level: str = "extended"
    tag: str | None = None
    preprocess: bool = True
    int8: bool = False
    fp16: bool = False
    calib_dir: Path | None = None
    calib_size: int = 256
    input_hw: tuple[int, int] = (768, 768)
    calibration_method: str = "minmax"
    per_channel: bool = True
    reduce_range: bool = False
    u8u8: bool = False
    keep_io_types: bool = True

    def validate(self) -> None:
        self.input_model = self.input_model.expanduser().resolve()
        self.output_dir = self.output_dir.expanduser().resolve()
        if not self.input_model.exists():
            raise PipelineError(
                f"Input ONNX not found: {self.input_model}",
                hint="Run yolo-onnx-export first or point --input to an existing .onnx file.",
            )
        if self.int8 and self.target != "cpu":
            raise PipelineError("--int8 is only supported in this pipeline for --target cpu.")
        if self.fp16 and self.target != "cuda":
            raise PipelineError("--fp16 is only supported in this pipeline for --target cuda.")
        if self.int8 and self.fp16:
            raise PipelineError("Choose either --int8 or --fp16, not both.")
        if self.int8 and self.calib_dir is None:
            raise PipelineError(
                "--calib-dir is required for INT8 quantization.",
                hint="Point --calib-dir to a directory with representative calibration images.",
            )
        if self.calib_dir is not None:
            self.calib_dir = self.calib_dir.expanduser().resolve()
            if not self.calib_dir.exists():
                raise PipelineError(
                    f"Calibration directory not found: {self.calib_dir}",
                    hint="Point --calib-dir to a directory with representative images for INT8 calibration.",
                )
        if self.calib_size <= 0:
            raise PipelineError("--calib-size must be a positive integer.")
        if self.input_hw[0] <= 0 or self.input_hw[1] <= 0:
            raise PipelineError("Calibration input size must contain positive integers.")
        ensure_target_provider_available(self.target)


def get_calibration_method(name: str):
    ort = require_onnxruntime()
    mapping = {
        "minmax": ort.quantization.CalibrationMethod.MinMax,
        "entropy": ort.quantization.CalibrationMethod.Entropy,
        "percentile": ort.quantization.CalibrationMethod.Percentile,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise PipelineError(
            f"Unsupported calibration method: {name}",
            hint="Use --calibration-method minmax, entropy, or percentile.",
        ) from exc


def save_offline_optimized_model(input_model: Path, output_model: Path, target: str, graph_level: str) -> Path:
    ort = require_onnxruntime()
    output_model.parent.mkdir(parents=True, exist_ok=True)
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = get_graph_level(graph_level)
    sess_options.optimized_model_filepath = str(output_model)
    session = ort.InferenceSession(str(input_model), sess_options=sess_options, providers=get_providers(target))
    del session
    return output_model


def run_quant_preprocess(input_model: Path, output_model: Path) -> Path:
    ort = require_onnxruntime()
    output_model.parent.mkdir(parents=True, exist_ok=True)
    ort.quantization.shape_inference.quant_pre_process(
        input_model=str(input_model),
        output_model_path=str(output_model),
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=True,
        auto_merge=False,
        verbose=1,
    )
    return output_model


def convert_model_to_fp16(input_model: Path, output_model: Path, keep_io_types: bool = True) -> Path:
    onnx = require_onnx()
    try:
        from onnxconverter_common import float16
    except ImportError as exc:
        raise PipelineError(
            "FP16 conversion requires onnxconverter-common.",
            hint="Install the CUDA profile with: source scripts/setup_env.sh cuda.",
        ) from exc
    model = onnx.load(str(input_model))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=keep_io_types)
    onnx.save(model_fp16, str(output_model))
    return output_model


def quantize_model_int8(
    input_model: Path,
    output_model: Path,
    calib_dir: Path,
    calib_size: int,
    input_hw: tuple[int, int],
    calibration_method: str,
    per_channel: bool,
    reduce_range: bool,
    u8u8: bool,
) -> Path:
    ort = require_onnxruntime()
    from .calibrate import ImageCalibrationDataReader, collect_image_paths

    image_paths = collect_image_paths(calib_dir, calib_size)
    reader = ImageCalibrationDataReader(input_model, image_paths, input_hw)

    if u8u8:
        activation_type = ort.quantization.QuantType.QUInt8
        weight_type = ort.quantization.QuantType.QUInt8
    else:
        activation_type = ort.quantization.QuantType.QInt8
        weight_type = ort.quantization.QuantType.QInt8

    ort.quantization.quantize_static(
        model_input=str(input_model),
        model_output=str(output_model),
        calibration_data_reader=reader,
        quant_format=ort.quantization.QuantFormat.QDQ,
        activation_type=activation_type,
        weight_type=weight_type,
        per_channel=per_channel,
        reduce_range=reduce_range,
        calibrate_method=get_calibration_method(calibration_method),
    )
    return output_model


def optimize_onnx(cfg: OptimizeConfig, *, progress_callback: ProgressCallback | None = None) -> dict[str, Path]:
    cfg.validate()
    ensure_dir(cfg.output_dir)

    total_steps = 2
    if cfg.preprocess:
        total_steps += 1
    if cfg.fp16:
        total_steps += 1
    if cfg.int8:
        total_steps += 1

    step = 0
    if progress_callback is not None:
        progress_callback("onnx:optimize:init", step, total_steps, "onnx:optimize: validate")

    stem = cfg.input_model.stem
    artifacts: dict[str, Path] = {"input": cfg.input_model}
    working_model = cfg.input_model

    if cfg.preprocess:
        step += 1
        if progress_callback is not None:
            progress_callback("onnx:optimize", step, total_steps, "onnx:optimize: preprocess")
        prep_model = cfg.output_dir / build_onnx_artifact_name(stem, stage="preprocess", tag=cfg.tag)
        working_model = run_quant_preprocess(cfg.input_model, prep_model)
        artifacts["preprocess"] = working_model

    step += 1
    if progress_callback is not None:
        progress_callback("onnx:optimize", step, total_steps, "onnx:optimize: fp32 graph optimize")
    fp32_opt = cfg.output_dir / build_onnx_artifact_name(
        stem,
        stage="optimize",
        target=cfg.target,
        graph_level=cfg.graph_level,
        precision="fp32",
        tag=cfg.tag,
    )
    save_offline_optimized_model(working_model, fp32_opt, cfg.target, cfg.graph_level)
    artifacts["optimize_fp32"] = fp32_opt

    if cfg.fp16:
        step += 1
        if progress_callback is not None:
            progress_callback("onnx:optimize", step, total_steps, "onnx:optimize: fp16 convert")
        fp16_raw = cfg.output_dir / build_onnx_artifact_name(
            stem,
            stage="convert",
            target=cfg.target,
            precision="fp16",
            variant="raw",
            tag=cfg.tag,
        )
        fp16_opt = cfg.output_dir / build_onnx_artifact_name(
            stem,
            stage="optimize",
            target=cfg.target,
            graph_level=cfg.graph_level,
            precision="fp16",
            tag=cfg.tag,
        )
        convert_model_to_fp16(working_model, fp16_raw, keep_io_types=cfg.keep_io_types)
        artifacts["convert_fp16_raw"] = fp16_raw
        save_offline_optimized_model(fp16_raw, fp16_opt, cfg.target, cfg.graph_level)
        artifacts["optimize_fp16"] = fp16_opt

    if cfg.int8:
        step += 1
        if progress_callback is not None:
            progress_callback("onnx:optimize", step, total_steps, "onnx:optimize: int8 quantize")
        int8_raw = cfg.output_dir / build_onnx_artifact_name(
            stem,
            stage="quantize",
            target=cfg.target,
            precision="int8",
            variant="raw",
            tag=cfg.tag,
        )
        int8_opt = cfg.output_dir / build_onnx_artifact_name(
            stem,
            stage="optimize",
            target="cpu",
            graph_level=cfg.graph_level,
            precision="int8",
            tag=cfg.tag,
        )
        quantize_model_int8(
            input_model=working_model,
            output_model=int8_raw,
            calib_dir=cfg.calib_dir,  # type: ignore[arg-type]
            calib_size=cfg.calib_size,
            input_hw=cfg.input_hw,
            calibration_method=cfg.calibration_method,
            per_channel=cfg.per_channel,
            reduce_range=cfg.reduce_range,
            u8u8=cfg.u8u8,
        )
        artifacts["quantize_int8_raw"] = int8_raw
        save_offline_optimized_model(int8_raw, int8_opt, "cpu", cfg.graph_level)
        artifacts["optimize_int8"] = int8_opt

    step += 1
    if progress_callback is not None:
        progress_callback("onnx:optimize", step, total_steps, "onnx:optimize: done")

    return artifacts
