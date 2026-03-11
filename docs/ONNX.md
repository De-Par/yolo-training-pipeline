# ONNX Guide

**Navigation**
[`Home`](../README.md) · [`Datasets`](DATASETS.md) · [`Training`](TRAINING.md) · [`ONNX`](ONNX.md) · [`CLI`](CLI.md) · [`Architecture`](ARCHITECTURE.md)

This guide covers ONNX export, runtime-specific optimization, and deployment-prep artifacts.

## Contents

- [Where ONNX Fits](#where-onnx-fits)
- [Environment Profiles](#environment-profiles)
- [Stage 1: Export to ONNX](#stage-1-export-to-onnx)
- [Stage 2: Optimize for Runtime](#stage-2-optimize-for-runtime)
- [One-Shot Pipeline](#one-shot-pipeline)
- [Artifact Naming](#artifact-naming)
- [Deployment Notes](#deployment-notes)

<table>
  <tr>
    <td><strong>📝 Note</strong><br>ONNX is a post-training branch. It starts from a YOLO checkpoint such as <code>weights/best.pt</code> and produces deployment-oriented artifacts.</td>
  </tr>
</table>

## Where ONNX Fits

The intended flow is:

1. train a model with `yolo-train`
2. verify metrics with `yolo-report-ap`
3. export the chosen checkpoint to ONNX
4. optimize the ONNX graph for the target runtime
5. ship those artifacts into deployment

Use ONNX after training, not instead of training.

## Environment Profiles

Choose the setup profile based on what you need:

- `base`: training, stats, AP reports, and plain ONNX export
- `cpu`: everything in `base` plus ONNX Runtime CPU optimization and INT8 quantization
- `cuda`: everything in `base` plus ONNX Runtime CUDA optimization and FP16 conversion

Examples:

```bash
source scripts/setup_env.sh base
source scripts/setup_env.sh cpu
source scripts/setup_env.sh cuda
```

<table>
  <tr>
    <td><strong>⚠️ Warning</strong><br><code>yolo-onnx-optimize</code> and <code>yolo-onnx-pipeline</code> require an ONNX Runtime profile. Plain export works in <code>base</code>, but runtime optimization does not.</td>
  </tr>
</table>

## Stage 1: Export to ONNX

Export a trained checkpoint:

```bash
yolo-onnx-export \
  --weights runs/my_run/weights/best.pt \
  --output deploy/onnx/model.export.fp32.onnx \
  --imgsz 1024
```

What this does:

- loads the YOLO `.pt` checkpoint through Ultralytics
- exports an ONNX graph
- optionally simplifies the graph
- writes a deployment-ready `.onnx` baseline

Typical output:

```text
deploy/onnx/model.export.fp32.onnx
```

## Stage 2: Optimize for Runtime

Optimize the exported ONNX for a target runtime.

### CPU example with INT8

```bash
yolo-onnx-optimize \
  --input deploy/onnx/model.export.fp32.onnx \
  --output-dir deploy/onnx/cpu \
  --target cpu \
  --int8 \
  --calib-dir data/raw/fashionpedia/train/images
```

### CUDA example with FP16

```bash
yolo-onnx-optimize \
  --input deploy/onnx/model.export.fp32.onnx \
  --output-dir deploy/onnx/cuda \
  --target cuda \
  --fp16
```

What optimization can do:

- ONNX Runtime graph optimization
- optional quantization preprocessing
- INT8 QDQ quantization for CPU
- FP16 conversion for CUDA

INT8 notes:

- requires `--target cpu`
- requires `--calib-dir`
- uses representative images for calibration

FP16 notes:

- requires `--target cuda`
- requires the CUDA setup profile because `onnxconverter-common` is installed there

## One-Shot Pipeline

If you already know the target runtime, use the combined command:

```bash
yolo-onnx-pipeline \
  --weights runs/my_run/weights/best.pt \
  --artifact-dir deploy/onnx/cuda \
  --target cuda \
  --fp16
```

This command:

1. exports the checkpoint to ONNX
2. reuses that exported model as optimizer input
3. writes the optimized artifacts into one target directory

Use it when you want a short deployment-prep flow.

## Artifact Naming

Artifact names now follow a strict dot-separated schema: `<stem>.<stage>[.<target>][.<graph_level>][.<precision>][.<variant>][.<tag>].onnx`.

Typical examples:

- `model.export.fp32.onnx`
- `model.preprocess.onnx`
- `model.optimize.cpu.extended.fp32.onnx`
- `model.convert.cuda.fp16.raw.onnx`
- `model.optimize.cuda.extended.fp16.onnx`
- `model.quantize.cpu.int8.raw.onnx`
- `model.optimize.cpu.extended.int8.onnx`

Keep deployment artifacts outside `runs/` so they do not mix with training outputs.

## Deployment Notes

Recommended pattern:

- keep training outputs under `runs/`
- keep ONNX artifacts under `deploy/onnx/`
- version deployment artifacts by runtime target and precision
- always validate accuracy on `val` or `test` before promoting a checkpoint into deployment export

For deployment handoff, capture at least:

- checkpoint source
- dataset YAML used for evaluation
- `yolo-report-ap` report
- exported ONNX path
- optimized ONNX path
- runtime target (`cpu` or `cuda`)
- optimization options such as `int8`, `fp16`, `graph-level`, and calibration source

---

**Next**
[`CLI Reference`](CLI.md) · [`Training Guide`](TRAINING.md) · [`Architecture`](ARCHITECTURE.md) · [`Home`](../README.md)
