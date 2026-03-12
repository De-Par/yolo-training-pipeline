# Benchmark Guide

**Navigation**
[`Home`](../README.md) · [`Datasets`](DATASETS.md) · [`Training`](TRAINING.md) · [`Bench`](BENCH.md) · [`ONNX`](ONNX.md) · [`CLI`](CLI.md) · [`Architecture`](ARCHITECTURE.md)

This guide covers the benchmark branch of the pipeline: how to measure inference latency, FPS, and holdout quality for trained YOLO checkpoints.

## Contents

- [Goal](#goal)
- [Main Entry Point](#main-entry-point)
- [What Benchmark Measures](#what-benchmark-measures)
- [Benchmark Config Structure](#benchmark-config-structure)
- [Dataset Source Model](#dataset-source-model)
- [CPU Benchmark Pattern](#cpu-benchmark-pattern)
- [GPU Benchmark Pattern](#gpu-benchmark-pattern)
- [Outputs](#outputs)
- [Interpretation Notes](#interpretation-notes)
- [Common Failure Modes](#common-failure-modes)

<table>
  <tr>
    <td><strong>📝 Note</strong><br>Benchmarking is a post-training stage. Use a holdout split such as <code>test</code> by default, not the same split that informed training decisions.</td>
  </tr>
</table>

## Goal

The benchmark stage answers two separate questions with one report:

1. How fast is inference on the target hardware envelope?
2. What quality does the model achieve on the chosen holdout source?

The report combines:

- per-point latency in `ms / image`
- per-point throughput in `FPS`
- average original and effective input sizes
- global quality metrics such as `mAP50-95`
- per-class `mAP50-95` with instance counts

## Main Entry Point

Use:

```bash
yolo-benchmark-report --config configs/bench/cpu.example.yaml
```

Tracked examples:

- [`../configs/bench/cpu.example.yaml`](../configs/bench/cpu.example.yaml)
- [`../configs/bench/gpu.example.yaml`](../configs/bench/gpu.example.yaml)

## What Benchmark Measures

The speed path measures backend inference latency on already prepared input tensors.

It includes:

- model/backend invocation
- device execution
- synchronization where needed
- output tensor or array materialization

It does not include:

- reading images from disk
- preprocessing before the timed batch
- full Ultralytics postprocess to final rendered detections

This is intentional. The benchmark is meant to compare backend execution fairly.

## Benchmark Config Structure

Top-level sections:

```yaml
hardware:
imgsz:
dataset:
benchmark:
quality:
output:
model: path/to/model.pt
```

### `model`

A direct path to the benchmarked checkpoint or model file.

Supported suffixes:

- `.pt`
- `.onnx`

Backend selection is inferred from the suffix.

### `hardware`

Defines benchmark points.

Important fields:

- `kind`: `cpu` or `gpu`
- `device`: runtime device string such as `cpu` or `cuda:0`
- `points`: hardware points for the speed plots

CPU points may define:

- `label`
- `cores`
- `threads`

`cores` supports:

- explicit ids: `[0, 1, 2]`
- inclusive ranges: `["0-3"]`
- mixed form: `["0-3", 6, "8-9"]`

GPU points may define:

- `label`
- optional `pre_cmd`
- optional `post_cmd`

### `imgsz`

Controls the benchmark preprocessor.

Modes:

- `square`: force `N x N`
- `rect`: force `H x W`
- `dynamic`: preserve aspect ratio and pad to stride

When `mode=dynamic`, the quality path still needs one numeric fallback for `Ultralytics val`, so keep `val_imgsz_fallback` set.

### `benchmark`

Controls the speed path.

Important fields:

- `batch`: usually `1` for latency comparisons
- `warmup_iters`: warmup passes on one image before timing starts
- `max_images`: cap timed images per hardware point
- `shuffle`: shuffle the chosen speed source before capping
- `seed`: deterministic sampling seed
- `half`: relevant only for PyTorch CUDA

### `quality`

Controls the quality evaluation path.

Public fields:

- `batch`
- `save_json`
- `plots`

Everything else is fixed internally for reproducibility:

- `workers = 0`
- `verbose = false`
- `conf = 0.001`
- `iou = 0.7`

### `output`

Defines the final artifacts:

- `dir`
- `report_png`
- `report_json`
- `speed_csv`

## Dataset Source Model

Benchmark uses one default source plus optional overrides.

```yaml
dataset:
  source:
    data_yaml: data/converted/my_dataset/my_dataset.yaml
    split: test
```

This means both speed and quality use the same holdout source by default.

Optional overrides:

```yaml
dataset:
  source:
    data_yaml: data/converted/my_dataset/my_dataset.yaml
    split: test

  speed:
    split: test

  quality:
    split: val
```

Rules:

- `dataset.source` is required
- `dataset.speed` is optional
- `dataset.quality` is optional
- if no override is given, both inherit `dataset.source`
- if `dataset.quality` is explicitly defined with `data_yaml` but no `split`, it defaults to `test`

Direct-path mode is also supported.

Speed-only direct path:

```yaml
dataset:
  source:
    data_yaml: data/converted/my_dataset/my_dataset.yaml
    split: test

  speed:
    images_dir: data/bench_images
```

Quality direct path:

```yaml
dataset:
  source:
    data_yaml: data/converted/my_dataset/my_dataset.yaml
    split: test

  quality:
    images_dir: data/converted/my_dataset/images/val
    annotations_dir: data/converted/my_dataset/labels/val
```

Quality direct path requires labels.

## CPU Benchmark Pattern

Use CPU points to compare different affinity and thread envelopes.

Typical pattern:

```yaml
hardware:
  kind: cpu
  device: cpu
  points:
    - label: 1 core
      cores: [0]
      threads: 1
    - label: 2 cores
      cores: [0, 1]
      threads: 2
    - label: 4 cores
      cores: ["0-3"]
      threads: 4
```

Recommended defaults:

- `batch: 1`
- `warmup_iters: 8` or `16`
- `max_images`: a moderate cap such as `128` or `256`

## GPU Benchmark Pattern

Use GPU points to compare default and tuned deployment modes.

Typical pattern:

```yaml
hardware:
  kind: gpu
  device: cuda:0
  points:
    - label: gpu default
    - label: gpu tuned
      # pre_cmd: ...
      # post_cmd: ...
```

Notes:

- `.pt` goes through Ultralytics/PyTorch
- `.onnx` goes through ONNX Runtime
- `benchmark.half` matters only for PyTorch CUDA

## Outputs

A completed benchmark run writes:

- one PNG report
- one JSON summary
- one CSV with per-point speed rows
- optional `quality_artifacts/` only when `quality.save_json` or `quality.plots` is enabled

Example:

```text
runs/bench/cpu_example/
├── benchmark_report.cpu.png
├── benchmark_report.cpu.json
└── benchmark_speed.cpu.csv
```

## Interpretation Notes

### Why use `test` by default

If the model was tuned while looking at `val`, benchmarking on `val` inflates confidence.
The honest default is a holdout split such as `test`.

### Why warmup exists

The first few forwards include one-time overheads such as:

- kernel initialization
- memory setup
- graph/runtime initialization

That is why the timed section starts only after `warmup_iters`.

### What the numbers in per-class bars mean

Per-class bars show:

- `mAP50-95`
- instance count in parentheses

Example:

```text
0.143 (57 inst)
```

means:

- `mAP50-95 = 0.143`
- `57` ground-truth instances of that class in the quality source

## Common Failure Modes

### Quality source has no labels

If `test` is unlabeled, either:

- keep `dataset.source` on `test` for speed
- override `dataset.quality` to a labeled split such as `val`

### Speed source is too large

Set:

```yaml
benchmark:
  max_images: 128
```

This caps the timed subset per hardware point and keeps runs practical.

### CPU benchmark numbers are noisy

Check:

- `cores`
- `threads`
- other heavy processes on the machine
- whether you are mixing performance and efficiency cores unintentionally

### Direct-path quality gives wrong class counts

Use a proper `annotations_dir` that mirrors the image layout, or use a normal dataset YAML source.

---

**Next**
[`Home`](../README.md) · [`CLI Reference`](CLI.md) · [`Training Guide`](TRAINING.md) · [`Architecture`](ARCHITECTURE.md)
