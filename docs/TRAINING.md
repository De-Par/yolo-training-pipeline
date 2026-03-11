# Training Guide

**Navigation**
[`Home`](../README.md) · [`Datasets`](DATASETS.md) · [`Training`](TRAINING.md) · [`ONNX`](ONNX.md) · [`CLI`](CLI.md) · [`Architecture`](ARCHITECTURE.md)

This guide covers the training side of the pipeline.

## Contents

- [Main Entry Point](#main-entry-point)
- [Tracked Config](#tracked-config)
- [Training on Converted vs Prepared Datasets](#training-on-converted-vs-prepared-datasets)
- [Config Plus Override Pattern](#config-plus-override-pattern)
- [Training Outputs](#training-outputs)
- [Per-Class AP Analysis](#per-class-ap-analysis)
- [Val Versus Test](#val-versus-test)
- [Practical Advice](#practical-advice)
- [Common Failure Modes](#common-failure-modes)

## Main Entry Point

Use:

```bash
yolo-train --cfg configs/train/nvidia.example.yaml
```

or provide the required fields from CLI.

## Tracked Config

The repository currently tracks one training preset example:

- [`../configs/train/nvidia.example.yaml`](../configs/train/nvidia.example.yaml)

This is a documented NVIDIA-oriented baseline example, not a locked policy file.

## Training on Converted vs Prepared Datasets

You can train on either of these:

1. a freshly converted YOLO-styled dataset
2. a YOLO-styled dataset that was later mutated by `yolo-prepare-dataset`

In both cases the training input is just a normal dataset YAML:

```bash
yolo-train \
  --data data/converted/my_dataset/my_dataset.yaml \
  --model models/YOLOv26/yolo26n.pt
```

If you applied preparation in place, the same dataset directory now contains the updated YAML and class mapping.

## Config Plus Override Pattern

Recommended pattern:

```bash
yolo-train \
  --cfg configs/train/nvidia.example.yaml \
  --data data/converted/my_dataset/my_dataset.yaml \
  --model models/YOLOv26/yolo26n.pt \
  --name exp-run
```

Why this is preferred:

- stable defaults stay in YAML
- experiment-specific knobs stay on CLI
- runs are easier to reproduce

Keep CLI overrides short. The intended split is:

- CLI for the most important run-level changes: `model`, `data`, `epochs`, `imgsz`, `batch`, `device`, `name`
- YAML config for everything else: `workers`, `amp`, `cache`, `compile`, `val`, `seed`, `project`, `exist_ok`, `verbose`, and other Ultralytics keys

## Training Outputs

Ultralytics writes into:

```text
<project>/<name>/
```

Typical artifacts:

- `weights/best.pt`
- `weights/last.pt`
- `results.csv`
- `args.yaml`
- training curves and metric plots
- validation visualizations

## Per-Class AP Analysis

Use `yolo-report-ap` after training:

```bash
yolo-report-ap \
  --model runs/my_run/weights/best.pt \
  --data data/converted/my_dataset/my_dataset.yaml \
  --split val \
  --device 0
```

Terminology:

- `AP` = `Average Precision`
- `mAP` = `mean Average Precision`

In detection, `AP` is computed per class from the precision/recall curve.
`mAP` is the mean of those per-class AP values.

This is especially useful when:

- the dataset has a long tail
- aggregate mAP hides dead classes
- you need evidence for future recipe changes in `yolo-prepare-dataset`

## Val Versus Test

Use `val` while you are still making decisions about:

- split recipes
- class merges or drops
- model choice
- training config

Use `test` only for the final holdout check after those decisions are already fixed.

If your dataset contains `test`, the final command should typically be:

```bash
yolo-report-ap \
  --model runs/my_run/weights/best.pt \
  --data data/converted/my_dataset/my_dataset.yaml \
  --split test
```

## Practical Advice

### Use `yolo-print-stats` before training

Before tuning the optimizer, verify the dataset itself:

- class distribution
- tiny-object ratio
- split sizes
- empty labels

### Use `yolo-prepare-dataset` only for intentional task changes

Good reasons:

- merge classes into a coarser taxonomy
- drop classes that are out of scope
- create a smaller smoke dataset

Bad reason:

- using it as a generic preprocessing reflex without looking at stats first

### If you hit OOM

Reduce in this order:

1. `batch`
2. `imgsz`
3. model size
4. workers if system memory is also tight

### Reproducibility

- keep run names explicit
- keep the prepare recipe under version control
- keep the train config under version control
- treat the converted dataset as the reset point if preparation is destructive

### ONNX is downstream of training

ONNX export and optimization belong to deployment prep, not training itself.
Use [`ONNX Guide`](ONNX.md) after you already have a checkpoint that you want to serve.

## Common Failure Modes

### Training falls back to CPU

Check:

- `--device`
- installed torch build
- CUDA availability inside the active environment

### Metrics look wrong

Before changing optimizer settings, verify:

- the dataset YAML points to the intended dataset version
- `yolo-print-stats` matches your expectations after preparation
- you are not comparing runs built from different split fractions or class mappings

---

**Next**
[`CLI Reference`](CLI.md) · [`ONNX Guide`](ONNX.md) · [`Architecture`](ARCHITECTURE.md) · [`Home`](../README.md)
