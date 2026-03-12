# Architecture

**Navigation**
[`Home`](../README.md) · [`Datasets`](DATASETS.md) · [`Training`](TRAINING.md) · [`Bench`](BENCH.md) · [`ONNX`](ONNX.md) · [`CLI`](CLI.md) · [`Architecture`](ARCHITECTURE.md)

The project is structured around a strict separation between library code and CLI code.

## Contents

- [High-Level Shape](#high-level-shape)
- [`core/`](#core)
- [`tools/`](#tools)
- [`scripts/`](#scripts)
- [Design Rules](#design-rules)
- [Packaging](#packaging)

<table>
  <tr>
    <td><strong>📝 Note</strong><br>The goal is simple: business logic stays importable, CLI stays thin, and each pipeline stage has one responsibility.</td>
  </tr>
</table>

## High-Level Shape

- `core/` contains domain logic
- `tools/` contains argument parsing and user-facing CLI behavior
- `scripts/` contains shell automation and demo flows
- `configs/` contains tracked example YAMLs
- `docs/` contains user-facing documentation

## `core/`

### `core/bench/`

Benchmark-side business logic.

Current responsibilities:

- benchmark config validation
- dataset source resolution for speed and quality runs
- backend latency measurement for `.pt` and `.onnx` models
- PNG, JSON, and CSV benchmark report generation

### `core/common/`

Shared helpers and exceptions, for example:

- filesystem helpers
- console formatting
- progress reporting
- domain-level error types
- CLI runtime adapters

### `core/datasets/`

Dataset-side business logic.

Current responsibilities:

- raw dataset conversion through `convert_dataset_to_yolo`
- YOLO dataset inspection through `stats`
- in-place dataset mutation through `prepare_yolo_dataset`
- dataset YAML generation

Important design split:

- `convert_dataset_to_yolo` translates raw schemas into YOLO format
- `stats` inspects YOLO datasets without changing them
- `prepare_yolo_dataset` mutates YOLO datasets intentionally through a recipe

### `core/training/`

Training-side business logic.

Current responsibilities:

- config loading
- training plan construction
- Ultralytics training execution
- per-class AP export

### `core/onnx/`

Deployment-prep business logic.

Current responsibilities:

- exporting YOLO checkpoints to ONNX
- ONNX Runtime graph optimization
- calibration image handling for INT8 quantization
- FP16 conversion for CUDA deployment
- export plus optimize orchestration

Design intent:

- `exporter` is checkpoint-to-ONNX only
- `optimizer` is ONNX-to-runtime-artifacts only
- `pipeline` composes both when a one-shot flow is useful

## `tools/`

`tools/` is the CLI layer only.

Responsibilities:

- parse CLI args
- map args to `core` inputs
- print user-facing logs and errors
- define stable console entrypoints

Subareas:

- top-level `tools/*.py` for dataset, training, and benchmark commands
- `tools/onnx/*.py` for ONNX export and optimization commands

Non-responsibilities:

- dataset conversion internals
- dataset inspection internals
- recipe application logic
- training planning internals
- ONNX optimization internals

## `scripts/`

`scripts/` is intentionally separate from the public CLI surface.

Use it for:

- environment bootstrap
- demo dataset downloads through separate Fashionpedia and DeepFashion2 scripts
- demo model downloads

Do not put pipeline-critical business logic there.

## Design Rules

1. Raw dataset peculiarities enter the system only through `yolo-convert-dataset` adapters.
2. `yolo-print-stats` is the mandatory analysis layer before dataset surgery.
3. `yolo-prepare-dataset` is optional and destructive by design.
4. Training consumes ordinary dataset YAMLs produced by conversion or preparation.
5. `yolo-report-ap` is the metrics/reporting layer after training.
6. ONNX export and optimization are post-training deployment stages.
7. `tools/` should stay replaceable without rewriting `core/`.
8. New raw schemas should extend `convert_dataset_to_yolo`, not create parallel pipelines.
9. Benchmarking stays post-training and should measure holdout splits by default.

## Packaging

Public console scripts are defined in [`../pyproject.toml`](../pyproject.toml):

- `yolo-convert-dataset`
- `yolo-print-stats`
- `yolo-prepare-dataset`
- `yolo-train`
- `yolo-report-ap`
- `yolo-onnx-export`
- `yolo-onnx-optimize`
- `yolo-onnx-pipeline`
- `yolo-benchmark-report`

---

**Next**
[`Home`](../README.md) · [`CLI Reference`](CLI.md) · [`Dataset Guide`](DATASETS.md) · [`Bench Guide`](BENCH.md) · [`ONNX Guide`](ONNX.md)
