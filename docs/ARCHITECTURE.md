# 🧱 Architecture

**Navigation**
[`Home`](../README.md) · [`Datasets`](DATASETS.md) · [`Training`](TRAINING.md) · [`CLI`](CLI.md) · [`Architecture`](ARCHITECTURE.md)

The project is structured around a strict separation between library code and CLI code.

## Contents

- [🗺️ High-Level Shape](#️-high-level-shape)
- [📦 `core/`](#-core)
- [🛠️ `tools/`](#️-tools)
- [📜 `scripts/`](#-scripts)
- [🎯 Design Rules](#-design-rules)
- [📦 Packaging](#-packaging)

<table>
  <tr>
    <td><strong>📝 Note</strong><br>The goal is simple: business logic stays importable, CLI stays thin, and each pipeline stage has one responsibility.</td>
  </tr>
</table>

## 🗺️ High-Level Shape

- `core/` contains domain logic
- `tools/` contains argument parsing and user-facing CLI behavior
- `scripts/` contains shell automation and demo flows
- `configs/train/` contains tracked training presets
- `configs/prepare/` contains tracked dataset-mutation recipes
- `docs/` contains user-facing documentation

## 📦 `core/`

### `core/common/`

Shared helpers and exceptions, for example:

- filesystem helpers
- copy/symlink helpers
- domain-level error types

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

## 🛠️ `tools/`

`tools/` is the CLI layer only.

Responsibilities:

- parse CLI args
- map args to `core` inputs
- print user-facing logs and errors
- define stable console entrypoints

Non-responsibilities:

- dataset conversion internals
- dataset inspection internals
- recipe application logic
- training planning internals

## 📜 `scripts/`

`scripts/` is intentionally separate from the public CLI surface.

Use it for:

- environment bootstrap
- demo dataset downloads
- demo model downloads

Do not put pipeline-critical business logic there.

## 🎯 Design Rules

1. Raw dataset peculiarities enter the system only through `yolo-convert-dataset` adapters.
2. `yolo-print-stats` is the mandatory analysis layer before dataset surgery.
3. `yolo-prepare-dataset` is optional and destructive by design.
4. Training consumes ordinary dataset YAMLs produced by conversion or preparation.
5. `tools/` should stay replaceable without rewriting `core/`.
6. New raw schemas should extend `convert_dataset_to_yolo`, not create parallel pipelines.

## 📦 Packaging

Public console scripts are defined in [`pyproject.toml`](../pyproject.toml):

- `yolo-convert-dataset`
- `yolo-print-stats`
- `yolo-prepare-dataset`
- `yolo-train`
- `yolo-report-ap`

---

**Next**
[`Home`](../README.md) · [`CLI Reference`](CLI.md) · [`Dataset Guide`](DATASETS.md)
