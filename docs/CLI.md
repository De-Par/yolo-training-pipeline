# CLI Reference

**Navigation**
[`Home`](../README.md) · [`Datasets`](DATASETS.md) · [`Training`](TRAINING.md) · [`Bench`](BENCH.md) · [`ONNX`](ONNX.md) · [`CLI`](CLI.md) · [`Architecture`](ARCHITECTURE.md)

This document describes the current public CLI surface.

## Contents

- [Public Commands](#public-commands)
- [`yolo-convert-dataset`](#yolo-convert-dataset)
- [`yolo-print-stats`](#yolo-print-stats)
- [`yolo-prepare-dataset`](#yolo-prepare-dataset)
- [`yolo-train`](#yolo-train)
- [`yolo-report-ap`](#yolo-report-ap)
- [`yolo-onnx-export`](#yolo-onnx-export)
- [`yolo-onnx-optimize`](#yolo-onnx-optimize)
- [`yolo-onnx-pipeline`](#yolo-onnx-pipeline)
- [`yolo-benchmark-report`](#yolo-benchmark-report)
- [Fallback Entry Points](#fallback-entry-points)

<table>
  <tr>
    <td><strong>📝 Note</strong><br>Installed commands are the primary interface. <code>python tools/...</code> is fallback only.</td>
  </tr>
</table>

## Public Commands

| Command | Purpose |
|---|---|
| `yolo-convert-dataset` | Convert raw annotations into a normalized YOLO-styled dataset |
| `yolo-print-stats` | Print detailed stats for a YOLO-styled dataset |
| `yolo-prepare-dataset` | Mutate a YOLO-styled dataset in place from a YAML recipe |
| `yolo-train` | Run Ultralytics training |
| `yolo-report-ap` | Validate a checkpoint and export per-class AP reports |
| `yolo-onnx-export` | Export a YOLO checkpoint to ONNX |
| `yolo-onnx-optimize` | Optimize an ONNX model for CPU or CUDA deployment |
| `yolo-onnx-pipeline` | Export and optimize in one command |
| `yolo-benchmark-report` | Measure backend latency and quality on a benchmark split and render a PNG report |

## `yolo-convert-dataset`

Convert raw detection data into a YOLO-styled dataset without changing class semantics.

### Main outputs

- `images/train`, `images/val`
- `labels/train`, `labels/val`
- `classes.txt`
- `conversion_report_train.json`
- `conversion_report_val.json`
- `<dataset>.yaml`

### Syntax

```bash
yolo-convert-dataset \
  --dataset-name my_dataset \
  --input-format coco-detection \
  --train-images-dir data/raw/my_dataset/train/images \
  --train-annotations data/raw/my_dataset/train/annotations.json \
  --val-images-dir data/raw/my_dataset/val/images \
  --val-annotations data/raw/my_dataset/val/annotations.json \
  --clean
```

### Important arguments

| Option | Required | Description |
|---|:---:|---|
| `--dataset-name` | yes | logical dataset name for `data/converted/<name>` |
| `--input-format` | yes | raw schema adapter: `coco-detection` or `per-image-json` |
| `--train-images-dir` | yes | train image directory |
| `--train-annotations` | yes | train annotations file or directory |
| `--val-images-dir` | yes | val image directory |
| `--val-annotations` | yes | val annotations file or directory |
| `--output-root` | no | converted dataset root, default `data/converted` |
| `--link-mode` | no | `symlink` or `copy` |
| `--train-fraction` | no | optional conversion-time sampling |
| `--val-fraction` | no | optional conversion-time sampling |
| `--clean` | no | clean output dataset before regeneration |

### `per-image-json` only arguments

| Option | Description |
|---|---|
| `--class-names-file` | class name list used to resolve category ids |
| `--object-prefix` | object key prefix, default `item` |
| `--category-id-key` | key used for class id lookup |
| `--bbox-key` | key holding box coordinates |
| `--bbox-format` | `xyxy` or `xywh` |
| `--image-width-key` | width field name; falls back to the actual image width when missing |
| `--image-height-key` | height field name; falls back to the actual image height when missing |

`yolo-prepare-dataset` recipe note:

- Combined resplit modes always rename moved image/label pairs to stable unique hash-based names.

## `yolo-benchmark-report`

Measure inference latency/FPS across configured hardware points and render a single PNG report with speed, dataset sizing summary, and quality metrics. By default both use one benchmark source; optional overrides let you split them deliberately. Quality artifacts are written only when `quality.plots` or `quality.save_json` is enabled.

### Syntax

```bash
yolo-benchmark-report --config configs/bench/cpu.example.yaml
```

### Config highlights

- `hardware.kind`: `cpu` or `gpu`
- `hardware.points`: speed points for the latency/FPS plot; CPU `cores` may use ids or ranges like `["0-7", 10]`
- `imgsz.mode`: `square`, `rect`, or `dynamic`
- `model`: backend is inferred from `.pt` or `.onnx`
- `dataset.source`: default benchmark source, usually a holdout split such as `test`
- `dataset.speed`: optional speed override via another split or direct `images_dir`
- `dataset.quality`: optional labeled quality override via another split or direct `images_dir + annotations_dir` (defaults to `test` when it inherits a dataset YAML source and no split is given)
- by default both speed and quality inherit `dataset.source`; override only when you intentionally want another source
- `benchmark.batch`: usually `1` for apples-to-apples latency measurements
- `benchmark.warmup_iters`: warm up on one image before timing starts
- `benchmark.max_images`: cap the number of timed images per hardware point
- `output.report_png`: final report path

## `yolo-print-stats`

Print detailed stats for a YOLO-styled dataset.

### Main outputs

- printed summary block
- printed per-class table
- `dataset_stats.json`
- `dataset_stats_train.png`
- `dataset_stats_val.png`
- `dataset_stats_test.png` when the dataset contains `test`

### Syntax

```bash
yolo-print-stats --dataset-dir data/converted/my_dataset
```

### Important arguments

| Option | Required | Description |
|---|:---:|---|
| `--dataset-dir` | yes | YOLO-styled dataset directory |
| `--output-json` | no | override JSON output path |
| `--output-png` | no | override base PNG path, writes one `<stem>_<split>.png` file per detected split |

## `yolo-prepare-dataset`

Mutate a YOLO-styled dataset in place using a YAML recipe.

### Syntax

```bash
yolo-prepare-dataset \
  --dataset-dir data/converted/my_dataset \
  --recipe configs/prepare/prepare.example.yaml
```

### Important arguments

| Option | Required | Description |
|---|:---:|---|
| `--dataset-dir` | yes | YOLO-styled dataset to mutate |
| `--recipe` | yes | YAML recipe describing split management and class transforms |

### Recipe structure

```yaml
dataset_name: my_dataset_prepared
empty_policy: drop

split:
  mode: keep_existing
  seed: 42
  train_fraction: 1.0
  val_fraction: 1.0
  test_fraction: 0.0
  min_val_instances_per_class: 0
  min_test_instances_per_class: 0
  per_class_min_val_instances: {}
  per_class_min_test_instances: {}

classes:
  keep: []
  drop: []
  remap:
    - name: footwear
      from: [shoe, boot, sandal]
```

Split modes:

- `keep_existing`: keep current `train` / `val` / `test` boundaries
- `sample_existing`: downsample current splits independently
- `resplit_combined_random`: merge current splits and resplit by image fractions
- `resplit_combined_by_instances`: merge current splits and satisfy `val` / `test` instance minimums before filling nominal fractions

For `per_class_min_*`, examples in this repository prefer compact YAML flow-style mappings such as `{23: 50, "shirt, blouse": 30}`.

Selector syntax:

- exact class names, for example `"shirt, blouse"`
- numeric class ids, for example `23`
- id ranges, for example `"30-35"`

If a class name contains commas, quote it exactly as written in `classes.txt`.

## `yolo-train`

Run Ultralytics training on any YOLO dataset YAML produced by conversion or preparation.

### Syntax

```bash
yolo-train --cfg configs/train/nvidia.example.yaml
```

### Important arguments

| Option | Required | Description |
|---|:---:|---|
| `--cfg` | no | YAML config with train parameters |
| `--data` | no | dataset YAML override |
| `--model` | no | model checkpoint or architecture override |
| `--epochs` | no | training epochs override |
| `--imgsz` | no | square image size override |
| `--batch` | no | batch size override |
| `--device` | no | `cpu`, `0`, `cuda:0`, or `mps` |
| `--name` | no | run directory name override |

Everything else should live in the training YAML config.

## `yolo-report-ap`

Run validation and export per-class AP metrics to CSV and JSON.

`AP` means `Average Precision`. `mAP` means `mean Average Precision`.

### Syntax

```bash
yolo-report-ap \
  --model runs/my_run/weights/best.pt \
  --data data/converted/my_dataset/my_dataset.yaml \
  --split val
```

### Important arguments

| Option | Required | Description |
|---|:---:|---|
| `--model` | yes | path to checkpoint |
| `--data` | yes | path to dataset YAML |
| `--split` | no | `train`, `val`, or `test` |
| `--imgsz` | no | validation image size |
| `--batch` | no | validation batch size |
| `--device` | no | validation device |
| `--workers` | no | dataloader workers |
| `--conf` | no | confidence threshold override |
| `--iou` | no | IoU threshold override |
| `--output-dir` | no | report destination |
| `--top-k` | no | how many best/worst classes to print |
| `--verbose` | no | verbose validation output |

## `yolo-onnx-export`

Export a YOLO `.pt` checkpoint to ONNX.

### Syntax

```bash
yolo-onnx-export \
  --weights runs/my_run/weights/best.pt \
  --output deploy/onnx/model.export.fp32.onnx \
  --imgsz 1024
```

### Important arguments

| Option | Required | Description |
|---|:---:|---|
| `--weights` | yes | source YOLO `.pt` checkpoint |
| `--output` | no | output `.onnx` path |
| `--imgsz` | no | one integer or two integers: `H W` |
| `--batch` | no | export batch size |
| `--device` | no | export device, for example `cpu` or `0` |
| `--opset` | no | ONNX opset override |
| `--dynamic` | no | export dynamic axes |
| `--no-simplify` | no | disable graph simplification |

## `yolo-onnx-optimize`

Optimize an ONNX model for CPU or CUDA deployment.

### Syntax

```bash
yolo-onnx-optimize \
  --input deploy/onnx/model.export.fp32.onnx \
  --output-dir deploy/onnx/cpu \
  --target cpu
```

### Important arguments

| Option | Required | Description |
|---|:---:|---|
| `--input` | yes | source `.onnx` model |
| `--output-dir` | yes | directory for optimized artifacts |
| `--target` | no | `cpu` or `cuda` |
| `--graph-level` | no | ORT graph optimization level |
| `--tag` | no | optional artifact name tag |
| `--imgsz` | no | calibration image size as `N` or `H W` |
| `--no-preprocess` | no | skip ORT quantization preprocessing |
| `--int8` | no | generate INT8 QDQ artifacts for CPU |
| `--fp16` | no | generate FP16 artifacts for CUDA |
| `--calib-dir` | no | representative image directory for INT8 |
| `--calib-size` | no | max calibration images |
| `--calibration-method` | no | `minmax`, `entropy`, or `percentile` |
| `--u8u8` | no | use QUInt8 instead of QInt8 |
| `--reduce-range` | no | enable reduced-range quantization |
| `--no-per-channel` | no | disable per-channel weight quantization |
| `--keep-io-types` | no | keep original IO tensor types during FP16 conversion |

<table>
  <tr>
    <td><strong>⚠️ Warning</strong><br><code>yolo-onnx-optimize</code> requires an ONNX Runtime environment profile. Use <code>source scripts/setup_env.sh cpu</code> for CPU optimization or <code>source scripts/setup_env.sh cuda</code> for CUDA optimization.</td>
  </tr>
</table>

## `yolo-onnx-pipeline`

Export a YOLO checkpoint to ONNX and optimize it in one command.

### Syntax

```bash
yolo-onnx-pipeline \
  --weights runs/my_run/weights/best.pt \
  --artifact-dir deploy/onnx/cuda \
  --target cuda \
  --fp16
```

### Important arguments

| Option | Required | Description |
|---|:---:|---|
| `--weights` | yes | source YOLO `.pt` checkpoint |
| `--artifact-dir` | yes | directory where exported and optimized artifacts will be written |
| `--imgsz` | no | export/calibration image size as `N` or `H W` |
| `--batch` | no | export batch size |
| `--export-device` | no | device used during Ultralytics export |
| `--opset` | no | ONNX opset override |
| `--dynamic` | no | export dynamic axes |
| `--target` | no | `cpu` or `cuda` |
| `--graph-level` | no | ORT graph optimization level |
| `--int8` | no | generate INT8 artifacts |
| `--fp16` | no | generate FP16 artifacts |
| `--calib-dir` | no | calibration image directory for INT8 |

## Fallback Entry Points

If you are not using editable install yet, these file-based entrypoints still work:

- `python tools/convert_dataset_to_yolo.py`
- `python tools/print_yolo_dataset_stats.py`
- `python tools/prepare_yolo_dataset.py`
- `python tools/train.py`
- `python tools/report_ap.py`
- `python tools/onnx/export.py`
- `python tools/onnx/optimize.py`
- `python tools/onnx/pipeline.py`

---

**Next**
[`Dataset Guide`](DATASETS.md) · [`Training Guide`](TRAINING.md) · [`ONNX Guide`](ONNX.md) · [`Architecture`](ARCHITECTURE.md)
