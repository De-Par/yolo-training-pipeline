# 🧰 CLI Reference

**Navigation**
[`Home`](../README.md) · [`Datasets`](DATASETS.md) · [`Training`](TRAINING.md) · [`CLI`](CLI.md) · [`Architecture`](ARCHITECTURE.md)

This document describes the current public CLI surface.

## Contents

- [🚀 Public Commands](#-public-commands)
- [🔄 `yolo-convert-dataset`](#-yolo-convert-dataset)
- [📊 `yolo-print-stats`](#-yolo-print-stats)
- [🧱 `yolo-prepare-dataset`](#-yolo-prepare-dataset)
- [🏋️ `yolo-train`](#️-yolo-train)
- [📈 `yolo-report-ap`](#-yolo-report-ap)
- [🛟 Fallback Entry Points](#-fallback-entry-points)

<table>
  <tr>
    <td><strong>📝 Note</strong><br>Installed commands are the primary interface. <code>python tools/...</code> is fallback only.</td>
  </tr>
</table>

## 🚀 Public Commands

| Command | Purpose |
|---|---|
| `yolo-convert-dataset` | Convert raw annotations into a normalized YOLO-styled dataset |
| `yolo-print-stats` | Print detailed stats for a YOLO-styled dataset |
| `yolo-prepare-dataset` | Mutate a YOLO-styled dataset in place from a YAML recipe |
| `yolo-train` | Run Ultralytics training |
| `yolo-report-ap` | Validate a checkpoint and export per-class AP reports |

## 🔄 `yolo-convert-dataset`

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
| `--image-width-key` | width field name |
| `--image-height-key` | height field name |

## 📊 `yolo-print-stats`

Print detailed stats for a YOLO-styled dataset.

### Main outputs

- printed summary block
- printed per-class table
- `dataset_stats.json`
- `dataset_stats_train.png`
- `dataset_stats_val.png`

### Syntax

```bash
yolo-print-stats --dataset-dir data/converted/my_dataset
```

### Important arguments

| Option | Required | Description |
|---|:---:|---|
| `--dataset-dir` | yes | YOLO-styled dataset directory |
| `--output-json` | no | override JSON output path |
| `--output-png` | no | override base PNG path, writes `<stem>_train.png` and `<stem>_val.png` |

### What it reports

- image counts by split
- label file counts by split
- empty label counts
- missing and orphan label files
- total instance counts
- mean bbox geometry
- bbox area bins
- per-class train / val / total counts
- two split-specific mosaic PNG reports in the same style as the Ultralytics labels overview

<table>
  <tr>
    <td><strong>💡 Tip</strong><br>Use this command before every dataset mutation step. It is the analysis layer of the pipeline.</td>
  </tr>
</table>

## 🧱 `yolo-prepare-dataset`

Mutate a YOLO-styled dataset in place using a YAML recipe.

This command is optional. After conversion, the dataset is already trainable.

### Main effects

- reduce train/val splits by sampling
- drop classes
- rename classes
- merge several old classes into a new class name
- rewrite labels in place
- regenerate `classes.txt` and `<dataset>.yaml`
- write `prepare_report.json`

### Syntax

```bash
yolo-prepare-dataset \
  --dataset-dir data/converted/my_dataset \
  --recipe configs/prepare/yolo_dataset.yaml
```

### Important arguments

| Option | Required | Description |
|---|:---:|---|
| `--dataset-dir` | yes | YOLO-styled dataset to mutate |
| `--recipe` | yes | YAML recipe describing sampling and class transforms |

### Recipe structure

```yaml
dataset_name: my_dataset_prepared
sample_seed: 42
empty_policy: drop

sampling:
  train_fraction: 1.0
  val_fraction: 1.0

classes:
  keep: []
  drop: []
  remap:
    - name: footwear
      from: [shoe, boot, sandal]
```

Selector syntax:

- exact class names, for example `"shirt, blouse"`
- numeric class ids, for example `23`
- id ranges, for example `"30-35"`

If a class name contains commas, quote it exactly as written in `classes.txt`.

Example:

```yaml
classes:
  keep: ["shirt, blouse", 23, "30-35"]
  remap:
    - name: footwear
      from: [23]
    - name: tops
      from: ["shirt, blouse", "top, t-shirt, sweatshirt", 2, 3]
```

### Important behavior

- this command mutates the dataset in place
- it does not duplicate the dataset by default
- if you want the original converted dataset again, rerun `yolo-convert-dataset`
- a recipe that requests no changes is rejected on purpose

<table>
  <tr>
    <td><strong>⚠️ Warning</strong><br>Treat <code>yolo-prepare-dataset</code> as a destructive step. It is meant for intentional dataset surgery, not casual inspection.</td>
  </tr>
</table>

## 🏋️ `yolo-train`

Run Ultralytics training on any YOLO dataset YAML produced by conversion or preparation.

### Syntax

```bash
yolo-train --cfg configs/train/nvidia.yaml
```

### Important arguments

| Option | Required | Description |
|---|:---:|---|
| `--cfg` | no | YAML config with train parameters |
| `--model` | cfg/cli | model checkpoint or architecture |
| `--data` | cfg/cli | dataset YAML |
| `--epochs` | no | number of epochs |
| `--imgsz` | no | image size |
| `--batch` | no | batch size |
| `--device` | no | `cpu`, `0`, `cuda:0`, `mps` |
| `--name` | no | run directory name |

Everything else should live in the training YAML config, for example:

- `workers`
- `project`
- `seed`
- `amp`
- `cache`
- `compile`
- `val`
- `exist_ok`
- `verbose`
- any other Ultralytics train key

## 📈 `yolo-report-ap`

Run validation and export per-class AP metrics to CSV and JSON.

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

## 🛟 Fallback Entry Points

If you are not using editable install yet, these file-based entrypoints still work:

- `python tools/convert_dataset_to_yolo.py`
- `python tools/print_yolo_dataset_stats.py`
- `python tools/prepare_yolo_dataset.py`
- `python tools/train.py`
- `python tools/report_ap.py`

---

**Next**
[`Dataset Guide`](DATASETS.md) · [`Training Guide`](TRAINING.md) · [`Architecture`](ARCHITECTURE.md)
