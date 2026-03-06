# 🗂️ Dataset Guide

**Navigation**
[`Home`](../README.md) · [`Datasets`](DATASETS.md) · [`Training`](TRAINING.md) · [`CLI`](CLI.md) · [`Architecture`](ARCHITECTURE.md)

This guide documents the dataset side of the pipeline.

## Contents

- [🧱 Stage Order](#stage-order)
- [🔄 Stage 3: Convert Raw Data](#stage-3-convert-raw-data)
- [📊 Stage 4: Print YOLO Dataset Stats](#stage-4-print-yolo-dataset-stats)
- [🧪 Stage 5: Prepare YOLO Dataset](#stage-5-prepare-yolo-dataset)
- [🧾 Dataset Layouts](#dataset-layouts)
- [🧪 Example Scenarios](#example-scenarios)
- [💡 Practical Advice](#practical-advice)

<table>
  <tr>
    <td><strong>📝 Note</strong><br><code>scripts/</code> may contain demo flows, but the canonical dataset pipeline is the <code>yolo-*</code> CLI.</td>
  </tr>
</table>

## 🧱 Stage Order

The intended dataset flow is:

1. download or organize raw data
2. `yolo-convert-dataset`
3. `yolo-print-stats`
4. optionally `yolo-prepare-dataset`
5. `yolo-print-stats` again if the dataset was mutated

Why this matters:

- conversion should be a faithful schema translation
- stats should drive decisions
- preparation should only apply explicit dataset changes

## 🔄 Stage 3: Convert Raw Data

Use `yolo-convert-dataset` to normalize raw annotations into a YOLO-styled dataset.

Current supported raw adapters:

- `coco-detection`
- `per-image-json`

### `coco-detection`

Use this when your raw dataset provides:

- train image directory
- val image directory
- COCO JSON for train
- COCO JSON for val

Example:

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

### `per-image-json`

Use this when each image has a separate JSON file and category ids are resolved through a class-name file.

Example:

```bash
yolo-convert-dataset \
  --dataset-name my_dataset \
  --input-format per-image-json \
  --train-images-dir data/raw/my_dataset/train/images \
  --train-annotations data/raw/my_dataset/train/annos \
  --val-images-dir data/raw/my_dataset/val/images \
  --val-annotations data/raw/my_dataset/val/annos \
  --class-names-file data/raw/my_dataset/classes.txt \
  --bbox-format xyxy \
  --clean
```

### Output after conversion

```text
data/converted/<dataset_name>/
├── classes.txt
├── conversion_report_train.json
├── conversion_report_val.json
├── <dataset_name>.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Important point:

- after conversion the dataset is already trainable
- preparation is optional, not mandatory

## 📊 Stage 4: Print YOLO Dataset Stats

Use `yolo-print-stats` immediately after conversion.

Example:

```bash
yolo-print-stats --dataset-dir data/converted/my_dataset
```

This step reports:

- image counts
- label counts
- empty labels
- missing or orphan labels
- mean bbox geometry
- object size bins
- per-class counts for every detected split plus totals

It also writes:

```text
data/converted/my_dataset/dataset_stats.json
```

and:

```text
data/converted/my_dataset/dataset_stats_train.png
data/converted/my_dataset/dataset_stats_val.png
data/converted/my_dataset/dataset_stats_test.png   # if test exists
```

This is the report you should use when deciding whether to:

- reduce splits
- drop classes
- merge noisy classes
- rename classes

## 🧪 Stage 5: Prepare YOLO Dataset

Use `yolo-prepare-dataset` only when you intentionally want to change the YOLO-styled dataset.

This step mutates the dataset in place and is driven by a YAML recipe.

The tracked starter recipe is:

- [`configs/prepare/yolo_dataset.yaml`](../configs/prepare/yolo_dataset.yaml)

Edit the recipe first. The tracked file is intentionally no-op by default and is rejected until you request a real split or class change.

Example:

```bash
yolo-prepare-dataset \
  --dataset-dir data/converted/my_dataset \
  --recipe configs/prepare/yolo_dataset.yaml
```

### What preparation can change

- split mode
- `train` / `val` / optional `test` fractions
- per-class minimum instances for `val` / `test`
- class removal
- class renaming
- class merging into a new name
- empty-image behavior after label rewriting

If the recipe requests no changes at all, the command fails fast instead of doing a meaningless in-place rewrite.

### Recipe example

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

- `keep_existing`: preserve the current split boundaries
- `sample_existing`: shrink the current splits independently
- `resplit_combined_random`: merge the current splits and rebuild `train` / `val` / optional `test`
- `resplit_combined_by_instances`: merge the current splits and guarantee minimum instance coverage in `val` / `test` before filling the target fractions

For `per_class_min_*`, this project uses compact YAML flow-style mappings in examples, for example `{23: 50, "shirt, blouse": 30}`.

Selector syntax:

- exact class names, for example `"shirt, blouse"`
- numeric class ids, for example `23`
- id ranges, for example `"30-35"`

If a class name contains commas, quote it exactly as written in `classes.txt`.

Example:

```yaml
split:
  mode: resplit_combined_by_instances
  seed: 42
  train_fraction: 0.8
  val_fraction: 0.1
  test_fraction: 0.1
  min_val_instances_per_class: 20
  per_class_min_test_instances: {23: 50}

classes:
  keep: ["shirt, blouse", 23, "30-35"]
  remap:
    - name: footwear
      from: [23]
    - name: tops
      from: ["shirt, blouse", "top, t-shirt, sweatshirt", 2, 3]
```

### Why it is in-place

The command rewrites the existing YOLO-styled dataset instead of creating a second copy.

This saves disk space, but it is destructive.

If you need the original converted dataset again, rerun `yolo-convert-dataset`.

<table>
  <tr>
    <td><strong>⚠️ Warning</strong><br>Do not treat <code>yolo-prepare-dataset</code> as a harmless formatting step. It changes the dataset itself.</td>
  </tr>
</table>

## 🧾 Dataset Layouts

### Converted dataset layout

```text
data/converted/<dataset_name>/
├── classes.txt
├── conversion_report_train.json
├── conversion_report_val.json
├── dataset_stats.json
├── <dataset_name>.yaml
├── images/
└── labels/
```

### After in-place preparation

The same directory is mutated and additionally contains:

```text
prepare_report.json
```

`classes.txt` and `<dataset>.yaml` are regenerated to match the new label space.

## 🧪 Example Scenarios

### Scenario 1: train directly after conversion

```bash
yolo-convert-dataset \
  --dataset-name warehouse_items \
  --input-format coco-detection \
  --train-images-dir data/raw/warehouse_items/train/images \
  --train-annotations data/raw/warehouse_items/train/annotations.json \
  --val-images-dir data/raw/warehouse_items/val/images \
  --val-annotations data/raw/warehouse_items/val/annotations.json \
  --clean

yolo-print-stats --dataset-dir data/converted/warehouse_items

yolo-train --data data/converted/warehouse_items/warehouse_items.yaml --model models/YOLOv26/yolo26n.pt
```

### Scenario 2: merge classes before training

```bash
yolo-print-stats --dataset-dir data/converted/warehouse_items

yolo-prepare-dataset \
  --dataset-dir data/converted/warehouse_items \
  --recipe configs/prepare/yolo_dataset.yaml

yolo-print-stats --dataset-dir data/converted/warehouse_items
```

### Scenario 3: create a smoke dataset by shrinking splits

```yaml
split:
  mode: sample_existing
  train_fraction: 0.01
  val_fraction: 0.1
```

Apply with:

```bash
yolo-prepare-dataset \
  --dataset-dir data/converted/warehouse_items \
  --recipe configs/prepare/yolo_dataset.yaml
```

## 💡 Practical Advice

- Convert first, mutate later.
- Print stats before every destructive step.
- Keep the prepare recipe under version control.
- If a dataset mutation was a mistake, reconvert instead of trying to undo it manually.
- If you need a new raw schema, extend `convert_dataset_to_yolo`, not `prepare_yolo_dataset`.

---

**Next**
[`CLI Reference`](CLI.md) · [`Training Guide`](TRAINING.md) · [`Architecture`](ARCHITECTURE.md)
