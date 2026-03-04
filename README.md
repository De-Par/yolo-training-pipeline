# 🧠 YOLO Training Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#)
[![Ultralytics](https://img.shields.io/badge/Ultralytics%20trainer-YOLO-red)](#)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20WSL2%20%7C%20macOS-green)](#)

Practical scripts to **prepare datasets** and **train Ultralytics YOLO** models for **object detection** across multiple generations  
(e.g. **YOLOv3 / YOLOv5 / YOLOv8 / YOLO11 / YOLO26**).

This repo is intentionally split into **two independent steps**:

1) **Prepare dataset** → convert annotations to YOLO format + generate `data.yaml`  
2) **Train model** → run `ultralytics.YOLO(...).train()` using CLI args and/or YAML config


## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [Environment setup](#environment-setup)
- [Download helpers](#download-helpers)
    - [Download YOLO weights](#download-yolo-weights)
    - [Download datasets](#download-datasets)
- [Prepare datasets](#prepare-datasets)
    - [Fashionpedia](#prepare-fashionpedia)
    - [DeepFashion2](#prepare-deepfashion2)
    - [Custom COCO](#prepare-custom-coco)
    - [Sampling subsets](#sampling-subsets)
    - [Output format](#output-format)
    - [CLI reference: prepare_dataset.py](#cli-reference-prepare_datasetpy)
- [Train models](#train-models)
    - [Train with CLI](#train-with-cli)
    - [Train with YAML config](#train-with-yaml-config)
    - [CLI reference: train_yolo.py](#cli-reference-train_yolopy)
- [Performance notes](#performance-notes)
- [Troubleshooting](#troubleshooting)


## Features

- ✅ Works with **multiple YOLO generations** (not tied to a single checkpoint family)
- ✅ Prepare **Fashionpedia**, **DeepFashion2**, and **custom COCO**
- ✅ Training supports **YAML configs** + **CLI overrides**
- ✅ Conversion reporting: missing images / ambiguous basenames (COCO modes)


## Quick start

```bash
# 1) Create venv & install dependencies
source scripts/setup_env.sh

# 2) Download a model
./scripts/download_models.sh --generation v26 --task detect --size n

# 3) Download a dataset 
./scripts/download_datasets.sh --dataset fashionpedia

# 4) Prepare dataset
python tools/prepare_dataset.py \
    --dataset fashionpedia \
    --raw-root data/raw/fashionpedia \
    --workdir data/processed \
    --train-fraction 0.1 \
    --val-fraction 0.2

# 5) Train model (GPU)
python tools/train_yolo.py \
    --data data/processed/fashionpedia/fashionpedia.yaml \
    --model models/YOLOv26/yolo26n.pt \
    --device 0 --amp --cache disk --batch 32 --workers 12
```


## Environment setup

- Python: use `python3 -m venv .venv`
- Install deps: `pip install -r requirements.txt`
- Recommended: run everything inside the repo root


## Download helpers

### Download YOLO weights

Use `scripts/download_models.sh`:

```bash
# Download all weights for YOLOv26 -> models/YOLOv26/
./scripts/download_models.sh --generation v26

# Only detection models
./scripts/download_models.sh --generation v26 --task detect

# Only selected sizes
./scripts/download_models.sh --generation v26 --size n,s

# Preview without downloading
./scripts/download_models.sh --generation v26 --dry-run
```

Model size suffixes:

| Suffix | Meaning |
|---|---|
| `n` | nano |
| `s` | small |
| `m` | medium |
| `l` | large |
| `x` | extra-large |

### Download datasets

#### Fashionpedia (fully automated)

```bash
./scripts/download_datasets.sh --dataset fashionpedia
```

#### DeepFashion2 (semi-automated)

```bash
mkdir -p data/raw/deepfashion2/downloads
./scripts/download_datasets.sh --dataset deepfashion2
```

Optionally provide URLs/password:

```bash
export DEEPFASHION2_URLS="https://.../train.zip https://.../validation.zip"
export DEEPFASHION2_PASSWORD="<password_if_needed>"
./scripts/download_datasets.sh --dataset deepfashion2
```


## Prepare datasets

The prepare step converts annotations and generates a YOLO-ready folder with:

- `images/train`, `images/val`
- `labels/train`, `labels/val`
- `classes.txt`
- `<dataset_name>.yaml` (Ultralytics `data.yaml`)

### Prepare Fashionpedia

Expected raw layout:

```text
<RAW_ROOT>/
  train/
    images/
    annotations.json
  val/
    images/
    annotations.json
```

Command:

```bash
python tools/prepare_dataset.py \
    --dataset fashionpedia \
    --raw-root data/raw/fashionpedia \
    --workdir data/processed
```

### Prepare DeepFashion2

Expected raw layout:

```text
<RAW_ROOT>/
  train/
    image/
    annos/
  validation/
    image/
    annos/
```

Command:

```bash
python tools/prepare_dataset.py \
    --dataset deepfashion2 \
    --raw-root data/raw/deepfashion2 \
    --workdir data/processed
```

### Prepare custom COCO

Use this if you have COCO JSON (`images`, `annotations`, `categories`):

```bash
python tools/prepare_dataset.py \
    --dataset custom-coco \
    --custom-name my_dataset \
    --train-images-dir /path/to/train/images \
    --train-annotations /path/to/train_annotations.json \
    --val-images-dir /path/to/val/images \
    --val-annotations /path/to/val_annotations.json \
    --workdir data/processed
```

COCO JSON requirements:
- `images`: `id`, `file_name`, `width`, `height`
- `annotations`: `image_id`, `category_id`, `bbox` (XYWH)
- `categories`: `id`, `name`

Converter behavior:
- remaps non-contiguous category IDs to contiguous YOLO IDs
- tries recursive basename lookup for missing `images_dir/file_name`
- skips ambiguous duplicate basenames and reports them

### Sampling subsets

For quick experiments you can prepare only a fraction of train/val:

```bash
python tools/prepare_dataset.py \
    --dataset fashionpedia \
    --raw-root data/raw/fashionpedia \
    --workdir data/processed \
    --train-fraction 0.1 \
    --val-fraction 0.2 \
    --sample-seed 42
```

### Output format

```text
data/processed/<dataset_name>/
  images/train/
  images/val/
  labels/train/
  labels/val/
  classes.txt
  <dataset_name>.yaml
```

YOLO label row format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Rules:
- normalized coordinates `[0, 1]`
- `class_id` is zero-based
- label stem must match image stem


## CLI reference: prepare_dataset.py

> Run `python tools/prepare_dataset.py --help` for the authoritative list.

| Argument | Type | Required | Default | Description |
|---|---:|:---:|---:|---|
| `--dataset` | enum | ✅ | – | `fashionpedia` / `deepfashion2` / `custom-coco` |
| `--raw-root` | path | ⚠️ | – | Required for `fashionpedia` and `deepfashion2` |
| `--workdir` | path | ❌ | `data/processed` | Output root for processed dataset |
| `--link-mode` | enum | ❌ | `symlink` | `symlink` or `copy` for images |
| `--train-fraction` | float | ❌ | `1.0` | Fraction of train split to prepare `(0,1]` |
| `--val-fraction` | float | ❌ | `1.0` | Fraction of val split to prepare `(0,1]` |
| `--sample-seed` | int | ❌ | `42` | Deterministic sampling seed |
| `--custom-name` | str | ❌ | `custom` | Dataset folder name for `custom-coco` |
| `--train-images-dir` | path | ⚠️ | – | `custom-coco`: train images dir |
| `--train-annotations` | path | ⚠️ | – | `custom-coco`: train COCO JSON |
| `--val-images-dir` | path | ⚠️ | – | `custom-coco`: val images dir |
| `--val-annotations` | path | ⚠️ | – | `custom-coco`: val COCO JSON |

⚠️ “Required” depends on `--dataset` mode.


## Train models

Training is handled by `tools/train_yolo.py`.

It supports:
- **CLI-only**
- **YAML config (`--cfg`)**
- **YAML + CLI overrides**
- warnings for **unsupported YAML keys** (ignored)

### Train with CLI

```bash
python tools/train_yolo.py \
    --data data/processed/fashionpedia/fashionpedia.yaml \
    --model models/YOLOv26/yolo26n.pt \
    --device 0 --amp --batch 32 --workers 12
```

### Train with YAML config

Minimal config (`configs/config.example.yaml`) can contain `model:` and `data:`:

```bash
python tools/train_yolo.py --cfg configs/config.example.yaml
```

### YAML + CLI override

```bash
python tools/train_yolo.py \
    --cfg configs/config.example.yaml \
    --batch 48 --device 0 --cache ram --compile
```


## CLI reference: train_yolo.py

> Run `python tools/train_yolo.py --help` for the authoritative list.

| Argument | Type | Required | Default | Description |
|---|---:|:---:|---:|---|
| `--cfg` | path | ❌ | – | YAML config with train parameters |
| `--data` | path | ⚠️ | – | Dataset `data.yaml` (can be taken from YAML cfg) |
| `--model` | path | ⚠️ | – | Model `.pt`/`.yaml` (can be taken from YAML cfg) |
| `--epochs` | int | ❌ | `100` | Training epochs |
| `--imgsz` | int | ❌ | `640` | Image size |
| `--batch` | int | ❌ | `32` | Batch size |
| `--device` | str/int | ❌ | auto | `0`, `cpu`, `cuda:0`, etc. |
| `--workers` | int | ❌ | `12` | Dataloader workers |
| `--project` | str | ❌ | `runs/train` | Output base directory |
| `--name` | str | ❌ | `yolo-run` | Run name |
| `--seed` | int | ❌ | `42` | Random seed |
| `--amp` | flag | ❌ | YAML/true | Enable AMP |
| `--no-amp` | flag | ❌ | – | Disable AMP |
| `--cache` | enum | ❌ | – | `ram` / `disk` |
| `--compile` | flag | ❌ | `false` | Enable `torch.compile` (if supported) |
| `--no-val` | flag | ❌ | – | Disable validation |
| `--exist-ok` | flag | ❌ | `false` | Don’t increment run folder name |
| `--verbose` | flag | ❌ | `false` | Extra logs |

⚠️ `--data` and `--model` are required **unless** provided in `--cfg`.


## Performance notes

### WSL / disks

- Avoid training from `/mnt/c/...` (NTFS bridge overhead).
- Prefer:
    - code in `~/...` (Linux FS), and
    - datasets on `/mnt/e/...` (SSD) if too large for Linux FS.

Suggested layout:

```text
~/projects/yolo-train
/mnt/e/datasets/...
/mnt/e/models/...
```

### NVIDIA GPU

Recommended baseline:

```yaml
device: 0
amp: true
cache: ram
batch: 32
workers: 12
imgsz: 640
epochs: 100
```

If you hit OOM:
1) reduce `batch`
2) then reduce `imgsz`
3) then turn off `cache: ram`


## Troubleshooting

- **Prepared dataset has zero train/val images**
    - Check dataset paths and expected raw structure.
- **Missing images warnings in COCO mode**
    - `file_name` in JSON may not match files on disk.
- **OOM / CUDA out of memory**
    - Reduce `batch`, then `imgsz`.
- **Config key ignored**
    - `train_yolo.py` prints warnings if YAML contains keys unsupported by your Ultralytics version.
