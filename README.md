# 🚀 YOLO Training Pipeline

A practical, end-to-end training pipeline for **object detection** with Ultralytics YOLO models across multiple generations (for example: **YOLOv3, YOLOv5, YOLOv8, YOLO11, YOLO26**).

This repository helps you:
- 📦 Download model weights by generation
- 🧰 Prepare Fashionpedia / DeepFashion2 / custom COCO datasets
- 🧾 Convert annotations to YOLO format
- 🏋️ Train models using CLI or YAML config


## 🧭 Navigation

- [Quick Start in 1 Minute](#quick-start)
- [Download YOLO Weights by Generation](#download-models)
- [Dataset Download Helpers](#dataset-download)
- [Main Pipeline (`tools/run_pipeline.py`)](#main-pipeline)
- [Fashionpedia Mode](#fashionpedia-mode)
- [DeepFashion2 Mode](#deepfashion2-mode)
- [Custom COCO Mode](#custom-coco-mode)
- [Prepare-Only Mode (No Training)](#prepare-only)
- [Universal Training Script (`tools/train_yolo.py`)](#train-yolo)
- [Example Configs](#example-configs)
- [Output Format Produced by Pipeline](#output-format)
- [Custom YOLO Dataset Requirements (Direct Training)](#custom-yolo-requirements)
- [Make Targets](#make-targets)
- [Validation](#validation)
- [Troubleshooting](#troubleshooting)


## ✨ Features

- ✅ Supports **multiple YOLO generations** (not tied to one model family)
- ✅ Supports **Fashionpedia**, **DeepFashion2**, and **custom COCO** pipelines
- ✅ Supports **custom YOLO-ready datasets** (direct training)
- ✅ Includes **prepare-only mode** for safe dataset validation
- ✅ Includes **config-driven training** (`--cfg`) + CLI overrides


## 🗂️ Project Structure

```text
scripts/
  download_datasets.sh        # Download/unpack helpers for raw datasets
  download_models.sh          # Download YOLO weights by generation

tools/
  convert_coco_to_yolo.py
  convert_deepfashion2_to_yolo.py
  run_pipeline.py             # Prepare dataset + optional training
  train_yolo.py               # Universal trainer for all generations

configs/
  fashionpedia_train.example.yaml
  deepfashion2_train.example.yaml
  custom_train.example.yaml

models/
  YOLOv*/...

data/
  raw/
  processed/
```

<a id="quick-start"></a>
## ⚡ Quick Start in 1 Minute

```bash
# 0) Create venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 1) Install dependencies
python -m pip install -r requirements.txt

# 2) Download YOLO weights (example: YOLOv26)
./scripts/download_models.sh --generation v26 --task detect --size n

# 3) Download Fashionpedia raw data
./scripts/download_datasets.sh --dataset fashionpedia --out-dir data/raw

# 4) Prepare dataset (safe check, no training yet)
python tools/run_pipeline.py \
  --dataset fashionpedia \
  --raw-root data/raw/fashionpedia \
  --workdir data/processed \
  --prepare-only

# 5) Start training
python tools/train_yolo.py \
  --data data/processed/fashionpedia/fashionpedia.yaml \
  --model models/YOLOv26/yolo26n.pt \
  --epochs 100 --imgsz 640 --batch 16 --device 0
```


<a id="download-models"></a>
## 📥 Download YOLO Weights by Generation

Use `scripts/download_models.sh`:

```bash
# Download all weights for YOLOv26 -> models/YOLOv26/
./scripts/download_models.sh --generation v26

# Detection-only models
./scripts/download_models.sh --generation v26 --task detect

# Only selected size variants
./scripts/download_models.sh --generation v26 --size n,s

# Preview without downloading
./scripts/download_models.sh --generation v26 --dry-run
```

### Model size suffixes

| Suffix | Meaning |
|---|---|
| `n` | nano |
| `s` | small |
| `m` | medium |
| `l` | large |
| `x` | extra-large |


<a id="dataset-download"></a>
## 🧵 Dataset Download Helpers

### Fashionpedia (fully automated)

```bash
./scripts/download_datasets.sh --dataset fashionpedia --out-dir data/raw
```

### DeepFashion2 (semi-automated)

```bash
# Option A: place archives manually
mkdir -p data/raw/deepfashion2/downloads
./scripts/download_datasets.sh --dataset deepfashion2 --out-dir data/raw

# Option B: provide direct archive URLs
export DEEPFASHION2_URLS="https://.../train.zip https://.../validation.zip"
export DEEPFASHION2_PASSWORD="<password_if_needed>"
./scripts/download_datasets.sh --dataset deepfashion2 --out-dir data/raw
```


<a id="main-pipeline"></a>
## 🧠 Main Pipeline (`tools/run_pipeline.py`)

Supported modes:
- `fashionpedia`
- `deepfashion2`
- `custom-coco`

The script prepares YOLO dataset structure and generates `data.yaml`. It can also start training unless `--prepare-only` is used.


<a id="fashionpedia-mode"></a>
## 👗 Fashionpedia Mode

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

Run:

```bash
python tools/run_pipeline.py \
  --dataset fashionpedia \
  --raw-root data/raw/fashionpedia \
  --workdir data/processed \
  --model models/YOLOv26/yolo26n.pt
```


<a id="deepfashion2-mode"></a>
## 👕 DeepFashion2 Mode

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

Run:

```bash
python tools/run_pipeline.py \
  --dataset deepfashion2 \
  --raw-root data/raw/deepfashion2 \
  --workdir data/processed \
  --model models/YOLOv26/yolo26n.pt
```


<a id="custom-coco-mode"></a>
## 🧩 Custom COCO Mode

Use this when your dataset is in COCO JSON format.

Run:

```bash
python tools/run_pipeline.py \
  --dataset custom-coco \
  --custom-name my_dataset \
  --train-images-dir /path/to/train/images \
  --train-annotations /path/to/train_annotations.json \
  --val-images-dir /path/to/val/images \
  --val-annotations /path/to/val_annotations.json \
  --workdir data/processed \
  --model models/YOLOv11/yolo11n.pt
```

### COCO annotation requirements

Your JSON must include:
- `images`: `id`, `file_name`, `width`, `height`
- `annotations`: `image_id`, `category_id`, `bbox` (XYWH)
- `categories`: `id`, `name`

Notes:
- Non-contiguous source category IDs are remapped to contiguous YOLO class IDs.
- If `images_dir/file_name` is missing, converter tries recursive basename lookup.
- Ambiguous duplicate basenames are skipped and reported.


<a id="prepare-only"></a>
## 🧪 Prepare-Only Mode (No Training)

Use this to validate dataset conversion safely:

```bash
python tools/run_pipeline.py \
  --dataset custom-coco \
  --custom-name my_dataset \
  --train-images-dir /path/to/train/images \
  --train-annotations /path/to/train_annotations.json \
  --val-images-dir /path/to/val/images \
  --val-annotations /path/to/val_annotations.json \
  --workdir data/processed \
  --prepare-only
```


<a id="train-yolo"></a>
## 🏋️ Universal Training Script (`tools/train_yolo.py`)

### Option A: CLI-only

```bash
python tools/train_yolo.py \
  --data data/processed/fashionpedia/fashionpedia.yaml \
  --model models/YOLOv26/yolo26n.pt \
  --epochs 100 --imgsz 640 --batch 16 --device 0
```

### Option B: Config file (`--cfg`)

```bash
python tools/train_yolo.py --cfg configs/fashionpedia_train.example.yaml
```

### Option C: Config + CLI override

```bash
python tools/train_yolo.py \
  --cfg configs/fashionpedia_train.example.yaml \
  --model models/YOLOv11/yolo11s.pt \
  --epochs 150
```


<a id="example-configs"></a>
## 📄 Example Configs

Available templates:
- `configs/fashionpedia_train.example.yaml`
- `configs/deepfashion2_train.example.yaml`
- `configs/custom_train.example.yaml`

Each template includes readable comments for every field.


<a id="output-format"></a>
## 🧱 Output Format Produced by Pipeline

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
- Coordinates must be normalized to `[0, 1]`
- `class_id` is zero-based
- Label stem must match image stem


<a id="custom-yolo-requirements"></a>
## 🧭 Custom YOLO Dataset Requirements (Direct Training)

If your dataset is already in YOLO format, you can skip conversion and train directly.

Expected structure:

```text
<DATASET_ROOT>/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

`data.yaml` example:

```yaml
path: /absolute/path/to/<DATASET_ROOT>
train: images/train
val: images/val
names:
  0: class_0
  1: class_1
```


<a id="make-targets"></a>
## 🛠️ Make Targets

```bash
make install
make check
make download-fashionpedia
make download-deepfashion2
make download-datasets
make fashionpedia-pipeline RAW_ROOT=data/raw/fashionpedia MODEL=models/YOLOv26/yolo26n.pt
make deepfashion2-pipeline RAW_ROOT=data/raw/deepfashion2 MODEL=models/YOLOv26/yolo26n.pt
```


<a id="validation"></a>
## ✅ Validation

```bash
make check
```

Checks include:
- Shell script syntax
- Python syntax
- CLI availability for all tools


<a id="troubleshooting"></a>
## 🧯 Troubleshooting

- **`No models found for generation ...`**
  - Try `--release-tag` or run again later.
- **OOM / CUDA out of memory**
  - Reduce `batch` and/or `imgsz`.
- **Prepared dataset has zero train/val images**
  - Verify raw paths and annotation/image layout.
- **Missing images warnings in COCO mode**
  - Check `file_name` paths in JSON vs actual files.


## 📌 Notes

- This repo is focused on **detection** workflows.
- Model compatibility depends on installed `ultralytics` version and checkpoint format.
