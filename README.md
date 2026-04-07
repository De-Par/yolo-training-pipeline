# 🧠 YOLO Training Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://docs.python.org/)
[![Ultralytics](https://img.shields.io/badge/YOLO-Ultralytics%20trainer-ef4444)](https://docs.ultralytics.com/)
[![ONNX](https://img.shields.io/badge/ONNX-export%20%26%20optimize-f59e0b)](https://onnxruntime.ai/docs/)
[![Status](https://img.shields.io/badge/Status-generic%20pipeline-14b8a6)](#)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20WSL2%20%7C%20macOS-64748b)](#)


A generic pipeline for converting raw detection datasets into YOLO format, inspecting and optionally mutating YOLO-styled datasets, training Ultralytics YOLO models, exporting per-class AP reports, and preparing ONNX artifacts for deployment.

---

**Navigation:**
[Home](#) · [Datasets](docs/DATASETS.md) · [Training](docs/TRAINING.md) · [Benching](docs/BENCH.md) · [ONNX](docs/ONNX.md) · [CLI](docs/CLI.md) · [Architecture](docs/ARCHITECTURE.md)


```mermaid
flowchart TB
  %% Stage 1
    subgraph S1[ ]
      direction TB
      L1[Stage 1: Preparation]:::stage
      A0(0. Setup environment) --> A1(1. Download model) --> A2(2. Download dataset)
    end

  %% Stage 2
  subgraph S2[ ]
    direction TB
    L2[Stage 2: Dataset preprocessing]:::stage
    B3(3. Convert to YOLO format)
    B4(4. Modify)
    B5(5. Collect statistics)

    B3 -. optional .-> B4
    B3 --> B5
    B5 -. not satisfied? .-> B3
    B4 .-> B5
  end

  %% Stage 3
  subgraph S3[ ]
    direction TB
    L3[Stage 3: Model training]:::stage
    C6([6. Train])
  end

  %% Stage 4
  subgraph S4[ ]
    direction TB
    L4[Stage 4: Model evaluation and exporting]:::stage
    D7(7. AP report)
    D8(8. Export to ONNX)
    D9(9. Quality / performance bench)

    D7 -. optional .-> D8
    D7 --> D9
    D8 .-> D9
  end

  %% Stage 5
  subgraph S5[ ]
    direction TB
    L5[Stage 5: Deployment]:::stage
    D10{{10. Deploy}}
    E11(CPU inference)
    E12(GPU inference)
    E13(Demo app)

    D10 --> E11
    D10 --> E12
    D10 -. built-in .-> E13
  end

  %% Main pipeline
  A2 --> B3
  B5 --> C6
  C6 --> D7
  D9 --> D10

  %% Styles
  classDef stage fill:#eef2ff,stroke:#6306f1,color:#312e81,stroke-width:1px
  classDef prep fill:#f8fafc,stroke:#475569,color:#0f172a,stroke-width:1px
  classDef process fill:#fff7ed,stroke:#f59e0b,color:#7c2d12,stroke-width:1px
  classDef core fill:#f0fdf4,stroke:#16a34a,color:#14532d,stroke-width:1.2px
  classDef eval fill:#eff6ff,stroke:#2563eb,color:#1e3a8a,stroke-width:1px
  classDef deploy fill:#ecfeff,stroke:#0891b2,color:#164e63,stroke-width:1.2px
  classDef demo fill:#fdf4ff,stroke:#c026d3,color:#701a75,stroke-width:1px

  class A0,A1,A2 prep
  class B3,B4,B5 process
  class C6 core
  class D7,D8,D9 eval
  class D10,E11,E12 deploy
  class E13 demo
  class L1,L2,L3,L4,L5 stage

  %% Stage box styling
  style S1 fill:#ffffff,stroke:#cbd5e1,stroke-width:1px
  style S2 fill:#ffffff,stroke:#cbd5e1,stroke-width:1px
  style S3 fill:#ffffff,stroke:#cbd5e1,stroke-width:1px
  style S4 fill:#ffffff,stroke:#cbd5e1,stroke-width:1px
  style S5 fill:#ffffff,stroke:#cbd5e1,stroke-width:1px
```


## Documentation Map

| I want to... | Go here |
|---|---|
| understand dataset conversion and prepare rules | [Datasets](docs/DATASETS.md) |
| run and tune training | [Training](docs/TRAINING.md) |
| export and optimize ONNX artifacts | [ONNX](docs/ONNX.md) |
| benchmark latency, FPS, and holdout quality | [Benching](docs/BENCH.md) |
| see exact commands and flags | [CLI](docs/CLI.md) |
| understand code layout | [Architecture](docs/ARCHITECTURE.md) |
| inspect the tracked train config example | [nvidia.example.yaml](configs/train/nvidia.example.yaml) |
| inspect the tracked prepare recipe example | [prepare.example.yaml](configs/prepare/prepare.example.yaml) |
| inspect the tracked benchmark config examples | [cpu.example.yaml](configs/bench/cpu.example.yaml), [gpu.example.yaml](configs/bench/gpu.example.yaml) |

## Public CLI Surface

Installable commands from [pyproject.toml](pyproject.toml):

| Command | Purpose |
|---|---|
| `yolo-convert-dataset` | Convert raw annotations into a YOLO-styled dataset |
| `yolo-print-stats` | Print detailed stats for a YOLO-styled dataset |
| `yolo-prepare-dataset` | Mutate a YOLO-styled dataset in place using a YAML recipe |
| `yolo-train` | Launch Ultralytics training |
| `yolo-report-ap` | Export per-class AP metrics |
| `yolo-onnx-export` | Export a YOLO `.pt` checkpoint to ONNX |
| `yolo-onnx-optimize` | Optimize an ONNX model for CPU or CUDA runtime |
| `yolo-onnx-pipeline` | Export and optimize in one command |
| `yolo-benchmark-report` | Measure inference speed vs hardware points and render a PNG quality/speed report |

The benchmark config uses `dataset.source` as the default benchmark source, usually `test`. Optional `dataset.speed` and `dataset.quality` blocks override that source only when you explicitly need different splits or direct image/label paths. If `dataset.quality` uses a dataset YAML and does not set `split`, it also defaults to `test`.

Supported raw input adapters today:

- `coco-detection`
- `per-image-json`

Demo download scripts:

- `scripts/download_fashionpedia.sh` for the public Fashionpedia demo flow
- `scripts/download_deepfashion2.sh` for the official DeepFashion2 download and organize flow
  - `--source official` downloads the official Google Drive folder via `gdown`
  - `--source local` only unpacks archives already placed in `data/raw/deepfashion2/downloads/`
  - protected nested zip files require `DEEPFASHION2_ARCHIVE_PASSWORD` or `--archive-password`
  - the script also writes `data/raw/deepfashion2/classes.txt` for the `per-image-json` converter

<table>
  <tr>
    <td><strong>📝 Note</strong><br>The pipeline is dataset-generic. Dataset-specific names such as Fashionpedia appear only in examples and demo scripts under <code>scripts/</code>.</td>
  </tr>
</table>

## Project Layout

```text
.
├── configs/
│   ├── bench/
│   │   └── *.example.yaml
│   ├── prepare/
│   │   └── *.example.yaml
│   └── train/
│       └── *.example.yaml
├── core/
|   ├── bench/
│   ├── common/
│   ├── datasets/
│   ├── onnx/
│   └── training/
├── docs/
│   ├── ARCHITECTURE.md
|   ├── BENCH.md
│   ├── CLI.md
│   ├── DATASETS.md
│   ├── ONNX.md
│   └── TRAINING.md
├── scripts/
│   ├── download_deepfashion2.sh
│   ├── download_fashionpedia.sh
│   ├── download_yolo_models.sh
│   └── setup_env.sh
├── tools/
│   ├── convert_dataset_to_yolo.py
│   ├── onnx/
│   ├── prepare_yolo_dataset.py
│   ├── print_yolo_dataset_stats.py
│   ├── report_ap.py
|   ├── yolo_bench_report.py
│   └── train.py
└── pyproject.toml
```

## Quick Start

This is a full demo flow on `Fashionpedia`, from environment setup to the first training run and optional ONNX export.

### 1. Create the environment

Pick the environment profile that matches your next steps:

- `base`: dataset pipeline, training, report export, ONNX export only
- `cpu`: everything in `base` plus ONNX Runtime CPU optimization and INT8 quantization
- `cuda`: everything in `base` plus ONNX Runtime CUDA optimization and FP16 conversion

Examples:

```bash
source scripts/setup_env.sh base
source scripts/setup_env.sh cpu
source scripts/setup_env.sh cuda
```

What `setup_env.sh` does:

- creates `.venv`
- upgrades `pip`, `setuptools`, and `wheel`
- installs the project from `pyproject.toml`
- activates `.venv` in the current shell
- exposes the `yolo-*` commands through editable install

Quick sanity check:

```bash
yolo-convert-dataset --help
yolo-print-stats --help
yolo-prepare-dataset --help
yolo-train --help
yolo-report-ap --help
yolo-onnx-export --help
yolo-onnx-optimize --help
yolo-onnx-pipeline --help
yolo-benchmark-report --help
```

<table>
  <tr>
    <td><strong>⚠️ Warning</strong><br>Run <code>source scripts/setup_env.sh ...</code>, not <code>bash scripts/setup_env.sh ...</code>, if you want the environment to stay active in the current shell.</td>
  </tr>
</table>

### 2. Download model weights and raw Fashionpedia data

Download a YOLO checkpoint that you want to fine-tune:

```bash
./scripts/download_yolo_models.sh --generation v26 --task detect --size n
```

Typical resulting path:

```text
models/YOLOv26/yolo26n.pt
```

Then download the demo dataset with the dedicated Fashionpedia script:

```bash
./scripts/download_fashionpedia.sh
```

Expected raw layout:

```text
data/raw/fashionpedia/
├── train/
│   ├── annotations.json
│   └── images/
└── val/
    ├── annotations.json
    └── images/
```

DeepFashion2 uses a separate raw flow:

```bash
export DEEPFASHION2_ARCHIVE_PASSWORD='your-password'   # required for protected nested split archives
./scripts/download_deepfashion2.sh --source official
```

Expected raw layout after the script finishes:

```text
data/raw/deepfashion2/
├── classes.txt
├── train/
│   ├── annos/
│   └── image/
├── validation/
│   ├── annos/
│   └── image/
├── test/
│   └── image/
├── json_for_validation/
└── json_for_test/
```

The DeepFashion2 script downloads the official archive folder, organizes `train` / `validation` / `test`, preserves the auxiliary evaluation metadata directories when present, and writes `classes.txt` itself in the official 13-class order, so it is ready for the `per-image-json` adapter.

During conversion, if `width` / `height` are missing from the per-image JSON annotation, the converter automatically falls back to the actual image size. This is required for DeepFashion2.

### 3. Convert Fashionpedia to a YOLO-styled dataset

This stage only converts the raw schema into a YOLO-styled dataset. It should not change class semantics or act as an optimizer stage.

```bash
yolo-convert-dataset \
  --dataset-name fashionpedia \
  --input-format coco-detection \
  --train-images-dir data/raw/fashionpedia/train/images \
  --train-annotations data/raw/fashionpedia/train/annotations.json \
  --val-images-dir data/raw/fashionpedia/val/images \
  --val-annotations data/raw/fashionpedia/val/annotations.json \
  --output-root data/converted \
  --clean
```

DeepFashion2 conversion uses the `per-image-json` adapter instead:

```bash
yolo-convert-dataset \
  --dataset-name deepfashion2 \
  --input-format per-image-json \
  --train-images-dir data/raw/deepfashion2/train/image \
  --train-annotations data/raw/deepfashion2/train/annos \
  --val-images-dir data/raw/deepfashion2/validation/image \
  --val-annotations data/raw/deepfashion2/validation/annos \
  --class-names-file data/raw/deepfashion2/classes.txt \
  --object-prefix item \
  --category-id-key category_id \
  --bbox-key bounding_box \
  --bbox-format xyxy \
  --clean
```

### 4. Inspect the YOLO-styled dataset

Combined prepare resplit always renames moved image/label pairs to stable unique hash-based names. This avoids filename and label-stem collisions when train/val/test are merged and rebuilt.

Use stats before touching the dataset:

```bash
yolo-print-stats --dataset-dir data/converted/fashionpedia
```

Main outputs:

- `dataset_stats.json`
- `dataset_stats_train.png`
- `dataset_stats_val.png`
- `dataset_stats_test.png` when the dataset contains `test`

This is the report to use when deciding whether to resplit, drop classes, or merge noisy labels.

### 5. Optionally mutate the YOLO-styled dataset in place

This step is optional. Use it only when you want to change the dataset itself. The tracked starter recipe is here: [prepare.example.yaml](configs/prepare/prepare.example.yaml)

Apply it like this:

```bash
yolo-prepare-dataset \
  --dataset-dir data/converted/fashionpedia \
  --recipe configs/prepare/prepare.example.yaml
```

Typical reasons to use it:

- resplit `train` / `val` and optionally create `test`
- enforce minimum instance coverage in `val` / `test`
- shrink the dataset for smoke runs
- drop classes
- rename classes
- merge several classes into a new taxonomy

Recommended follow-up:

```bash
yolo-print-stats --dataset-dir data/converted/fashionpedia
```

### 6. Launch training

The tracked NVIDIA-oriented example is here: [nvidia.example.yaml](configs/train/nvidia.example.yaml)

At minimum, make sure these fields point to the dataset and checkpoint you want:

```yaml
model: models/YOLOv26/yolo26n.pt
data: data/converted/fashionpedia/fashionpedia.yaml
```

Then start training:

```bash
yolo-train --cfg configs/train/nvidia.example.yaml
```

Or override the most important runtime knobs from CLI:

```bash
yolo-train \
  --cfg configs/train/nvidia.example.yaml \
  --model models/YOLOv26/yolo26n.pt \
  --data data/converted/fashionpedia/fashionpedia.yaml \
  --name yolo26n-fashionpedia-run
```

Typical outputs:

```text
runs/<name>/
├── args.yaml
├── results.csv
└── weights/
    ├── best.pt
    └── last.pt
```

### 7. Export per-class AP after training

After training, export validation metrics by class:

```bash
yolo-report-ap \
  --model runs/yolo26n-fashionpedia-run/weights/best.pt \
  --data data/converted/fashionpedia/fashionpedia.yaml \
  --split val
```

Use `val` while you are still changing the recipe, model, or training config.
Use `test` only for the final holdout evaluation after those decisions are fixed.

### 8. Optionally prepare ONNX artifacts for deployment

Export a checkpoint to ONNX:

```bash
yolo-onnx-export \
  --weights runs/yolo26n-fashionpedia-run/weights/best.pt \
  --output deploy/onnx/fashionpedia.export.fp32.onnx \
  --imgsz 1024
```

Optimize an ONNX model for a target runtime:

```bash
yolo-onnx-optimize \
  --input deploy/onnx/fashionpedia.export.fp32.onnx \
  --output-dir deploy/onnx/cpu \
  --target cpu \
  --int8 \
  --calib-dir data/converted/fashionpedia/images/train
```

Or run export + optimize in one command:

```bash
yolo-onnx-pipeline \
  --weights runs/yolo26n-fashionpedia-run/weights/best.pt \
  --artifact-dir deploy/onnx/cuda \
  --target cuda \
  --fp16
```

<table>
  <tr>
    <td><strong>📝 Note</strong><br>ONNX export is a post-training deployment branch. It does not replace <code>yolo-report-ap</code>; you still need a metrics step on <code>val</code> or <code>test</code>.</td>
  </tr>
</table>

## Design Intent

The repository is organized around clean stage boundaries.

### Conversion

`yolo-convert-dataset` absorbs raw schema differences and produces a clean YOLO-styled dataset.

It should not:

- drop classes for experimentation
- merge labels for task design
- act as a tuning stage

### Stats

`yolo-print-stats` is the inspection stage.

Use it to answer questions like:

- which classes dominate?
- how many empty labels exist?
- how many tiny objects exist?
- how strong is the train/val imbalance?
- which classes should be merged or removed?

### Preparation

`yolo-prepare-dataset` is the mutation stage.

It is optional because after conversion you already have a valid YOLO dataset.
Treat it as destructive and deliberate.

### Training and report export

`yolo-train` produces checkpoints.
`yolo-report-ap` is the evaluation/reporting stage for per-class AP.

### ONNX and deployment prep

`yolo-onnx-export` and `yolo-onnx-optimize` are deployment-oriented stages.
Use them after training when you need runtime-specific artifacts for CPU or CUDA inference.

## Command Recap

```bash
source scripts/setup_env.sh base

./scripts/download_yolo_models.sh --generation v26 --task detect --size n
./scripts/download_fashionpedia.sh

yolo-convert-dataset \
  --dataset-name fashionpedia \
  --input-format coco-detection \
  --train-images-dir data/raw/fashionpedia/train/images \
  --train-annotations data/raw/fashionpedia/train/annotations.json \
  --val-images-dir data/raw/fashionpedia/val/images \
  --val-annotations data/raw/fashionpedia/val/annotations.json \
  --clean

yolo-print-stats --dataset-dir data/converted/fashionpedia

yolo-prepare-dataset \
  --dataset-dir data/converted/fashionpedia \
  --recipe configs/prepare/prepare.example.yaml

yolo-print-stats --dataset-dir data/converted/fashionpedia

yolo-train \
  --cfg configs/train/nvidia.example.yaml \
  --model models/YOLOv26/yolo26n.pt \
  --data data/converted/fashionpedia/fashionpedia.yaml \
  --name yolo26n-fashionpedia-run

yolo-report-ap \
  --model runs/yolo26n-fashionpedia-run/weights/best.pt \
  --data data/converted/fashionpedia/fashionpedia.yaml \
  --split val
```

## Practical Advice

- Use stats before every destructive dataset step.
- Treat the converted dataset as the reset point if preparation is destructive.
- Keep the prepare recipe and train config under version control.
- Use `base` for pure training work, `cpu` for CPU ONNX optimization, and `cuda` for CUDA ONNX optimization.
- Keep deployment artifacts outside `runs/` so training outputs and deployment outputs do not mix.

---

**Next:**
[Home](#) · [Datasets](docs/DATASETS.md) · [Training](docs/TRAINING.md) · [Benching](docs/BENCH.md) · [ONNX](docs/ONNX.md) · [CLI](docs/CLI.md) · [Architecture](docs/ARCHITECTURE.md)
