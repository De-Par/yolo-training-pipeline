#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Download raw datasets for YOLO26 pipeline

Usage:
  scripts/download_datasets.sh [--dataset all|fashionpedia|deepfashion2] [--out-dir data/raw]

Environment variables for DeepFashion2 mode:
  DEEPFASHION2_URLS       Space-separated direct archive URLs for DeepFashion2 files
                          Example: "https://.../train.zip https://.../validation.zip"
  DEEPFASHION2_PASSWORD   Optional password for encrypted archives (from official form)

Notes:
  - Fashionpedia downloads are fully automatic
  - DeepFashion2 official distribution may require manual access approval and password
    If you already downloaded archives manually, place them in:
      <out-dir>/deepfashion2/downloads/
    and run this script with --dataset deepfashion2 to unpack/organize
EOF
}

DATASET="all"
OUT_DIR="data/raw"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

case "$DATASET" in
    all|fashionpedia|deepfashion2) ;;
    *)
        echo "Invalid --dataset: $DATASET" >&2
        exit 1
        ;;
esac

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Missing required command: $1" >&2
        exit 1
    }
}

need_cmd curl
need_cmd unzip

mkdir -p "$OUT_DIR"

fashionpedia() {
    local root="$OUT_DIR/fashionpedia"
    local dl="$root/downloads"
    mkdir -p "$dl" "$root/train/images" "$root/val/images"

    echo "[Fashionpedia] Downloading archives and annotations..."
    curl -fL --retry 5 -o "$dl/train2020.zip" \
        "https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip"
    curl -fL --retry 5 -o "$dl/val_test2020.zip" \
        "https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip"
    curl -fL --retry 5 -o "$root/train/annotations.json" \
        "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json"
    curl -fL --retry 5 -o "$root/val/annotations.json" \
        "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json"

    echo "[Fashionpedia] Extracting images..."
    unzip -q -o "$dl/train2020.zip" -d "$root/train/images"
    unzip -q -o "$dl/val_test2020.zip" -d "$root/val/images"

    # Flatten one nested directory (e.g. train/, train2020/, test/, val_test2020/)
    # so final layout stays compatible with pipeline expectations: images/*.jpg
    flatten_images_dir() {
        local dir="$1"
        local nested_dir

        # If files are already present at top level, nothing to do
        if find "$dir" -maxdepth 1 -type f | grep -q .; then
            return 0
        fi

        # Find first nested directory that contains files
        nested_dir="$(find "$dir" -mindepth 1 -maxdepth 2 -type d | while IFS= read -r d; do
            if find "$d" -maxdepth 1 -type f | grep -q .; then
                echo "$d"
                break
            fi
        done)"

        if [[ -n "${nested_dir:-}" ]]; then
            find "$nested_dir" -maxdepth 1 -type f -exec mv -f {} "$dir/" \;
            rmdir "$nested_dir" 2>/dev/null || true
        fi
    }

    flatten_images_dir "$root/train/images"
    flatten_images_dir "$root/val/images"

    echo "[Fashionpedia] Ready at: $root"
}

extract_archive() {
    local file="$1"
    local out="$2"
    local pass="${3:-}"

    case "$file" in
        *.zip)
            if [[ -n "$pass" ]]; then
                unzip -q -P "$pass" -o "$file" -d "$out"
            else
                unzip -q -o "$file" -d "$out"
            fi
            ;;
        *.tar)
            tar -xf "$file" -C "$out"
            ;;
        *.tar.gz|*.tgz)
            tar -xzf "$file" -C "$out"
            ;;
        *)
            echo "Skipping unsupported archive: $file"
            ;;
    esac
}

deepfashion2() {
    local root="$OUT_DIR/deepfashion2"
    local dl="$root/downloads"
    local unpack="$root/unpacked"
    local pass="${DEEPFASHION2_PASSWORD:-}"
    mkdir -p "$dl" "$unpack"

    if [[ -n "${DEEPFASHION2_URLS:-}" ]]; then
        echo "[DeepFashion2] Downloading archives from DEEPFASHION2_URLS..."
        for url in ${DEEPFASHION2_URLS}; do
            local name
            name="$(basename "$url")"
            curl -fL --retry 5 -o "$dl/$name" "$url"
        done
    fi

    if ! find "$dl" -maxdepth 1 -type f \( -name '*.zip' -o -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' \) | grep -q .; then
    cat <<EOF
[DeepFashion2] No archives found in $dl
Please either:
1) Put downloaded DeepFashion2 archives into $dl
2) Or set DEEPFASHION2_URLS with direct links and rerun.
EOF
        return 1
    fi

    echo "[DeepFashion2] Extracting archives..."
    while IFS= read -r archive; do
        extract_archive "$archive" "$unpack" "$pass"
    done < <(find "$dl" -maxdepth 1 -type f \( -name '*.zip' -o -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' \) | sort)

    mkdir -p "$root/train/image" "$root/train/annos" "$root/validation/image" "$root/validation/annos"

    echo "[DeepFashion2] Organizing files..."
    find "$unpack" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | while IFS= read -r f; do
        case "$f" in
        *train*/*image*|*train*/image/*)
            cp -f "$f" "$root/train/image/"
            ;;
        *validation*/*image*|*val*/*image*|*validation*/image/*)
            cp -f "$f" "$root/validation/image/"
            ;;
        esac
    done

    find "$unpack" -type f -name '*.json' | while IFS= read -r f; do
        case "$f" in
        *train*/*anno*|*train*/annos/*)
            cp -f "$f" "$root/train/annos/"
            ;;
        *validation*/*anno*|*val*/*anno*|*validation*/annos/*)
            cp -f "$f" "$root/validation/annos/"
            ;;
        esac
    done

    local train_img_count val_img_count train_ann_count val_ann_count
    train_img_count="$(find "$root/train/image" -type f | wc -l | tr -d ' ')"
    val_img_count="$(find "$root/validation/image" -type f | wc -l | tr -d ' ')"
    train_ann_count="$(find "$root/train/annos" -type f -name '*.json' | wc -l | tr -d ' ')"
    val_ann_count="$(find "$root/validation/annos" -type f -name '*.json' | wc -l | tr -d ' ')"

    echo "[DeepFashion2] Ready at: $root"
    echo "[DeepFashion2] Counts: train_images=$train_img_count val_images=$val_img_count train_annos=$train_ann_count val_annos=$val_ann_count"

    if [[ "$train_img_count" -eq 0 || "$val_img_count" -eq 0 || "$train_ann_count" -eq 0 || "$val_ann_count" -eq 0 ]]; then
        echo "[DeepFashion2] Warning: one or more expected folders are empty. Verify archive layout and password." >&2
    fi
}

if [[ "$DATASET" == "fashionpedia" || "$DATASET" == "all" ]]; then
    fashionpedia
fi

if [[ "$DATASET" == "deepfashion2" || "$DATASET" == "all" ]]; then
    deepfashion2
fi

echo "Done. Raw datasets directory: $OUT_DIR"
