#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${PROJECT_ROOT}"

usage() {
    cat <<'EOF'
Download demo clothes datasets for the repository

Usage:
  scripts/download_clothes_dataset.sh [--dataset all|fashionpedia|deepfashion2] [--out-dir data/raw]

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

    # Flatten all nested directories (e.g. train/, train2020/, test/, val_test2020/)
    # so final layout stays compatible with pipeline expectations: images/*.jpg.
    flatten_images_dir() {
        local dir="$1"
        local moved=0
        local overwritten=0

        # If files are already present at top level, nothing to do
        if ! find "$dir" -mindepth 1 -maxdepth 3 -type d | grep -q .; then
            return 0
        fi

        while IFS= read -r nested_dir; do
            while IFS= read -r file; do
                local dst="$dir/$(basename "$file")"
                if [[ -e "$dst" ]]; then
                    overwritten=$((overwritten + 1))
                fi
                mv -f "$file" "$dst"
                moved=$((moved + 1))
            done < <(find "$nested_dir" -maxdepth 1 -type f)
        done < <(find "$dir" -mindepth 1 -maxdepth 3 -type d | sort -r)

        find "$dir" -mindepth 1 -type d -empty -delete
        if [[ "$moved" -gt 0 ]]; then
            echo "[Fashionpedia] Flattened $moved files in $dir"
        fi
        if [[ "$overwritten" -gt 0 ]]; then
            echo "[Fashionpedia] Warning: $overwritten files were overwritten while flattening $dir" >&2
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
    local img_copied=0 img_collision=0 img_identical=0
    local ann_copied=0 ann_collision=0 ann_identical=0
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
    copy_file_safe() {
        local src="$1"
        local dst_dir="$2"
        local kind="$3"
        local dst="$dst_dir/$(basename "$src")"
        if [[ -e "$dst" ]]; then
            if cmp -s "$src" "$dst"; then
                if [[ "$kind" == "img" ]]; then
                    img_identical=$((img_identical + 1))
                else
                    ann_identical=$((ann_identical + 1))
                fi
            else
                if [[ "$kind" == "img" ]]; then
                    img_collision=$((img_collision + 1))
                else
                    ann_collision=$((ann_collision + 1))
                fi
                echo "[DeepFashion2] Warning: collision for $(basename "$src"), skipping." >&2
            fi
            return
        fi
        cp -f "$src" "$dst"
        if [[ "$kind" == "img" ]]; then
            img_copied=$((img_copied + 1))
        else
            ann_copied=$((ann_copied + 1))
        fi
    }

    while IFS= read -r f; do
        case "$f" in
        *train*/*image*|*train*/image/*)
            copy_file_safe "$f" "$root/train/image" "img"
            ;;
        *validation*/*image*|*val*/*image*|*validation*/image/*)
            copy_file_safe "$f" "$root/validation/image" "img"
            ;;
        esac
    done < <(find "$unpack" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \))

    while IFS= read -r f; do
        case "$f" in
        *train*/*anno*|*train*/annos/*)
            copy_file_safe "$f" "$root/train/annos" "ann"
            ;;
        *validation*/*anno*|*val*/*anno*|*validation*/annos/*)
            copy_file_safe "$f" "$root/validation/annos" "ann"
            ;;
        esac
    done < <(find "$unpack" -type f -name '*.json')

    local train_img_count val_img_count train_ann_count val_ann_count
    train_img_count="$(find "$root/train/image" -type f | wc -l | tr -d ' ')"
    val_img_count="$(find "$root/validation/image" -type f | wc -l | tr -d ' ')"
    train_ann_count="$(find "$root/train/annos" -type f -name '*.json' | wc -l | tr -d ' ')"
    val_ann_count="$(find "$root/validation/annos" -type f -name '*.json' | wc -l | tr -d ' ')"

    echo "[DeepFashion2] Ready at: $root"
    echo "[DeepFashion2] Counts: train_images=$train_img_count val_images=$val_img_count train_annos=$train_ann_count val_annos=$val_ann_count"
    echo "[DeepFashion2] Copied: images=$img_copied annos=$ann_copied"
    if [[ "$img_collision" -gt 0 || "$ann_collision" -gt 0 ]]; then
        echo "[DeepFashion2] Collisions skipped: images=$img_collision annos=$ann_collision" >&2
    fi
    if [[ "$img_identical" -gt 0 || "$ann_identical" -gt 0 ]]; then
        echo "[DeepFashion2] Identical duplicates skipped: images=$img_identical annos=$ann_identical"
    fi

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
