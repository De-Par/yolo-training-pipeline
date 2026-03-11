#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${PROJECT_ROOT}"

usage() {
    cat <<'USAGE'
Download the Fashionpedia demo dataset

Usage:
  scripts/download_fashionpedia.sh [--out-dir data/raw]

Notes:
  - Downloads the public Fashionpedia train/val image archives and annotations
  - Produces the raw layout expected by the demo commands in README.md
USAGE
}

OUT_DIR="data/raw"

while [[ $# -gt 0 ]]; do
    case "$1" in
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

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Missing required command: $1" >&2
        exit 1
    }
}

need_cmd curl
need_cmd unzip

root="$OUT_DIR/fashionpedia"
dl="$root/downloads"
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

flatten_images_dir() {
    local dir="$1"
    local moved=0
    local overwritten=0

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
