#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${PROJECT_ROOT}"

SOURCE="kaggle"
OUT_DIR="data/raw"
DATASET_SLUG="${DEEPFASHION2_KAGGLE_DATASET:-rishi1903/deepfashion-2}"
ARCHIVE_PASSWORD="${DEEPFASHION2_ARCHIVE_PASSWORD:-}"

usage() {
    cat <<USAGE
Download or organize the DeepFashion2 demo dataset

Usage:
  scripts/download_deepfashion2.sh [--source kaggle|local] [--out-dir data/raw] [--dataset-slug <owner/dataset>] [--archive-password <password>]

Options:
  --source kaggle|local       kaggle downloads an unofficial Kaggle mirror via kaggle CLI
                              local only unpacks archives already placed in <out-dir>/deepfashion2/downloads/
  --out-dir PATH              Base raw-data directory. Default: data/raw
  --dataset-slug SLUG         Kaggle dataset slug. Default: ${DATASET_SLUG}
  --archive-password PASS     Password for encrypted nested zip archives
                              Prefer DEEPFASHION2_ARCHIVE_PASSWORD env var to avoid leaking the password into shell history

Requirements for --source kaggle:
  - kaggle CLI installed and authenticated
  - a reachable dataset slug

Notes:
  - This script intentionally avoids the official password-protected archive flow unless you explicitly provide the password
  - The default Kaggle source is a mirror, not the official distribution
  - Verify provenance, contents, and licensing before production use
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)
            SOURCE="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --dataset-slug)
            DATASET_SLUG="$2"
            shift 2
            ;;
        --archive-password)
            ARCHIVE_PASSWORD="$2"
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

case "$SOURCE" in
    kaggle|local) ;;
    *)
        echo "Invalid --source: $SOURCE" >&2
        exit 1
        ;;
esac

need_cmd() {
    local cmd="$1"
    command -v "$cmd" >/dev/null 2>&1 || {
        echo "Missing required command: $cmd" >&2
        if [[ "$cmd" == "kaggle" ]]; then
            echo "[DeepFashion2] Install project dependencies again with: source scripts/setup_env.sh base" >&2
            echo "[DeepFashion2] Then configure Kaggle API access, for example via ~/.kaggle/kaggle.json" >&2
        fi
        exit 1
    }
}

need_cmd python3
need_cmd tar
need_cmd mktemp

root="$OUT_DIR/deepfashion2"
dl="$root/downloads"
unpack="$root/unpacked"
mkdir -p "$dl" "$unpack"

print_extract_progress() {
    local label="$1"
    local index="$2"
    local total="$3"
    local archive="$4"
    echo "[DeepFashion2] ${label} [${index}/${total}]: $(basename "$archive")"
}

reset_unpack_dir() {
    EXTRACT_DIR="$unpack" python3 - <<'PY'
import os
import shutil
from pathlib import Path

root = Path(os.environ["EXTRACT_DIR"])
root.mkdir(parents=True, exist_ok=True)
for child in root.iterdir():
    if child.is_dir():
        shutil.rmtree(child)
    else:
        child.unlink()
PY
}

extract_zip_archive() {
    local file="$1"
    local out="$2"

    ZIP_FILE="$file" ZIP_OUT="$out" ZIP_PASSWORD="$ARCHIVE_PASSWORD" python3 - <<'PY'
import os
import sys
import zipfile
from pathlib import Path

archive = Path(os.environ["ZIP_FILE"])
out_dir = Path(os.environ["ZIP_OUT"])
password = os.environ.get("ZIP_PASSWORD", "")
pwd = password.encode("utf-8") if password else None

try:
    with zipfile.ZipFile(archive) as zf:
        encrypted = any(info.flag_bits & 0x1 for info in zf.infolist())
        if encrypted and not pwd:
            print(f"[DeepFashion2] Error: archive requires a password: {archive}", file=sys.stderr)
            print("[DeepFashion2] Provide --archive-password or set DEEPFASHION2_ARCHIVE_PASSWORD.", file=sys.stderr)
            raise SystemExit(1)
        try:
            zf.extractall(out_dir, pwd=pwd)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "password required" in message or "encrypted" in message:
                print(f"[DeepFashion2] Error: archive requires a password: {archive}", file=sys.stderr)
                print("[DeepFashion2] Provide --archive-password or set DEEPFASHION2_ARCHIVE_PASSWORD.", file=sys.stderr)
                raise SystemExit(1) from exc
            if "bad password" in message or "password" in message:
                print(f"[DeepFashion2] Error: invalid archive password: {archive}", file=sys.stderr)
                print("[DeepFashion2] Verify --archive-password or DEEPFASHION2_ARCHIVE_PASSWORD.", file=sys.stderr)
                raise SystemExit(1) from exc
            raise
except NotImplementedError as exc:
    print(f"[DeepFashion2] Error: unsupported zip encryption for archive: {archive}", file=sys.stderr)
    print("[DeepFashion2] Use a mirror with already unpacked train/validation content or provide a compatible decrypted archive.", file=sys.stderr)
    raise SystemExit(1) from exc
PY
}

extract_archive() {
    local file="$1"
    local out="$2"

    case "$file" in
        *.zip)
            extract_zip_archive "$file" "$out"
            ;;
        *.tar)
            tar -xf "$file" -C "$out"
            ;;
        *.tar.gz|*.tgz)
            tar -xzf "$file" -C "$out"
            ;;
        *)
            echo "[DeepFashion2] Skipping unsupported archive: $file"
            ;;
    esac
}

extract_archives_from_list() {
    local label="$1"
    local list_file="$2"
    local out_dir="$3"
    local total
    local index=0
    local archive

    total="$(wc -l < "$list_file" | tr -d ' ')"
    while IFS= read -r archive; do
        [[ -n "$archive" ]] || continue
        index=$((index + 1))
        print_extract_progress "$label" "$index" "$total" "$archive"
        extract_archive "$archive" "$out_dir"
    done < "$list_file"
}

extract_nested_archives() {
    local pass="$1"
    local processed_file="$2"
    local list_file
    local pending_file
    local archive

    list_file="$(mktemp)"
    pending_file="$(mktemp)"
    find "$unpack" -type f \( -name '*.zip' -o -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' \) | sort > "$list_file"

    while IFS= read -r archive; do
        [[ -n "$archive" ]] || continue
        if grep -Fqx "$archive" "$processed_file"; then
            continue
        fi
        printf '%s\n' "$archive" >> "$pending_file"
    done < "$list_file"

    if [[ ! -s "$pending_file" ]]; then
        rm -f "$list_file" "$pending_file"
        return 1
    fi

    echo "[DeepFashion2] Extracting nested archives (pass ${pass})..."
    extract_archives_from_list "Extracting nested archive" "$pending_file" "$unpack"
    cat "$pending_file" >> "$processed_file"

    rm -f "$list_file" "$pending_file"
    return 0
}

if [[ "$SOURCE" == "kaggle" ]]; then
    need_cmd kaggle
    echo "[DeepFashion2] Downloading Kaggle mirror: $DATASET_SLUG"
    kaggle datasets download -d "$DATASET_SLUG" -p "$dl" --force
fi

if ! find "$dl" -maxdepth 1 -type f \( -name '*.zip' -o -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' \) | grep -q .; then
    cat <<USAGE
[DeepFashion2] No archives found in $dl
Provide archives manually or rerun with:
  scripts/download_deepfashion2.sh --source kaggle --dataset-slug <owner/dataset>
USAGE
    exit 1
fi

reset_unpack_dir

list_file="$(mktemp)"
processed_file="$(mktemp)"
find "$dl" -maxdepth 1 -type f \( -name '*.zip' -o -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' \) | sort > "$list_file"

if [[ ! -s "$list_file" ]]; then
    rm -f "$list_file" "$processed_file"
    echo "[DeepFashion2] No supported archives found in $dl" >&2
    exit 1
fi

echo "[DeepFashion2] Extracting archives..."
extract_archives_from_list "Extracting archive" "$list_file" "$unpack"
cat "$list_file" >> "$processed_file"
rm -f "$list_file"

nested_pass=1
while extract_nested_archives "$nested_pass" "$processed_file"; do
    nested_pass=$((nested_pass + 1))
done
rm -f "$processed_file"

mkdir -p "$root/train/image" "$root/train/annos" "$root/validation/image" "$root/validation/annos"
find "$root/train/image" -mindepth 1 -delete
find "$root/train/annos" -mindepth 1 -delete
find "$root/validation/image" -mindepth 1 -delete
find "$root/validation/annos" -mindepth 1 -delete

img_copied=0
img_collision=0
img_identical=0
ann_copied=0
ann_collision=0
ann_identical=0

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

train_img_count="$(find "$root/train/image" -type f | wc -l | tr -d ' ')"
val_img_count="$(find "$root/validation/image" -type f | wc -l | tr -d ' ')"
train_ann_count="$(find "$root/train/annos" -type f -name '*.json' | wc -l | tr -d ' ')"
val_ann_count="$(find "$root/validation/annos" -type f -name '*.json' | wc -l | tr -d ' ')"

echo "[DeepFashion2] Ready at: $root"
echo "[DeepFashion2] Source: $SOURCE"
if [[ "$SOURCE" == "kaggle" ]]; then
    echo "[DeepFashion2] Kaggle dataset: $DATASET_SLUG"
fi
echo "[DeepFashion2] Counts: train_images=$train_img_count val_images=$val_img_count train_annos=$train_ann_count val_annos=$val_ann_count"
echo "[DeepFashion2] Copied: images=$img_copied annos=$ann_copied"
if [[ "$img_collision" -gt 0 || "$ann_collision" -gt 0 ]]; then
    echo "[DeepFashion2] Collisions skipped: images=$img_collision annos=$ann_collision" >&2
fi
if [[ "$img_identical" -gt 0 || "$ann_identical" -gt 0 ]]; then
    echo "[DeepFashion2] Identical duplicates skipped: images=$img_identical annos=$ann_identical"
fi
if [[ "$train_img_count" -eq 0 || "$val_img_count" -eq 0 || "$train_ann_count" -eq 0 || "$val_ann_count" -eq 0 ]]; then
    echo "[DeepFashion2] Error: one or more expected folders are empty after extraction and organization." >&2
    echo "[DeepFashion2] Verify the mirror layout and archive contents under: $root" >&2
    exit 1
fi

echo "[DeepFashion2] Warning: this script uses a Kaggle mirror by default, not the official password-protected distribution." >&2
