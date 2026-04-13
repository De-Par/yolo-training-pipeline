#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${PROJECT_ROOT}"

SOURCE="official"
OUT_DIR="data/raw"
ARCHIVE_PASSWORD="${DEEPFASHION2_ARCHIVE_PASSWORD:'2019Deepfashion2**'}"
OFFICIAL_FOLDER_URL="https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok?usp=sharing"
DOWNLOAD_ONLY=0

usage() {
    cat <<USAGE
Download or organize the official DeepFashion2 dataset.

Usage:
  scripts/download_deepfashion2.sh [--source official|local] [--out-dir data/raw] [--archive-password <password>] [--download-only]

Options:
  --source official|local    official downloads the official Google Drive folder via gdown
                            local only unpacks archives already placed in <out-dir>/deepfashion2/downloads/
  --out-dir PATH            Base raw-data directory. Default: data/raw
  --archive-password PASS   Password for encrypted nested zip archives
                            Prefer DEEPFASHION2_ARCHIVE_PASSWORD env var to avoid leaking the password into shell history
  --download-only           Download official archives into <out-dir>/deepfashion2/downloads/ and stop
  -h, --help                Show this help message

Requirements for --source official:
  - gdown installed in the current environment
  - access to the official DeepFashion2 Google Drive folder
  - the archive password for protected nested zip files

Official source:
  - GitHub: https://github.com/switchablenorms/DeepFashion2
  - Google Drive folder: ${OFFICIAL_FOLDER_URL}

Notes:
  - Obtain the password by filling the official request form linked from the DeepFashion2 project page
  - The script writes data/raw/deepfashion2/classes.txt in the official 13-class order
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
        --archive-password)
            ARCHIVE_PASSWORD="$2"
            shift 2
            ;;
        --download-only)
            DOWNLOAD_ONLY=1
            shift
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
    official|local) ;;
    *)
        echo "Invalid --source: $SOURCE" >&2
        exit 1
        ;;
esac

need_cmd() {
    local cmd="$1"
    command -v "$cmd" >/dev/null 2>&1 || {
        echo "Missing required command: $cmd" >&2
        if [[ "$cmd" == "gdown" ]]; then
            echo "[DeepFashion2] Install project dependencies again with: source scripts/setup_env.sh base" >&2
        fi
        exit 1
    }
}

need_cmd unzip
need_cmd zipinfo
need_cmd tar
need_cmd mktemp
need_cmd find

if [[ "$SOURCE" == "official" ]]; then
    need_cmd gdown
fi

root="$OUT_DIR/deepfashion2"
dl="$root/downloads"
unpack="$root/unpacked"
classes_file="$root/classes.txt"
mkdir -p "$dl" "$unpack"

print_extract_progress() {
    local label="$1"
    local index="$2"
    local total="$3"
    local archive="$4"
    echo "[DeepFashion2] ${label} [${index}/${total}]: $(basename "$archive")"
}

reset_unpack_dir() {
    rm -rf "$unpack"
    mkdir -p "$unpack"
}

zip_is_encrypted() {
    local archive="$1"
    unzip -Z -v "$archive" 2>/dev/null | grep -q "file security status:.*encrypted"
}

extract_zip_archive() {
    local file="$1"
    local out="$2"
    local log_file

    log_file="$(mktemp)"

    if [[ -n "$ARCHIVE_PASSWORD" ]]; then
        if ! unzip -P "$ARCHIVE_PASSWORD" -q -o "$file" -d "$out" >"$log_file" 2>&1; then
            cat "$log_file" >&2
            rm -f "$log_file"
            echo "[DeepFashion2] Error: failed to extract password-protected archive: $file" >&2
            echo "[DeepFashion2] Verify --archive-password or DEEPFASHION2_ARCHIVE_PASSWORD." >&2
            exit 1
        fi
        rm -f "$log_file"
        return
    fi

    if zip_is_encrypted "$file"; then
        rm -f "$log_file"
        echo "[DeepFashion2] Error: archive requires a password: $file" >&2
        echo "[DeepFashion2] Provide --archive-password or set DEEPFASHION2_ARCHIVE_PASSWORD." >&2
        exit 1
    fi

    if ! unzip -q -o "$file" -d "$out" >"$log_file" 2>&1; then
        cat "$log_file" >&2
        rm -f "$log_file"
        echo "[DeepFashion2] Error: failed to extract archive: $file" >&2
        exit 1
    fi

    rm -f "$log_file"
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

write_classes_file() {
    cat > "$classes_file" <<'EOF_CLASSES'
short sleeve top
long sleeve top
short sleeve outwear
long sleeve outwear
vest
sling
shorts
trousers
skirt
short sleeve dress
long sleeve dress
vest dress
sling dress
EOF_CLASSES
}

download_official_archives() {
    echo "[DeepFashion2] Downloading official archive folder..."
    gdown --folder "$OFFICIAL_FOLDER_URL" -O "$dl"
}

if [[ "$SOURCE" == "official" ]]; then
    download_official_archives
fi

if ! find "$dl" -maxdepth 1 -type f \( -name '*.zip' -o -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' \) | grep -q .; then
    cat <<USAGE
[DeepFashion2] No archives found in $dl
Provide archives manually or rerun with:
  scripts/download_deepfashion2.sh --source official
USAGE
    exit 1
fi

if [[ "$DOWNLOAD_ONLY" -eq 1 ]]; then
    echo "[DeepFashion2] Downloaded official archives to: $dl"
    exit 0
fi

reset_unpack_dir

top_level_list="$(mktemp)"
processed_nested="$(mktemp)"
trap 'rm -f "$top_level_list" "$processed_nested"' EXIT
find "$dl" -maxdepth 1 -type f \( -name '*.zip' -o -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' \) | sort > "$top_level_list"

extract_archives_from_list "Extracting archive" "$top_level_list" "$unpack"

pass=1
while extract_nested_archives "$pass" "$processed_nested"; do
    pass=$((pass + 1))
done

find_first_dir() {
    local dir_name="$1"
    find "$unpack" -type d -name "$dir_name" | sort | head -n 1
}

train_src="$(find_first_dir train)"
val_src="$(find_first_dir validation)"
test_src="$(find_first_dir test)"
json_val_src="$(find_first_dir json_for_validation)"
json_test_src="$(find_first_dir json_for_test)"

[[ -n "$train_src" ]] || { echo "[DeepFashion2] Error: train directory not found after extraction." >&2; exit 1; }
[[ -n "$val_src" ]] || { echo "[DeepFashion2] Error: validation directory not found after extraction." >&2; exit 1; }

echo "[DeepFashion2] Organizing files..."
rm -rf "$root/train" "$root/validation" "$root/test" "$root/json_for_validation" "$root/json_for_test"
mv "$train_src" "$root/train"
mv "$val_src" "$root/validation"
[[ -n "$test_src" ]] && mv "$test_src" "$root/test"
[[ -n "$json_val_src" ]] && mv "$json_val_src" "$root/json_for_validation"
[[ -n "$json_test_src" ]] && mv "$json_test_src" "$root/json_for_test"
rm -rf "$unpack"

write_classes_file

count_files() {
    local path="$1"
    local pattern="$2"
    if [[ -d "$path" ]]; then
        find "$path" -type f -name "$pattern" | wc -l | tr -d ' '
    else
        echo 0
    fi
}

train_images="$(count_files "$root/train/image" '*.jpg')"
val_images="$(count_files "$root/validation/image" '*.jpg')"
test_images="$(count_files "$root/test/image" '*.jpg')"
train_annos="$(count_files "$root/train/annos" '*.json')"
val_annos="$(count_files "$root/validation/annos" '*.json')"

echo "[DeepFashion2] Ready at: $root"
echo "[DeepFashion2] Source: $SOURCE"
echo "[DeepFashion2] Counts: train_images=${train_images} val_images=${val_images} test_images=${test_images} train_annos=${train_annos} val_annos=${val_annos}"
echo "[DeepFashion2] Wrote classes.txt: $classes_file"
echo "[DeepFashion2] Official reference: train_images=390884 val_images=33669 test_images=67342"

if [[ "$train_images" -eq 0 || "$val_images" -eq 0 ]]; then
    echo "[DeepFashion2] Error: one or more expected folders are empty. Verify the official archives and password." >&2
    exit 1
fi

if [[ "$train_images" -ne 390884 || "$val_images" -ne 33669 || "$test_images" -ne 67342 ]]; then
    echo "[DeepFashion2] Warning: counts differ from the official DeepFashion2 release. The downloaded source may be partial or repackaged." >&2
fi
