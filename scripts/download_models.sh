#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${PROJECT_ROOT}"

usage() {
    cat <<'EOF'
Download all YOLO model weights (.pt) for a specific generation (v3, v4, ..., v26)

Examples:
  ./scripts/download_models.sh --generation v26
  ./scripts/download_models.sh --generation 5 --release-tag v8.4.0
  ./scripts/download_models.sh --generation v26 --size n
  ./scripts/download_models.sh --generation v26 --size n,s --task detect
  ./scripts/download_models.sh --generation v26 --task detect --dry-run

Options:
  --generation <vN|N>   Required. Generation number with or without 'v' prefix
  --out-dir <path>      Base output directory (default: models)
                        Files are saved into <out-dir>/YOLOvN/
  --release-tag <tag>   Optional fixed release tag (e.g. v8.4.0)
                        If omitted, scans recent releases and takes newest match per filename
  --task <name>         Filter task: all|detect|seg|pose|obb|cls|objv1 (default: all)
  --size <value>        Filter model size: all|n|s|m|l|x or CSV list (e.g. n,s)
                        Default: all
  --dry-run             Print matched files/URLs without downloading
  -h, --help            Show help

Requirements:
  - curl
  - python3
EOF
}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Missing required command: $1" >&2
        exit 1
    }
}

need_cmd curl
need_cmd python3

GENERATION=""
OUT_DIR="models"
RELEASE_TAG=""
TASK="all"
SIZE_FILTER="all"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --generation)
            GENERATION="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --release-tag)
            RELEASE_TAG="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --size)
            SIZE_FILTER="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="1"
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

if [[ -z "$GENERATION" ]]; then
    echo "--generation is required" >&2
    usage
    exit 1
fi

GEN_NUM="${GENERATION#v}"
if ! [[ "$GEN_NUM" =~ ^[0-9]+$ ]]; then
    echo "Invalid generation: $GENERATION (expected vN or N)" >&2
    exit 1
fi

case "$TASK" in
    all|detect|seg|pose|obb|cls|objv1) ;;
    *)
        echo "Invalid --task: $TASK (allowed: all|detect|seg|pose|obb|cls|objv1)" >&2
        exit 1
        ;;
esac

if [[ "$SIZE_FILTER" != "all" ]]; then
    if ! [[ "$SIZE_FILTER" =~ ^[nsmxl](,[nsmxl])*$ ]]; then
        echo "Invalid --size: $SIZE_FILTER (allowed: all|n|s|m|l|x or CSV, e.g. n,s)" >&2
        exit 1
    fi
fi

GEN_DIR="$OUT_DIR/YOLOv$GEN_NUM"
mkdir -p "$GEN_DIR"

if [[ -n "$RELEASE_TAG" ]]; then
    API_URL="https://api.github.com/repos/ultralytics/assets/releases/tags/$RELEASE_TAG"
else
    API_URL="https://api.github.com/repos/ultralytics/assets/releases?per_page=100"
fi

echo "[INFO] Fetching release metadata from: $API_URL"
JSON_PAYLOAD="$(curl -fsSL "$API_URL")"

if [[ -z "$JSON_PAYLOAD" ]]; then
    echo "GitHub API returned an empty response. Try again or set --release-tag explicitly." >&2
    exit 1
fi

TMP_JSON="$(mktemp)"
trap 'rm -f "$TMP_JSON"' EXIT
printf '%s' "$JSON_PAYLOAD" > "$TMP_JSON"

MATCHED="$(python3 - "$GEN_NUM" "$TASK" "$SIZE_FILTER" "$RELEASE_TAG" "$TMP_JSON" <<'PY'
import json
import re
import sys

gen = sys.argv[1]
task_filter = sys.argv[2]
size_filter = sys.argv[3]
fixed_tag = sys.argv[4]
json_path = sys.argv[5]

allowed_sizes = set("nsmxl")
wanted_sizes = set(size_filter.split(",")) if size_filter != "all" else allowed_sizes

with open(json_path, "r", encoding="utf-8") as f:
    payload = json.load(f)
if isinstance(payload, dict):
    releases = [payload]
elif isinstance(payload, list):
    releases = payload
else:
    releases = []

pattern = re.compile(rf"^yolo{re.escape(gen)}.*\.pt$", re.IGNORECASE)


def size_of_name(name: str, generation: str) -> str:
    lowered = name.lower()
    prefix = f"yolo{generation}"
    if not lowered.startswith(prefix):
        return ""
    rest = lowered[len(prefix):]
    if not rest:
        return ""
    size = rest[0]
    if size in allowed_sizes:
        return size
    return ""


def task_of_name(name: str) -> str:
    # detect: yolo26n.pt, yolo26x.pt
    if "-" not in name:
        return "detect"
    suffix = name.split("-", 1)[1].lower()
    if suffix.startswith("seg"):
        return "seg"
    if suffix.startswith("pose"):
        return "pose"
    if suffix.startswith("obb"):
        return "obb"
    if suffix.startswith("cls"):
        return "cls"
    if suffix.startswith("objv1"):
        return "objv1"
    return "other"

# Keep the first occurrence of each filename (newest release order from API)
seen = set()
rows = []
for rel in releases:
    assets = rel.get("assets") or []
    for a in assets:
        name = a.get("name") or ""
        url = a.get("browser_download_url") or ""
        if not (name and url):
            continue
        if not pattern.match(name):
            continue
        model_size = size_of_name(name, gen)
        if size_filter != "all" and model_size not in wanted_sizes:
            continue
        model_task = task_of_name(name)
        if task_filter != "all" and model_task != task_filter:
            continue
        if name in seen:
            continue
        seen.add(name)
        rows.append((name, url, rel.get("tag_name", "")))

# Sort for stable output by filename
rows.sort(key=lambda x: x[0])
for name, url, tag in rows:
    # name<TAB>url<TAB>tag
    print(f"{name}\t{url}\t{tag}")
PY
)"

if [[ -z "$MATCHED" ]]; then
    if [[ -n "$RELEASE_TAG" ]]; then
        echo "No models found for generation v$GEN_NUM in release $RELEASE_TAG" >&2
    else
        echo "No models found for generation v$GEN_NUM in scanned releases" >&2
    fi
    exit 1
fi

TOTAL="$(printf '%s\n' "$MATCHED" | wc -l | tr -d ' ')"
echo "[INFO] Matched models: $TOTAL"
echo "$MATCHED" | awk -F '\t' '{printf("  - %s (%s)\n", $1, $3)}'

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[INFO] Dry-run mode: no files downloaded."
    exit 0
fi

count=0
while IFS=$'\t' read -r name url _tag; do
    [[ -z "$name" || -z "$url" ]] && continue
    out="$GEN_DIR/$name"
    count=$((count + 1))
    echo "[INFO] $count/$TOTAL Downloading $name"
    curl -fL -C - --retry 5 --connect-timeout 20 -o "$out" "$url"
    echo
done <<< "$MATCHED"

echo "[INFO] Downloaded $count files into: $GEN_DIR"
