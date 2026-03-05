#!/usr/bin/env python3
from __future__ import annotations

import argparse
import yaml
import torch

from pathlib import Path
from typing import Any, Dict, List, Set


DEFAULTS: Dict[str, Any] = {
    "epochs": 100,
    "imgsz": 640,
    "batch": 32,
    "device": None,  # auto
    "workers": 12,
    "project": "runs/train",
    "name": "yolo-run",
    "seed": 42,
    "amp": True,
    "cache": None,  # None | "ram" | "disk"
    "compile": False,
    "val": True,
    "exist_ok": False,
    "verbose": False,
}

# Keys we intentionally allow in cfg, even if they are not passed to model.train()
CFG_META_KEYS: Set[str] = {"model", "data"}

# Ultralytics train() accepts a fairly large set of keys; we validate dynamically against defaults
# This is stable within a given ultralytics version
def _ultra_supported_train_keys() -> Set[str]:
    try:
        # Ultralytics uses a unified default cfg
        from ultralytics.cfg import get_cfg  # type: ignore
        cfg = get_cfg()
        if isinstance(cfg, dict):
            return set(cfg.keys())
        # cfg may be a SimpleNamespace-like object
        return set(vars(cfg).keys())
    except Exception:
        # Fallback: we can only validate keys we know about locally
        return set(DEFAULTS.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO (Ultralytics) on a prepared data.yaml")
    parser.add_argument("--cfg", type=Path, help="YAML config with train parameters (optional)")

    # Not required on CLI: can be provided via --cfg
    parser.add_argument("--data", type=str, help="Path to prepared data.yaml (or via cfg: data)")
    parser.add_argument("--model", type=str, help="Path to YOLO checkpoint/config (or via cfg: model)")

    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device", type=str, help='Examples: "cpu", "0", "cuda:0"')
    parser.add_argument("--workers", type=int)
    parser.add_argument("--classes", type=str, help='Comma-separated class IDs, e.g. "0,2,5"')

    parser.add_argument("--project", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--seed", type=int)

    parser.add_argument("--amp", action="store_true", help="Enable AMP mixed precision")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")
    parser.add_argument("--cache", type=str, choices=["ram", "disk"])
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile if supported by Ultralytics")
    parser.add_argument("--no-val", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def load_cfg(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return cfg


def pick(name: str, cli_value: Any, cfg: Dict[str, Any], *, required: bool = False) -> Any:
    if cli_value is not None:
        return cli_value
    if name in cfg:
        return cfg[name]
    if required:
        raise ValueError(f"Missing required parameter '{name}'. Provide via --{name} or --cfg.")
    return DEFAULTS[name]


def parse_classes_cli(raw: str | None) -> List[int] | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.lower() == "none":
        return None
    out: List[int] = []
    for token in s.split(","):
        t = token.strip()
        if t == "":
            continue
        if t.startswith("+"):
            t = t[1:]
        if not t.isdigit():
            raise ValueError(f"Invalid --classes token '{token}'. Use comma-separated non-negative integers.")
        out.append(int(t))
    if not out:
        return None
    return sorted(set(out))


def auto_device(device_value: Any) -> Any:
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()
    if device_value is not None:
        s = str(device_value).strip()
        if s == "":
            if cuda_available:
                return 0
            if mps_available:
                return "mps"
            return "cpu"
        if s.lower() == "mps":
            if mps_available:
                return "mps"
            print("[WARN] Requested device='mps' but MPS is not available. Falling back to CPU.", flush=True)
            return "cpu"
        if s.isdigit():
            if not cuda_available:
                print(f"[WARN] Requested CUDA device='{s}' but CUDA is not available. Falling back to CPU.", flush=True)
                return "cpu"
            return int(s)
        if s.lower().startswith("cuda") and not cuda_available:
            print(f"[WARN] Requested device='{s}' but CUDA is not available. Falling back to CPU.", flush=True)
            return "cpu"
        return s
    if cuda_available:
        return 0
    if mps_available:
        return "mps"
    return "cpu"


def setup_torch_backends(device: Any) -> None:
    if torch.cuda.is_available() and device != "cpu":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def warn_unsupported_cfg_keys(cfg: Dict[str, Any], supported_train_keys: Set[str]) -> None:
    # Warn for cfg keys that are not understood by ultralytics train(),
    # excluding meta keys and excluding empty/null keys
    unknown = []
    for k, v in cfg.items():
        if k in CFG_META_KEYS:
            continue
        if v is None:
            continue
        if k not in supported_train_keys:
            unknown.append(k)
    if unknown:
        unknown_sorted = ", ".join(sorted(unknown))
        print(f"[WARN] Config contains keys not supported by ultralytics train(): {unknown_sorted}", flush=True)
        print("[WARN] They will be ignored unless you handle them manually.", flush=True)


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.cfg)

    supported_train_keys = _ultra_supported_train_keys()
    warn_unsupported_cfg_keys(cfg, supported_train_keys)

    # Allow required inputs from cfg if CLI missing
    model_value = args.model if args.model is not None else cfg.get("model")
    data_value = args.data if args.data is not None else cfg.get("data")
    if not model_value:
        raise ValueError("Missing required parameter 'model'. Provide via --model or cfg: model")
    if not data_value:
        raise ValueError("Missing required parameter 'data'. Provide via --data or cfg: data")

    device_value = auto_device(pick("device", args.device, cfg))
    setup_torch_backends(device_value)

    amp_value = pick("amp", None, cfg)
    if args.amp:
        amp_value = True
    if args.no_amp:
        amp_value = False
    if amp_value is None:
        amp_value = DEFAULTS["amp"]

    val_value = pick("val", None, cfg)
    if args.no_val:
        val_value = False
    if val_value is None:
        val_value = DEFAULTS["val"]

    classes_value = parse_classes_cli(args.classes)
    if classes_value is None and "classes" in cfg:
        classes_value = cfg.get("classes")

    project_value = pick("project", args.project, cfg)
    project_value = str(Path(project_value).expanduser().resolve())

    # Build kwargs from known local defaults + cfg + CLI
    train_kwargs: Dict[str, Any] = {
        "data": str(data_value),
        "epochs": pick("epochs", args.epochs, cfg),
        "imgsz": pick("imgsz", args.imgsz, cfg),
        "batch": pick("batch", args.batch, cfg),
        "device": device_value,
        "workers": pick("workers", args.workers, cfg),
        "project": project_value,
        "name": pick("name", args.name, cfg),
        "seed": pick("seed", args.seed, cfg),
        "amp": amp_value,
        "cache": pick("cache", args.cache, cfg),
        "compile": pick("compile", True if args.compile else None, cfg),
        "val": val_value,
        "classes": classes_value,
        "exist_ok": pick("exist_ok", True if args.exist_ok else None, cfg),
        "verbose": pick("verbose", True if args.verbose else None, cfg),
    }

    # Extra: also accept any additional supported ultralytics train() keys from YAML automatically
    for k, v in cfg.items():
        if k in CFG_META_KEYS:
            continue
        if v is None:
            continue
        if k in train_kwargs:
            continue   # already set by our merge logic
        if k in supported_train_keys:
            train_kwargs[k] = v

    # Drop keys that are definitely unsupported (for safety)
    train_kwargs = {k: v for k, v in train_kwargs.items() if (k in supported_train_keys or k in {"data"})}

    print("[INFO] Model:", model_value, flush=True)
    print("[INFO] Data:", data_value, flush=True)
    print("[INFO] Device:", device_value, flush=True)
    print("[INFO] torch.cuda.is_available:", torch.cuda.is_available(), flush=True)
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print("[INFO] torch.backends.mps.is_available:", mps_available, flush=True)
    if torch.cuda.is_available():
        try:
            print("[INFO] CUDA device 0:", torch.cuda.get_device_name(0), flush=True)
        except Exception:
            pass

    from ultralytics import YOLO

    model = YOLO(str(model_value))

    try:
        model.train(**train_kwargs)
    except TypeError as e:
        msg = str(e)
        if "unexpected keyword argument" in msg:
            print("[WARN] Ultralytics rejected some args (version mismatch).", flush=True)
            print("[WARN] Error:", msg, flush=True)
            print("[WARN] Tip: remove the mentioned key from YAML or upgrade/downgrade ultralytics.", flush=True)
        raise


if __name__ == "__main__":
    main()
