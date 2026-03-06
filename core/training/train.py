from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

from core.common import PipelineError


DEFAULTS: Dict[str, Any] = {
    "epochs": 100,
    "imgsz": 640,
    "batch": 32,
    "device": None,
    "workers": 12,
    "project": "runs/train",
    "name": "yolo-run",
    "seed": 42,
    "amp": True,
    "cache": None,
    "compile": False,
    "val": True,
    "exist_ok": False,
    "verbose": False,
}

CFG_META_KEYS: Set[str] = {"model", "data"}


def _ultra_supported_train_keys() -> Set[str]:
    try:
        from ultralytics.cfg import get_cfg  # type: ignore

        cfg = get_cfg()
        if isinstance(cfg, dict):
            return set(cfg.keys())
        return set(vars(cfg).keys())
    except Exception:
        return set(DEFAULTS.keys())


def load_cfg(path: Optional[Path]) -> Dict[str, Any]:
    import yaml

    if path is None:
        return {}
    if not path.exists():
        raise PipelineError(
            f"Training config does not exist: {path}",
            hint="Check --cfg and the current working directory, or omit --cfg if you want a pure CLI-driven run.",
        )
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise PipelineError(
            f"Config file must contain a YAML mapping: {path}",
            hint="Top-level YAML must be a key-value mapping, not a list or scalar.",
        )
    return cfg


def pick(name: str, override_value: Any, cfg: Mapping[str, Any], *, required: bool = False) -> Any:
    if override_value is not None:
        return override_value
    if name in cfg:
        return cfg[name]
    if required:
        raise PipelineError(
            f"Missing required parameter '{name}'.",
            hint=f"Provide --{name} on CLI or define '{name}:' in the training config YAML.",
        )
    return DEFAULTS[name]


def parse_classes_value(raw: Any) -> List[int] | None:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple, set)):
        out = sorted({int(value) for value in raw})
        return out or None

    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None

    out: List[int] = []
    for token in text.split(","):
        current = token.strip()
        if not current:
            continue
        if current.startswith("+"):
            current = current[1:]
        if not current.isdigit():
            raise PipelineError(
                f"Invalid classes token '{token}'.",
                hint="Use comma-separated non-negative integers, for example: --classes 0,2,5",
            )
        out.append(int(current))
    return sorted(set(out)) or None


def auto_device(device_value: Any) -> Any:
    import torch

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()
    if device_value is not None:
        text = str(device_value).strip()
        if text == "":
            if cuda_available:
                return 0
            if mps_available:
                return "mps"
            return "cpu"
        if text.lower() == "mps":
            return "mps" if mps_available else "cpu"
        if text.isdigit():
            return int(text) if cuda_available else "cpu"
        if text.lower().startswith("cuda") and not cuda_available:
            return "cpu"
        return text
    if cuda_available:
        return 0
    if mps_available:
        return "mps"
    return "cpu"


def _is_cuda_device(device: Any) -> bool:
    import torch

    if device is None:
        return torch.cuda.is_available()
    if isinstance(device, int):
        return True
    text = str(device).lower()
    return text.isdigit() or text.startswith("cuda")


def setup_torch_backends(device: Any) -> None:
    import torch

    if torch.cuda.is_available() and _is_cuda_device(device):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def find_unsupported_cfg_keys(cfg: Mapping[str, Any], supported_train_keys: Set[str]) -> List[str]:
    unknown: List[str] = []
    for key, value in cfg.items():
        if key in CFG_META_KEYS or value is None:
            continue
        if key not in supported_train_keys:
            unknown.append(str(key))
    return sorted(unknown)


def build_training_plan(cfg: Mapping[str, Any], overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    import torch

    overrides = dict(overrides or {})
    supported_train_keys = _ultra_supported_train_keys()

    model_value = overrides.get("model") if overrides.get("model") is not None else cfg.get("model")
    data_value = overrides.get("data") if overrides.get("data") is not None else cfg.get("data")
    if not model_value:
        raise PipelineError(
            "Missing required parameter 'model'.",
            hint="Provide --model or define 'model:' in the training config YAML.",
        )
    if not data_value:
        raise PipelineError(
            "Missing required parameter 'data'.",
            hint="Provide --data or define 'data:' in the training config YAML.",
        )

    device_value = auto_device(pick("device", overrides.get("device"), cfg))
    setup_torch_backends(device_value)

    amp_value = pick("amp", overrides.get("amp"), cfg)
    val_value = pick("val", overrides.get("val"), cfg)
    classes_value = parse_classes_value(overrides.get("classes"))
    if classes_value is None and "classes" in cfg:
        classes_value = parse_classes_value(cfg.get("classes"))

    project_value = str(Path(pick("project", overrides.get("project"), cfg)).expanduser().resolve())
    train_kwargs: Dict[str, Any] = {
        "data": str(data_value),
        "epochs": pick("epochs", overrides.get("epochs"), cfg),
        "imgsz": pick("imgsz", overrides.get("imgsz"), cfg),
        "batch": pick("batch", overrides.get("batch"), cfg),
        "device": device_value,
        "workers": pick("workers", overrides.get("workers"), cfg),
        "project": project_value,
        "name": pick("name", overrides.get("name"), cfg),
        "seed": pick("seed", overrides.get("seed"), cfg),
        "amp": amp_value,
        "cache": pick("cache", overrides.get("cache"), cfg),
        "compile": pick("compile", overrides.get("compile"), cfg),
        "val": val_value,
        "classes": classes_value,
        "exist_ok": pick("exist_ok", overrides.get("exist_ok"), cfg),
        "verbose": pick("verbose", overrides.get("verbose"), cfg),
    }

    for key, value in cfg.items():
        if key in CFG_META_KEYS or value is None or key in train_kwargs:
            continue
        if key in supported_train_keys:
            train_kwargs[key] = value

    train_kwargs = {key: value for key, value in train_kwargs.items() if (key in supported_train_keys or key == "data")}
    warnings: List[str] = []
    for key in find_unsupported_cfg_keys(cfg, supported_train_keys):
        warnings.append(f"Unsupported Ultralytics train key in config: {key}")

    env_info = {
        "device": device_value,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }
    if torch.cuda.is_available():
        try:
            env_info["cuda_device_0"] = torch.cuda.get_device_name(0)
        except Exception:
            pass

    if overrides.get("device") == "mps" and env_info["torch_mps_available"] is False:
        warnings.append("Requested device='mps', but MPS is not available; using CPU.")
    raw_device = overrides.get("device")
    if raw_device is not None:
        raw_device_text = str(raw_device).strip()
        if raw_device_text.isdigit() and env_info["torch_cuda_available"] is False:
            warnings.append(f"Requested CUDA device='{raw_device_text}', but CUDA is not available; using CPU.")
        if raw_device_text.lower().startswith("cuda") and env_info["torch_cuda_available"] is False:
            warnings.append(f"Requested device='{raw_device_text}', but CUDA is not available; using CPU.")

    return {
        "model": str(model_value),
        "data": str(data_value),
        "device": device_value,
        "train_kwargs": train_kwargs,
        "warnings": warnings,
        "env_info": env_info,
    }


def run_training(model_path: str, train_kwargs: Mapping[str, Any]) -> Any:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    try:
        return model.train(**dict(train_kwargs))
    except TypeError as exc:
        message = str(exc)
        if "unexpected keyword argument" in message:
            raise PipelineError(
                "Ultralytics rejected one of the train args.",
                hint="Remove the mentioned key from YAML or align the installed ultralytics version with the config.",
                details=f"Original error: {message}",
            ) from exc
        raise
