from __future__ import annotations

import os
import shutil

from pathlib import Path


def safe_link_or_copy(src: Path, dst: Path, link_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link_mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def clean_split_dirs(images_out: Path, labels_out: Path) -> None:
    if images_out.exists():
        shutil.rmtree(images_out)
    if labels_out.exists():
        shutil.rmtree(labels_out)


def ensure_local_mplconfigdir() -> Path:
    mpl_cache_dir = (Path(".cache") / "matplotlib").resolve()
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))
    return mpl_cache_dir
