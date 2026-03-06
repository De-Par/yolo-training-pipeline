from __future__ import annotations

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
