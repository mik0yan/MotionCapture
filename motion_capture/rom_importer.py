from __future__ import annotations

import filecmp
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ROMImportResult:
    paths: tuple[Path, ...]
    copied_count: int
    reused_count: int
    renamed_count: int


def import_rom_files(root: Path, source_files: Iterable[str | Path]) -> ROMImportResult:
    """Copy ROM files into root/roms without overwriting different existing tools."""
    root = root.resolve()
    sources = tuple(Path(value).expanduser().resolve() for value in source_files)
    if not sources:
        raise ValueError("请选择至少一个 ROM 文件")
    for source in sources:
        if source.suffix.lower() != ".rom":
            raise ValueError(f"NDI 工具配置必须是 .rom 文件：{source}")
        if not source.is_file():
            raise FileNotFoundError(f"找不到 ROM 文件：{source}")

    target_dir = root / "roms"
    target_dir.mkdir(parents=True, exist_ok=True)
    imported: list[Path] = []
    copied = 0
    reused = 0
    renamed = 0
    for source in sources:
        target = target_dir / source.name
        if source == target:
            reused += 1
        elif target.exists() and filecmp.cmp(source, target, shallow=False):
            reused += 1
        else:
            if target.exists():
                suffix_index = 2
                while True:
                    candidate = target_dir / f"{source.stem}_{suffix_index}{source.suffix.lower()}"
                    if not candidate.exists():
                        target = candidate
                        renamed += 1
                        break
                    if filecmp.cmp(source, candidate, shallow=False):
                        target = candidate
                        reused += 1
                        break
                    suffix_index += 1
            if not target.exists():
                shutil.copy2(source, target)
                copied += 1
        if target not in imported:
            imported.append(target)
    return ROMImportResult(tuple(imported), copied, reused, renamed)
