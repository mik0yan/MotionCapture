from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import TextIO

import numpy as np

from motion_capture.models import PoseSample


class TrajectoryRecorder:
    BASE_HEADER = (
        "timestamp_utc",
        "source",
        "tool_id",
        "frame_number",
        "valid",
        "quality",
        "x_mm",
        "y_mm",
        "z_mm",
    )
    QUATERNION_HEADER = ("qw", "qx", "qy", "qz")
    MATRIX_HEADER = tuple(f"r{row}{column}" for row in range(1, 4) for column in range(1, 4))

    def __init__(
        self,
        directory: Path,
        file_format: str = "csv",
        orientation: str = "quaternion",
        xlsx_autosave_rows: int = 250,
    ) -> None:
        self.directory = directory
        self.file_format = ""
        self.orientation = ""
        self.xlsx_autosave_rows = max(1, int(xlsx_autosave_rows))
        self.path: Path | None = None
        self._file: TextIO | None = None
        self._csv_writer: csv.writer | None = None
        self._workbook = None
        self._worksheet = None
        self._xlsx_rows_since_save = 0
        self.configure(file_format, orientation)

    @property
    def active(self) -> bool:
        return self._file is not None or self._workbook is not None

    @property
    def header(self) -> tuple[str, ...]:
        if self.orientation == "position":
            orientation_header: tuple[str, ...] = ()
        elif self.orientation == "quaternion":
            orientation_header = self.QUATERNION_HEADER
        else:
            orientation_header = self.MATRIX_HEADER
        return self.BASE_HEADER + orientation_header

    def configure(self, file_format: str, orientation: str) -> None:
        if self.active:
            raise RuntimeError("录制期间不能修改轨迹文件格式")
        normalized_format = file_format.strip().lower()
        normalized_orientation = orientation.strip().lower()
        if normalized_format not in {"csv", "xlsx"}:
            raise ValueError("轨迹文件格式仅支持 csv 或 xlsx")
        if normalized_orientation not in {"position", "quaternion", "matrix"}:
            raise ValueError("轨迹数据格式仅支持 position、quaternion 或 matrix")
        self.file_format = normalized_format
        self.orientation = normalized_orientation

    def start(self) -> Path:
        if self.active:
            raise RuntimeError("轨迹录制已经开始")
        self.directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.path = self.directory / f"motion_{timestamp}.{self.file_format}"
        if self.file_format == "csv":
            self._file = self.path.open("w", newline="", encoding="utf-8-sig")
            self._csv_writer = csv.writer(self._file)
            self._csv_writer.writerow(self.header)
            self._file.flush()
        else:
            try:
                from openpyxl import Workbook
            except ImportError as exc:
                raise RuntimeError("XLSX 录制需要安装 openpyxl") from exc
            self._workbook = Workbook()
            self._worksheet = self._workbook.active
            self._worksheet.title = "trajectory"
            self._worksheet.freeze_panes = "A2"
            self._worksheet.append(self.header)
            self._workbook.save(self.path)
            self._xlsx_rows_since_save = 0
        return self.path

    def write(self, samples: tuple[PoseSample, ...]) -> None:
        if not self.active:
            return
        if self.file_format == "csv":
            for sample in samples:
                self._csv_writer.writerow(self._sample_row(sample, formatted=True))
            self._file.flush()
            return

        for sample in samples:
            self._worksheet.append(self._sample_row(sample, formatted=False))
            self._xlsx_rows_since_save += 1
        if self._xlsx_rows_since_save >= self.xlsx_autosave_rows:
            self._workbook.save(self.path)
            self._xlsx_rows_since_save = 0

    def stop(self) -> Path | None:
        path = self.path
        if self._file is not None:
            self._file.close()
        if self._workbook is not None:
            self._workbook.save(self.path)
            self._workbook.close()
        self._file = None
        self._csv_writer = None
        self._workbook = None
        self._worksheet = None
        self._xlsx_rows_since_save = 0
        return path

    def _sample_row(self, sample: PoseSample, formatted: bool) -> tuple[object, ...]:
        quality: object = "" if sample.quality is None else sample.quality
        position: tuple[object, ...] = tuple(float(value) for value in sample.position_mm)
        if self.orientation == "position":
            orientation: tuple[object, ...] = ()
        elif self.orientation == "quaternion":
            orientation: tuple[object, ...] = sample.quaternion_wxyz
        else:
            orientation = tuple(float(value) for value in np.asarray(sample.rotation_matrix).reshape(-1))
        if formatted:
            quality = "" if sample.quality is None else f"{sample.quality:.6f}"
            position = tuple(f"{float(value):.6f}" for value in sample.position_mm)
            orientation = tuple(f"{float(value):.9f}" for value in orientation)
        return (
            sample.timestamp_utc.isoformat(),
            sample.source,
            sample.tool_id,
            sample.frame_number,
            int(sample.valid),
            quality,
            *position,
            *orientation,
        )


class CSVRecorder(TrajectoryRecorder):
    """Backward-compatible default recorder used by earlier integrations."""

    def __init__(self, directory: Path) -> None:
        super().__init__(directory, file_format="csv", orientation="quaternion")
