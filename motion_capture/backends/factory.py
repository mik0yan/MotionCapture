from __future__ import annotations

from collections.abc import Mapping

from motion_capture.backends.base import TrackerBackend
from motion_capture.backends.ndi import NDIBackend
from motion_capture.backends.realsense import RealSenseAprilTagBackend
from motion_capture.backends.simulator import SimulatorBackend
from motion_capture.config import AppConfig


def create_backend(
    source: str,
    config: AppConfig,
    tag_sizes_mm: Mapping[int, float] | None = None,
) -> TrackerBackend:
    if source == "simulator":
        return SimulatorBackend(config.realsense)
    if source == "realsense":
        return RealSenseAprilTagBackend(config.realsense, tag_sizes_mm=tag_sizes_mm)
    if source == "ndi":
        return NDIBackend(config.ndi)
    raise ValueError(f"未知数据源: {source}")
