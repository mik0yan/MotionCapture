from __future__ import annotations

from motion_capture.backends.base import TrackerBackend
from motion_capture.backends.ndi import NDIBackend
from motion_capture.backends.realsense import RealSenseAprilTagBackend
from motion_capture.backends.simulator import SimulatorBackend
from motion_capture.config import AppConfig


def create_backend(source: str, config: AppConfig) -> TrackerBackend:
    if source == "simulator":
        return SimulatorBackend()
    if source == "realsense":
        return RealSenseAprilTagBackend(config.realsense)
    if source == "ndi":
        return NDIBackend(config.ndi)
    raise ValueError(f"未知数据源: {source}")
