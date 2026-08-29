from __future__ import annotations

import numpy as np

from motion_capture.backends.base import TrackerBackend
from motion_capture.config import NDIConfig
from motion_capture.models import PoseSample, TrackingPacket


class NDIBackend(TrackerBackend):
    display_name = "NDI 光学跟踪"

    def __init__(self, config: NDIConfig) -> None:
        self.config = config
        self._tracker = None

    def start(self) -> None:
        try:
            from sksurgerynditracker.nditracker import NDITracker
        except ImportError as exc:
            raise RuntimeError("NDI 模式需要安装 scikit-surgerynditracker 及其 ndicapy 依赖") from exc
        if not self.config.rom_files:
            raise ValueError("NDI_ROM_FILES 至少需要配置一个 ROM 文件")
        missing = [str(path) for path in self.config.rom_files if not path.is_file()]
        if missing:
            raise FileNotFoundError("找不到 NDI ROM 文件: " + ", ".join(missing))

        settings: dict[str, object] = {
            "tracker type": self.config.tracker_type,
            "romfiles": [str(path) for path in self.config.rom_files],
        }
        if self.config.tracker_type == "vega":
            settings.update({"ip address": self.config.ip_address, "port": self.config.port})
        elif self.config.serial_port:
            settings["serial port"] = self.config.serial_port
        self._tracker = NDITracker(settings)
        self._tracker.start_tracking()

    def read(self) -> TrackingPacket:
        if self._tracker is None:
            raise RuntimeError("NDI 尚未启动")
        port_handles, _, frame_numbers, transforms, qualities = self._tracker.get_frame()
        samples: list[PoseSample] = []
        for index, transform in enumerate(transforms):
            matrix = np.asarray(transform, dtype=float)
            if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                continue
            handle = str(port_handles[index]) if index < len(port_handles) else str(index)
            frame = int(frame_numbers[index]) if index < len(frame_numbers) else -1
            quality_raw = qualities[index] if index < len(qualities) else None
            quality = float(quality_raw) if quality_raw is not None and np.isfinite(quality_raw) else None
            position = tuple(float(value) for value in matrix[:3, 3])
            samples.append(PoseSample("ndi", handle, frame, position, matrix[:3, :3].copy(), quality))
        return TrackingPacket(tuple(samples))

    def stop(self) -> None:
        if self._tracker is not None:
            try:
                self._tracker.stop_tracking()
            finally:
                self._tracker.close()
                self._tracker = None
