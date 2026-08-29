from __future__ import annotations

import time

import numpy as np

from motion_capture.backends.base import TrackerBackend
from motion_capture.models import PoseSample, TrackingPacket


class SimulatorBackend(TrackerBackend):
    display_name = "模拟器"

    def __init__(self) -> None:
        self._started_at = 0.0
        self._frame = 0

    def start(self) -> None:
        self._started_at = time.monotonic()
        self._frame = 0

    def read(self) -> TrackingPacket:
        elapsed = time.monotonic() - self._started_at
        self._frame += 1
        x = 180.0 * np.sin(elapsed * 0.7)
        y = 90.0 * np.sin(elapsed * 1.1)
        z = 650.0 + 100.0 * np.cos(elapsed * 0.5)
        yaw = elapsed * 0.45
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        sample = PoseSample("simulator", "demo_tool", self._frame, (float(x), float(y), float(z)), rotation, 0.99)
        image = self._render_image(x, y, elapsed)
        return TrackingPacket((sample,), image)

    @staticmethod
    def _render_image(x: float, y: float, elapsed: float) -> np.ndarray:
        height, width = 480, 720
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :, :] = (16, 24, 38)
        image[::40, :, :] = (30, 48, 68)
        image[:, ::40, :] = (30, 48, 68)
        cx = int(width / 2 + x)
        cy = int(height / 2 + y)
        radius = 18 + int(4 * np.sin(elapsed * 2))
        yy, xx = np.ogrid[:height, :width]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        image[mask] = (34, 211, 166)
        return image

    def stop(self) -> None:
        pass
