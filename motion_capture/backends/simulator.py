from __future__ import annotations

import time

import numpy as np

from motion_capture.backends.base import TrackerBackend
from motion_capture.config import RealSenseConfig
from motion_capture.models import PoseSample, TagDetection, TrackingPacket


class SimulatorBackend(TrackerBackend):
    display_name = "模拟器"

    def __init__(self, config: RealSenseConfig | None = None) -> None:
        self.config = config
        self._started_at = 0.0
        self._frame = 0

    def start(self) -> None:
        self._started_at = time.monotonic()
        self._frame = 0

    def read(self) -> TrackingPacket:
        elapsed = time.monotonic() - self._started_at
        self._frame += 1
        yaw = elapsed * 0.12
        rotation = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        configured_ids = self.config.tag_ids if self.config is not None else (3, 7, 12)
        tag_ids = tuple(configured_ids[:3]) or (3,)
        anchors = ((0.27, 0.30), (0.67, 0.24), (0.50, 0.67))
        base_positions = ((142.6, -38.4, 248.6), (84.2, 16.8, 196.4), (42.8, -12.5, 168.0))
        samples: list[PoseSample] = []
        detections: list[TagDetection] = []
        for index, tag_id in enumerate(tag_ids):
            phase = elapsed * (0.55 + index * 0.1)
            base_x, base_y, base_z = base_positions[index]
            position = (
                base_x + 2.4 * np.sin(phase),
                base_y + 1.8 * np.cos(phase * 0.9),
                base_z + 2.0 * np.sin(phase * 0.7),
            )
            quality = (0.96, 0.69, 0.58)[index]
            samples.append(
                PoseSample(
                    "simulator",
                    f"tag_{int(tag_id):02d}",
                    self._frame,
                    tuple(float(value) for value in position),
                    rotation,
                    quality,
                )
            )
            cx = anchors[index][0] * 1280 + 10 * np.sin(phase)
            cy = anchors[index][1] * 720 + 8 * np.cos(phase)
            size = 104.0
            detections.append(
                TagDetection(
                    int(tag_id),
                    (
                        (cx - size / 2, cy - size / 2),
                        (cx + size / 2, cy - size / 2),
                        (cx + size / 2, cy + size / 2),
                        (cx - size / 2, cy + size / 2),
                    ),
                    float(np.linalg.norm(position)),
                    quality,
                )
            )
        board_position = tuple(
            float(np.mean([sample.position_mm[axis] for sample in samples])) for axis in range(3)
        )
        samples.append(
            PoseSample(
                "simulator",
                "apriltag_board",
                self._frame,
                board_position,
                rotation,
                1.0,
            )
        )
        image = self._render_image(elapsed)
        return TrackingPacket(tuple(samples), image, tuple(detections))

    @staticmethod
    def _render_image(elapsed: float) -> np.ndarray:
        height, width = 720, 1280
        image = np.zeros((height, width, 3), dtype=np.uint8)
        y_gradient = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        image[:, :, 0] = (6 + y_gradient * 2).astype(np.uint8)
        image[:, :, 1] = (14 + y_gradient * 10).astype(np.uint8)
        image[:, :, 2] = (23 + y_gradient * 18).astype(np.uint8)
        cx = int(width * 0.5 + np.sin(elapsed * 0.2) * 30)
        cy = int(height * 0.64)
        radius = 280
        yy, xx = np.ogrid[:height, :width]
        glow = np.clip(1.0 - np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius, 0, 1)
        image[:, :, 2] = np.maximum(image[:, :, 2], (glow * 70).astype(np.uint8))
        return image

    def stop(self) -> None:
        pass
