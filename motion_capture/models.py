from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np


def matrix_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to normalized (w, x, y, z)."""
    m = np.asarray(matrix, dtype=float)
    trace = float(np.trace(m))
    if trace > 0:
        s = (trace + 1.0) ** 0.5 * 2
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = (1.0 + m[0, 0] - m[1, 1] - m[2, 2]) ** 0.5 * 2
            q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif i == 1:
            s = (1.0 + m[1, 1] - m[0, 0] - m[2, 2]) ** 0.5 * 2
            q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = (1.0 + m[2, 2] - m[0, 0] - m[1, 1]) ** 0.5 * 2
            q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    norm = float(np.linalg.norm(q))
    return tuple(float(value) for value in (q / norm))


def matrix_to_euler_xyz_degrees(matrix: np.ndarray) -> tuple[float, float, float]:
    """Return intrinsic XYZ Euler angles in degrees."""
    m = np.asarray(matrix, dtype=float)
    sy = float(np.hypot(m[0, 0], m[1, 0]))
    singular = sy < 1e-8
    if not singular:
        x = np.arctan2(m[2, 1], m[2, 2])
        y = np.arctan2(-m[2, 0], sy)
        z = np.arctan2(m[1, 0], m[0, 0])
    else:
        x = np.arctan2(-m[1, 2], m[1, 1])
        y = np.arctan2(-m[2, 0], sy)
        z = 0.0
    return tuple(float(value) for value in np.degrees([x, y, z]))


@dataclass(frozen=True)
class PoseSample:
    source: str
    tool_id: str
    frame_number: int
    position_mm: tuple[float, float, float]
    rotation_matrix: np.ndarray = field(repr=False)
    quality: float | None = None
    valid: bool = True
    timestamp_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def quaternion_wxyz(self) -> tuple[float, float, float, float]:
        return matrix_to_quaternion(self.rotation_matrix)

    @property
    def euler_xyz_degrees(self) -> tuple[float, float, float]:
        return matrix_to_euler_xyz_degrees(self.rotation_matrix)


@dataclass(frozen=True)
class TagDetection:
    tag_id: int
    corners_px: tuple[tuple[float, float], ...]
    distance_mm: float
    quality: float | None = None


@dataclass(frozen=True)
class TrackingPacket:
    samples: tuple[PoseSample, ...]
    image_rgb: np.ndarray | None = field(default=None, repr=False)
    tag_detections: tuple[TagDetection, ...] = ()
