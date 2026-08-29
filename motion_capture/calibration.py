from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from motion_capture.models import PoseSample


@dataclass(frozen=True)
class CalibrationResult:
    reference_transform: np.ndarray
    sample_count: int
    position_rms_mm: float
    max_angle_deviation_deg: float


def pose_matrix(sample: PoseSample) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(sample.rotation_matrix, dtype=np.float64)
    transform[:3, 3] = np.asarray(sample.position_mm, dtype=np.float64)
    return transform


def average_initial_pose(samples: Iterable[PoseSample]) -> CalibrationResult:
    """Average stationary pose samples and report translational/rotational spread."""
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("初始标定至少需要一个有效位姿样本")
    transforms = np.stack([pose_matrix(sample) for sample in sample_list])
    positions = transforms[:, :3, 3]
    mean_position = positions.mean(axis=0)

    mean_rotation = transforms[:, :3, :3].mean(axis=0)
    u, _, vt = np.linalg.svd(mean_rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    reference = np.eye(4, dtype=np.float64)
    reference[:3, :3] = rotation
    reference[:3, 3] = mean_position
    position_rms = float(np.sqrt(np.mean(np.sum((positions - mean_position) ** 2, axis=1))))

    angle_deviations: list[float] = []
    for sample_rotation in transforms[:, :3, :3]:
        delta = rotation.T @ sample_rotation
        cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
        angle_deviations.append(float(np.degrees(np.arccos(cosine))))
    return CalibrationResult(
        reference_transform=reference,
        sample_count=len(sample_list),
        position_rms_mm=position_rms,
        max_angle_deviation_deg=max(angle_deviations),
    )


def apply_reference_pose(sample: PoseSample, reference_transform: Sequence[float]) -> PoseSample:
    """Express a camera-space pose relative to the saved initial reference pose."""
    reference = np.asarray(reference_transform, dtype=np.float64).reshape(4, 4)
    relative = np.linalg.inv(reference) @ pose_matrix(sample)
    return PoseSample(
        source=sample.source,
        tool_id=sample.tool_id,
        frame_number=sample.frame_number,
        position_mm=tuple(float(value) for value in relative[:3, 3]),
        rotation_matrix=relative[:3, :3].copy(),
        quality=sample.quality,
        valid=sample.valid,
        timestamp_utc=sample.timestamp_utc,
    )


def is_identity_reference(reference_transform: Sequence[float]) -> bool:
    matrix = np.asarray(reference_transform, dtype=np.float64).reshape(4, 4)
    return bool(np.allclose(matrix, np.eye(4), atol=1e-9))
