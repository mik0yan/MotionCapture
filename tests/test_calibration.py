from datetime import datetime, timezone

import numpy as np

from motion_capture.calibration import apply_reference_pose, average_initial_pose
from motion_capture.models import PoseSample


def _rotation_z(degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    return np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )


def _sample(position, rotation, frame=1) -> PoseSample:
    return PoseSample(
        "realsense",
        "apriltag_board",
        frame,
        tuple(position),
        rotation,
        1.0,
        timestamp_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_average_initial_pose_projects_rotation_and_reports_stability() -> None:
    samples = [
        _sample((100.0 + delta, -40.0, 650.0), _rotation_z(30.0 + delta * 0.05), frame=index)
        for index, delta in enumerate((-0.4, -0.2, 0.0, 0.2, 0.4), start=1)
    ]

    result = average_initial_pose(samples)

    assert result.sample_count == 5
    assert np.allclose(result.reference_transform[:3, 3], (100.0, -40.0, 650.0))
    assert np.allclose(result.reference_transform[:3, :3], _rotation_z(30.0), atol=1e-5)
    assert result.position_rms_mm < 0.4
    assert result.max_angle_deviation_deg < 0.03


def test_apply_reference_pose_makes_initial_pose_the_origin() -> None:
    reference = np.eye(4)
    reference[:3, :3] = _rotation_z(25.0)
    reference[:3, 3] = (120.0, -30.0, 700.0)
    relative = np.eye(4)
    relative[:3, :3] = _rotation_z(10.0)
    relative[:3, 3] = (15.0, 5.0, -2.0)
    current = reference @ relative

    transformed = apply_reference_pose(
        _sample(current[:3, 3], current[:3, :3]),
        reference.reshape(-1),
    )

    assert np.allclose(transformed.position_mm, relative[:3, 3])
    assert np.allclose(transformed.rotation_matrix, relative[:3, :3])
