import numpy as np

from motion_capture.models import PoseSample
from motion_capture.ui.open3d_view import pose_transform_mm, trajectory_segments


def test_pose_transform_preserves_rotation_and_millimetres() -> None:
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    sample = PoseSample("test", "tool", 1, (12.5, -8.0, 640.0), rotation)

    transform = pose_transform_mm(sample)

    assert np.allclose(transform[:3, :3], rotation)
    assert np.allclose(transform[:3, 3], (12.5, -8.0, 640.0))
    assert np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0))


def test_trajectory_segments_connect_consecutive_points() -> None:
    assert trajectory_segments(1).shape == (0, 2)
    assert np.array_equal(trajectory_segments(4), np.array([[0, 1], [1, 2], [2, 3]]))
