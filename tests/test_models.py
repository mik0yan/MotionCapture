import numpy as np

from motion_capture.models import matrix_to_euler_xyz_degrees, matrix_to_quaternion


def test_identity_rotation_conversions() -> None:
    rotation = np.eye(3)
    assert np.allclose(matrix_to_quaternion(rotation), (1.0, 0.0, 0.0, 0.0))
    assert np.allclose(matrix_to_euler_xyz_degrees(rotation), (0.0, 0.0, 0.0))


def test_z_rotation_conversions() -> None:
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(matrix_to_quaternion(rotation), (2**-0.5, 0.0, 0.0, 2**-0.5))
    assert np.allclose(matrix_to_euler_xyz_degrees(rotation), (0.0, 0.0, 90.0))
