from __future__ import annotations

import cv2
import numpy as np
import pytest

from tools.apriltag_live import AprilTagDetector, TagObservation

TAG_SIZE_M = 0.080
WIDTH, HEIGHT = 1280, 720
CAMERA_MATRIX = np.array([[900.0, 0.0, WIDTH / 2], [0.0, 900.0, HEIGHT / 2], [0.0, 0.0, 1.0]], dtype=np.float64)
DIST_COEFFS = np.zeros(5, dtype=np.float64)


def render_tag(
    tag_id: int,
    position_m: tuple[float, float, float],
    tilt_rvec: tuple[float, float, float],
    tag_size_m: float = TAG_SIZE_M,
) -> np.ndarray:
    """把一个 AprilTag 渲染到指定的相机坐标位置，用于反解验证。"""
    half = tag_size_m / 2.0
    border_ratio = 0.25
    scale = 1.0 + 2 * border_ratio
    outer = half * scale
    object_points = np.array(
        [[-outer, outer, 0.0], [outer, outer, 0.0], [outer, -outer, 0.0], [-outer, -outer, 0.0]],
        dtype=np.float32,
    )

    # 相机 Y 轴向下，标记正对相机需绕 X 轴翻转 180°，否则贴图上下颠倒无法解码。
    facing = cv2.Rodrigues(np.array([np.pi, 0.0, 0.0]))[0]
    tilt = cv2.Rodrigues(np.array(tilt_rvec, dtype=np.float64))[0]
    rvec = cv2.Rodrigues(facing @ tilt)[0]
    tvec = np.array(position_m, dtype=np.float64).reshape(3, 1)

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)
    destination = projected.reshape(4, 2).astype(np.float32)

    # 按投影后的实际像素尺寸生成贴图，避免大幅下采样糊掉 tag 边缘。
    side = int(max(np.linalg.norm(destination[i] - destination[(i + 1) % 4]) for i in range(4)))
    inner = max(40, int(side / scale))
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, inner)
    pad = int(inner * border_ratio)
    marker = cv2.copyMakeBorder(marker, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

    size = marker.shape[0]
    source = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source, destination)
    scene = cv2.warpPerspective(marker, transform, (WIDTH, HEIGHT), borderValue=255)
    return cv2.cvtColor(scene, cv2.COLOR_GRAY2RGB)


@pytest.fixture
def detector() -> AprilTagDetector:
    return AprilTagDetector("tag36h11", TAG_SIZE_M, CAMERA_MATRIX, DIST_COEFFS)


@pytest.mark.parametrize(
    "tag_id, position_m, tilt_rvec",
    [
        (7, (0.0, 0.0, 0.800), (0.0, 0.0, 0.0)),
        (3, (0.050, -0.030, 0.600), (0.25, -0.15, 0.10)),
        (0, (-0.100, 0.060, 1.200), (0.10, 0.20, 0.0)),
    ],
)
def test_recovers_tag_id_and_position(detector, tag_id, position_m, tilt_rvec) -> None:
    scene = render_tag(tag_id, position_m, tilt_rvec)
    observations, _ = detector.detect(scene, annotate=False)

    assert len(observations) == 1
    observation = observations[0]
    assert observation.tag_id == tag_id

    expected_mm = np.array(position_m) * 1000.0
    error_mm = float(np.linalg.norm(np.array(observation.position_mm) - expected_mm))
    distance_mm = float(np.linalg.norm(expected_mm))
    # 远距离下 tag 占的像素变少，量化误差按比例放大，因此用相对判据。
    assert error_mm / distance_mm < 0.02
    assert observation.reprojection_error_px < 2.0


def test_returns_empty_without_tags(detector) -> None:
    blank = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    observations, canvas = detector.detect(blank, annotate=False)
    assert observations == []
    assert canvas.shape == blank.shape


def test_rejects_unsupported_family() -> None:
    with pytest.raises(ValueError, match="不支持的 AprilTag family"):
        AprilTagDetector("tag99h1", TAG_SIZE_M, CAMERA_MATRIX, DIST_COEFFS)


def test_observation_line_contains_id_and_distance() -> None:
    observation = TagObservation(
        tag_id=2,
        center_px=(640.0, 360.0),
        position_mm=(10.0, -20.0, 800.0),
        distance_mm=800.3,
        reprojection_error_px=0.4,
    )
    line = observation.format_line()
    assert "id=  2" in line
    assert "800.3" in line


from tools.apriltag_live import SlidingWindowStats, annotate_statistics


def observation_at(tag_id: int, position_mm: tuple[float, float, float]) -> TagObservation:
    return TagObservation(
        tag_id=tag_id,
        center_px=(640.0, 360.0),
        position_mm=position_mm,
        distance_mm=float(np.linalg.norm(position_mm)),
        reprojection_error_px=0.1,
    )


def test_variance_is_zero_for_static_tag() -> None:
    stats = SlidingWindowStats(window_s=5.0)
    for index in range(10):
        stats.update(index * 0.1, [observation_at(0, (10.0, -20.0, 800.0))])
    summary = stats.summary(0)
    assert summary is not None
    assert summary.variance_mm2 == pytest.approx((0.0, 0.0, 0.0))
    assert summary.sigma_mm == pytest.approx(0.0)
    assert summary.detection_rate == pytest.approx(1.0)


def test_variance_matches_numpy_reference() -> None:
    rng = np.random.default_rng(1234)
    positions = rng.normal(loc=(10.0, -20.0, 800.0), scale=(1.0, 2.0, 3.0), size=(400, 3))
    stats = SlidingWindowStats(window_s=100.0)
    for index, position in enumerate(positions):
        stats.update(index * 0.01, [observation_at(0, tuple(position))])

    summary = stats.summary(0)
    assert summary is not None
    expected = positions.var(axis=0, ddof=1)
    assert summary.variance_mm2 == pytest.approx(tuple(expected), rel=1e-9)
    assert summary.sigma_mm == pytest.approx(float(np.sqrt(expected.sum())), rel=1e-9)
    assert summary.mean_mm == pytest.approx(tuple(positions.mean(axis=0)), rel=1e-9)


def test_window_expires_old_samples() -> None:
    stats = SlidingWindowStats(window_s=5.0)
    # 前 5 秒在一个位置，后 5 秒换到另一个位置
    for index in range(50):
        stats.update(index * 0.1, [observation_at(0, (0.0, 0.0, 500.0))])
    for index in range(50, 101):
        stats.update(index * 0.1, [observation_at(0, (0.0, 0.0, 900.0))])

    summary = stats.summary(0)
    assert summary is not None
    # 旧位置已滑出窗口，均值应完全落在新位置上
    assert summary.mean_mm[2] == pytest.approx(900.0)
    assert summary.variance_mm2[2] == pytest.approx(0.0)
    assert summary.frames <= 51


def test_detection_rate_reflects_missing_frames() -> None:
    stats = SlidingWindowStats(window_s=5.0)
    for index in range(10):
        observations = [observation_at(0, (0.0, 0.0, 700.0))] if index % 2 == 0 else []
        stats.update(index * 0.1, observations)
    summary = stats.summary(0)
    assert summary is not None
    assert summary.samples == 5
    assert summary.frames == 10
    assert summary.detection_rate == pytest.approx(0.5)


def test_tag_disappears_from_window() -> None:
    stats = SlidingWindowStats(window_s=1.0)
    stats.update(0.0, [observation_at(4, (0.0, 0.0, 600.0))])
    assert stats.tracked_ids() == [4]
    stats.update(2.0, [])
    assert stats.tracked_ids() == []
    assert stats.summary(4) is None


def test_annotate_statistics_draws_without_error() -> None:
    stats = SlidingWindowStats(window_s=5.0)
    observations = [observation_at(0, (10.0, -20.0, 800.0))]
    for index in range(5):
        stats.update(index * 0.1, observations)
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    annotate_statistics(canvas, observations, stats)
    assert canvas.any()


@pytest.mark.parametrize(
    "tag_id, position_m",
    [
        (1, (0.0, 0.0, 0.250)),
        (2, (0.030, -0.020, 0.400)),
    ],
)
def test_recovers_small_24mm_tag(tag_id, position_m) -> None:
    """实际使用的码是 24mm，远小于 80mm，单独覆盖近距离场景。"""
    small_size = 0.024
    detector = AprilTagDetector("tag36h11", small_size, CAMERA_MATRIX, DIST_COEFFS)
    scene = render_tag(tag_id, position_m, (0.1, -0.05, 0.0), tag_size_m=small_size)
    observations, _ = detector.detect(scene, annotate=False)

    assert len(observations) == 1
    assert observations[0].tag_id == tag_id
    expected_mm = np.array(position_m) * 1000.0
    error_mm = float(np.linalg.norm(np.array(observations[0].position_mm) - expected_mm))
    assert error_mm / float(np.linalg.norm(expected_mm)) < 0.02
