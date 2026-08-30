import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from motion_capture.models import TagDetection
from motion_capture.storage import TagMonitorProfile
from motion_capture.ui import monitor_widgets
from motion_capture.ui.monitor_widgets import (
    CameraCanvas,
    TagMonitorCard,
    TagMotionTracker,
)


def _detection(tag_id: int, center_x: float, center_y: float) -> TagDetection:
    half = 52.0
    return TagDetection(
        tag_id,
        (
            (center_x - half, center_y - half),
            (center_x + half, center_y - half),
            (center_x + half, center_y + half),
            (center_x - half, center_y + half),
        ),
        180.0,
        0.9,
    )


def test_tag_monitor_card_matches_compact_layout_and_long_press() -> None:
    app = QApplication.instance() or QApplication([])
    profile = TagMonitorProfile(7, 30.0, 0.65, 80.0, True)
    card = TagMonitorCard(profile)
    requested: list[int] = []
    card.settings_requested.connect(requested.append)
    card.update_values((18.7, -6.1, 213.4), 214.3, 0.31, "proximity")
    card.show()

    QTest.mousePress(card, Qt.LeftButton, pos=card.rect().center())
    QTest.qWait(550)
    QTest.mouseRelease(card, Qt.LeftButton, pos=card.rect().center())
    app.processEvents()

    assert card.height() == 92
    assert card.coordinates_label.text() == "X +18.7  ·  Y -6.1  ·  Z +213.4"
    assert card.offset_label.text() == "偏移量：214.3 mm ± 0.31"
    assert requested == [7]


def test_tag_monitor_card_threshold_spin_types_and_steps() -> None:
    app = QApplication.instance() or QApplication([])
    profile = TagMonitorProfile(3, 30.0, 0.72, 80.0, True)
    card = TagMonitorCard(profile)
    emitted: list[tuple[int, float]] = []
    card.threshold_changed.connect(lambda tag_id, value: emitted.append((tag_id, value)))

    card.threshold_spin.setValue(0.85)
    app.processEvents()

    assert card.threshold_spin.value() == 0.85
    assert card.profile.quality_threshold == 0.85
    assert emitted == [(3, 0.85)]

    card.threshold_spin.setValue(2.5)
    assert card.profile.quality_threshold == 2.5

    card.set_profile(replace(profile, quality_threshold=0.5))
    assert card.threshold_spin.value() == 0.5
    assert emitted == [(3, 0.85), (3, 2.5)]


def test_tag_motion_tracker_measures_displacement_from_initial_position() -> None:
    tracker = TagMotionTracker()
    for step in range(10):
        now = step / 30.0
        tracker.update(now, (_detection(3, 300.0 + 60.0 * now, 200.0),))

    displacement = tracker.displacement(3)

    assert displacement is not None
    # 当前位置取最近 0.2 s 均值：末段窗口（t=0.1..0.3）中心约在 300 + 60*0.2 = 312
    assert displacement.dx_px == pytest.approx(12.0, abs=1.0)
    assert displacement.dy_px == pytest.approx(0.0, abs=1.0)


def test_tag_motion_tracker_stationary_tag_has_no_displacement() -> None:
    tracker = TagMotionTracker()
    for step in range(10):
        tracker.update(step / 30.0, (_detection(7, 500.0, 300.0),))

    displacement = tracker.displacement(7)

    assert displacement is not None
    assert displacement.distance_px == pytest.approx(0.0, abs=0.5)


def test_tag_motion_tracker_resets_initial_position_when_tag_disappears() -> None:
    tracker = TagMotionTracker()
    for step in range(6):
        tracker.update(step / 30.0, (_detection(3, 100.0, 100.0),))
    assert tracker.displacement(3) is not None

    tracker.update(0.3, ())
    tracker.update(0.34, (_detection(3, 160.0, 100.0),))

    # 基准重建：单样本不足以给出平滑后的当前位置
    assert tracker.displacement(3) is None


def test_tag_motion_tracker_reset_tag_rebuilds_initial_position() -> None:
    tracker = TagMotionTracker()
    for step in range(6):
        tracker.update(step / 30.0, (_detection(3, 100.0, 100.0),))
    tracker.update(0.3, (_detection(3, 140.0, 100.0),))
    assert tracker.displacement(3).dx_px == pytest.approx(10.0, abs=0.5)

    tracker.reset_tag(3)
    tracker.update(0.34, (_detection(3, 140.0, 100.0),))
    tracker.update(0.38, (_detection(3, 140.0, 100.0),))

    rebuilt = tracker.displacement(3)
    assert rebuilt is not None
    assert rebuilt.dx_px == pytest.approx(0.0, abs=0.5)


def test_camera_canvas_feeds_direction_layer_from_frame_stream(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    canvas = CameraCanvas()
    clock = {"now": 100.0}
    monkeypatch.setattr(monitor_widgets.time, "monotonic", lambda: clock["now"])

    for step in range(10):
        canvas.set_frame(None, (_detection(5, 200.0 + 60.0 * step / 30.0, 120.0),), {}, {5})
        clock["now"] += 1.0 / 30.0

    displacement = canvas.frame.displacements[5]
    assert displacement is not None
    assert displacement.dx_px > 8.0
    assert canvas.frame.direction_tags == frozenset({5})
    assert canvas.show_direction is True

    canvas.set_layer("direction", False)
    assert canvas.show_direction is False

    canvas.motion_tracker.reset()
    canvas.set_frame(None, (), {})
    assert canvas.frame.displacements == {}
    assert canvas.frame.direction_tags == frozenset()

