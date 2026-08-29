import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace

from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from motion_capture.storage import TagMonitorProfile
from motion_capture.ui.monitor_widgets import TagMonitorCard


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

