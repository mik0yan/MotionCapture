import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from motion_capture.config import load_config
from motion_capture.models import PoseSample, TagDetection, TrackingPacket
from motion_capture.storage import AppDatabase
from motion_capture.ui import main_window as main_window_module
from motion_capture.ui import monitor_widgets
from motion_capture.ui.main_window import MainWindow


def _packet(positions: dict[int, tuple[float, float, float]], centers: dict[int, tuple[float, float]]) -> TrackingPacket:
    samples = tuple(
        PoseSample("realsense", f"tag_{tag_id:02d}", 1, position, np.eye(3), 0.99)
        for tag_id, position in positions.items()
    )
    detections = tuple(
        TagDetection(tag_id, ((cx - 52, cy - 52), (cx + 52, cy - 52), (cx + 52, cy + 52), (cx - 52, cy + 52)), 200.0, 0.99)
        for tag_id, (cx, cy) in centers.items()
    )
    return TrackingPacket(samples, np.zeros((720, 1280, 3), dtype=np.uint8), detections)


def test_direction_arrow_gated_by_tag_threshold(tmp_path: Path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    (tmp_path / ".env").write_text(
        "MOCAP_SOURCE=realsense\n"
        "APRILTAG_IDS=0,1,2,3\n"
        "APRILTAG_BOARD_ROWS=2\n"
        "APRILTAG_BOARD_COLS=2\n",
        encoding="utf-8",
    )
    config = load_config(tmp_path, environ={})
    database = AppDatabase(tmp_path / "test.sqlite3")
    window = MainWindow(config, database=database)

    clock = {"now": 100.0}
    monkeypatch.setattr(main_window_module.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(monitor_widgets.time, "monotonic", lambda: clock["now"])
    try:
        threshold = window._tag_profiles[0].quality_threshold

        # 首帧建立基准：偏移 0，低于阈值，不显示箭头
        window._on_packet(_packet({0: (0.0, 0.0, 200.0), 1: (0.0, 0.0, 200.0)}, {0: (200.0, 200.0), 1: (600.0, 300.0)}))
        assert 0 not in window._tag_direction_ids
        assert window.camera_view.canvas.frame.direction_tags == frozenset()

        # tag_0 偏移 30 mm（超过阈值）且画面位移 40 px；tag_1 保持基准
        clock["now"] += 0.5
        window._on_packet(
            _packet(
                {0: (30.0, 0.0, 200.0), 1: (0.0, 0.0, 200.0)},
                {0: (240.0, 200.0), 1: (600.0, 300.0)},
            )
        )
        assert 30.0 >= threshold
        assert window._tag_direction_ids == frozenset({0})
        canvas = window.camera_view.canvas
        assert canvas.frame.direction_tags == frozenset({0})
        assert canvas.frame.displacements[0].dx_px == 40.0
        assert canvas.frame.displacements[1].distance_px == 0.0

        # 持续回到阈值内：5 s 判读窗口滚动排除早期偏移样本后箭头消失
        for step in range(6):
            clock["now"] += 1.2
            window._on_packet(
                _packet(
                    {0: (1.0, 0.0, 200.0), 1: (0.0, 0.0, 200.0)},
                    {0: (204.0, 200.0), 1: (600.0, 300.0)},
                )
            )
        assert window._tag_direction_ids == frozenset()
        assert canvas.frame.direction_tags == frozenset()
    finally:
        database.close()
