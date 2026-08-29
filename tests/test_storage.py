from dataclasses import replace
from pathlib import Path

import numpy as np

from motion_capture.config import load_config
from motion_capture.models import PoseSample
from motion_capture.storage import AppDatabase, TagMonitorProfile


def test_sqlite_config_is_created_and_round_trips(tmp_path: Path) -> None:
    config = load_config(tmp_path, environ={})
    with AppDatabase(config.database_path) as database:
        first = database.load_config(config)
        updated = replace(first, source="realsense", poll_hz=42)
        database.save_config(updated)
        loaded = database.load_config(config)

    assert config.database_path == tmp_path / "data/motion_capture.sqlite3"
    assert loaded.source == "realsense"
    assert loaded.poll_hz == 42


def test_sqlite_monitor_profiles_and_capture_samples(tmp_path: Path) -> None:
    config = load_config(tmp_path, environ={})
    with AppDatabase(config.database_path) as database:
        database.ensure_tag_monitors((3, 7), 24.0)
        profiles = database.list_tag_monitors((3, 7))
        database.save_tag_monitor(
            TagMonitorProfile(7, 30.0, 0.65, 75.0, False)
        )
        session_id = database.start_session("simulator", "测试设备")
        database.record_samples(
            session_id,
            (PoseSample("simulator", "tag_07", 8, (1.0, 2.0, 3.0), np.eye(3), 0.9),),
        )
        database.end_session(session_id, status="completed")

        updated = database.list_tag_monitors((7,))[0]
        count = database.sample_count(session_id)

    assert tuple(profile.tag_id for profile in profiles) == (3, 7)
    assert updated.tag_size_mm == 30.0
    assert updated.quality_threshold == 0.65
    assert updated.enabled is False
    assert count == 1
