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


def test_enable_all_tag_monitors_resets_stopped_cards(tmp_path: Path) -> None:
    config = load_config(tmp_path, environ={})
    with AppDatabase(config.database_path) as database:
        database.ensure_tag_monitors((1, 2), 24.0)
        database.save_tag_monitor(TagMonitorProfile(1, 24.0, 0.72, 80.0, False))
        database.save_tag_monitor(TagMonitorProfile(2, 24.0, 0.72, 80.0, False))

        database.enable_all_tag_monitors()
        enabled = {p.tag_id: p.enabled for p in database.list_tag_monitors()}

    assert enabled == {1: True, 2: True}


def test_v1_database_migrates_to_unbounded_quality_threshold(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "data/legacy.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE tag_monitors (
            tag_id INTEGER PRIMARY KEY,
            tag_size_mm REAL NOT NULL CHECK(tag_size_mm > 0),
            quality_threshold REAL NOT NULL CHECK(quality_threshold >= 0 AND quality_threshold <= 1),
            warning_distance_mm REAL NOT NULL CHECK(warning_distance_mm >= 0),
            enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
            updated_at TEXT NOT NULL
        );
        INSERT INTO tag_monitors VALUES(1, 24.0, 0.72, 80.0, 1, '2026-01-01T00:00:00+00:00');
        PRAGMA user_version = 1;
        """
    )
    connection.commit()
    connection.close()

    with AppDatabase(path) as database:
        database.save_tag_monitor(TagMonitorProfile(1, 24.0, 3.0, 80.0, True))
        profile = database.list_tag_monitors((1,))[0]

    assert profile.quality_threshold == 3.0
