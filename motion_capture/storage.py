from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from motion_capture.config import (
    AppConfig,
    FFmpegConfig,
    NDIConfig,
    Open3DConfig,
    build_realsense_config,
)
from motion_capture.models import PoseSample


@dataclass(frozen=True)
class TagMonitorProfile:
    tag_id: int
    tag_size_mm: float
    quality_threshold: float
    warning_distance_mm: float
    enabled: bool


class AppDatabase:
    """Local SQLite persistence for settings, monitor profiles and capture history."""

    SCHEMA_VERSION = 2

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self._last_commit = time.monotonic()
        self._create_schema()

    def _migrate_tag_monitors_unbounded_threshold(self) -> None:
        version = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        if version >= self.SCHEMA_VERSION:
            return
        table = self.connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='tag_monitors'"
        ).fetchone()
        if table is None or "quality_threshold <= 1" not in (table[0] or ""):
            return
        self.connection.executescript(
            """
            CREATE TABLE tag_monitors_migrated (
                tag_id INTEGER PRIMARY KEY,
                tag_size_mm REAL NOT NULL CHECK(tag_size_mm > 0),
                quality_threshold REAL NOT NULL CHECK(quality_threshold >= 0),
                warning_distance_mm REAL NOT NULL CHECK(warning_distance_mm >= 0),
                enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
                updated_at TEXT NOT NULL
            );
            INSERT INTO tag_monitors_migrated
                SELECT tag_id, tag_size_mm, quality_threshold, warning_distance_mm, enabled, updated_at
                FROM tag_monitors;
            DROP TABLE tag_monitors;
            ALTER TABLE tag_monitors_migrated RENAME TO tag_monitors;
            """
        )

    def _create_schema(self) -> None:
        self._migrate_tag_monitors_unbounded_threshold()
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tag_monitors (
                tag_id INTEGER PRIMARY KEY,
                tag_size_mm REAL NOT NULL CHECK(tag_size_mm > 0),
                quality_threshold REAL NOT NULL CHECK(quality_threshold >= 0),
                warning_distance_mm REAL NOT NULL CHECK(warning_distance_mm >= 0),
                enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS capture_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                device_name TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                export_path TEXT,
                status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed', 'cancelled'))
            );

            CREATE TABLE IF NOT EXISTS pose_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES capture_sessions(id) ON DELETE CASCADE,
                timestamp_utc TEXT NOT NULL,
                source TEXT NOT NULL,
                tool_id TEXT NOT NULL,
                frame_number INTEGER NOT NULL,
                valid INTEGER NOT NULL CHECK(valid IN (0, 1)),
                quality REAL,
                x_mm REAL NOT NULL,
                y_mm REAL NOT NULL,
                z_mm REAL NOT NULL,
                rx_deg REAL NOT NULL,
                ry_deg REAL NOT NULL,
                rz_deg REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_pose_samples_session_time
            ON pose_samples(session_id, timestamp_utc);
            """
        )
        self.connection.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")
        self.connection.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def save_config(self, config: AppConfig) -> None:
        payload = {
            "source": config.source,
            "poll_hz": config.poll_hz,
            "record_dir": str(config.record_dir),
            "realsense": asdict(config.realsense),
            "ndi": {
                **asdict(config.ndi),
                "rom_files": [str(path) for path in config.ndi.rom_files],
            },
            "open3d": asdict(config.open3d),
            "ffmpeg": asdict(config.ffmpeg),
        }
        self.connection.execute(
            """
            INSERT INTO app_settings(key, value_json, updated_at)
            VALUES('runtime_config', ?, ?)
            ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
            """,
            (json.dumps(payload, ensure_ascii=False, separators=(",", ":")), self._now()),
        )
        self.connection.commit()

    def load_config(self, fallback: AppConfig) -> AppConfig:
        row = self.connection.execute(
            "SELECT value_json FROM app_settings WHERE key='runtime_config'"
        ).fetchone()
        if row is None:
            self.save_config(fallback)
            return fallback
        try:
            payload = json.loads(row["value_json"])
            rs_data = payload["realsense"]
            realsense = build_realsense_config(
                serial=rs_data["serial"],
                width=rs_data["width"],
                height=rs_data["height"],
                fps=rs_data["fps"],
                tag_family=rs_data["tag_family"],
                tag_ids=rs_data["tag_ids"],
                board_rows=rs_data["board_rows"],
                board_cols=rs_data["board_cols"],
                tag_size_m=rs_data["tag_size_m"],
                tag_spacing_m=rs_data["tag_spacing_m"],
                min_visible_tags=rs_data["min_visible_tags"],
                record_format=rs_data["record_format"],
                reference_transform=rs_data["reference_transform"],
                calibration_samples=rs_data["calibration_samples"],
                calibration_max_std_mm=rs_data["calibration_max_std_mm"],
                calibration_max_angle_deg=rs_data["calibration_max_angle_deg"],
            )
            ndi_data = payload["ndi"]
            ndi = NDIConfig(
                tracker_type=str(ndi_data["tracker_type"]),
                ip_address=str(ndi_data["ip_address"]),
                port=int(ndi_data["port"]),
                serial_port=str(ndi_data["serial_port"]),
                rom_files=tuple(Path(path).resolve() for path in ndi_data["rom_files"]),
                record_orientation=str(ndi_data["record_orientation"]),
                record_format=str(ndi_data["record_format"]),
            )
            open3d = Open3DConfig(**payload["open3d"])
            ffmpeg = FFmpegConfig(**payload.get("ffmpeg", asdict(fallback.ffmpeg)))
            source = str(payload["source"])
            if source not in {"simulator", "realsense", "ndi"}:
                raise ValueError("SQLite 中的 source 无效")
            return replace(
                fallback,
                source=source,
                poll_hz=int(payload["poll_hz"]),
                record_dir=Path(payload["record_dir"]).expanduser().resolve(),
                realsense=realsense,
                ndi=ndi,
                open3d=open3d,
                ffmpeg=ffmpeg,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"SQLite 运行配置损坏：{exc}") from exc

    def ensure_tag_monitors(self, tag_ids: Iterable[int], tag_size_mm: float) -> None:
        now = self._now()
        self.connection.executemany(
            """
            INSERT OR IGNORE INTO tag_monitors(
                tag_id, tag_size_mm, quality_threshold, warning_distance_mm, enabled, updated_at
            ) VALUES(?, ?, 5.0, 80.0, 1, ?)
            """,
            ((int(tag_id), float(tag_size_mm), now) for tag_id in tag_ids),
        )
        self.connection.commit()

    def list_tag_monitors(self, tag_ids: Iterable[int] | None = None) -> tuple[TagMonitorProfile, ...]:
        if tag_ids is None:
            rows = self.connection.execute("SELECT * FROM tag_monitors ORDER BY tag_id").fetchall()
        else:
            normalized = tuple(int(tag_id) for tag_id in tag_ids)
            if not normalized:
                return ()
            placeholders = ",".join("?" for _ in normalized)
            rows = self.connection.execute(
                f"SELECT * FROM tag_monitors WHERE tag_id IN ({placeholders}) ORDER BY tag_id",
                normalized,
            ).fetchall()
        return tuple(
            TagMonitorProfile(
                tag_id=int(row["tag_id"]),
                tag_size_mm=float(row["tag_size_mm"]),
                quality_threshold=float(row["quality_threshold"]),
                warning_distance_mm=float(row["warning_distance_mm"]),
                enabled=bool(row["enabled"]),
            )
            for row in rows
        )

    def enable_all_tag_monitors(self) -> None:
        self.connection.execute(
            "UPDATE tag_monitors SET enabled = 1, updated_at = ? WHERE enabled = 0",
            (self._now(),),
        )
        self.connection.commit()

    def save_tag_monitor(self, profile: TagMonitorProfile) -> None:
        self.connection.execute(
            """
            INSERT INTO tag_monitors(
                tag_id, tag_size_mm, quality_threshold, warning_distance_mm, enabled, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(tag_id) DO UPDATE SET
                tag_size_mm=excluded.tag_size_mm,
                quality_threshold=excluded.quality_threshold,
                warning_distance_mm=excluded.warning_distance_mm,
                enabled=excluded.enabled,
                updated_at=excluded.updated_at
            """,
            (
                profile.tag_id,
                profile.tag_size_mm,
                profile.quality_threshold,
                profile.warning_distance_mm,
                int(profile.enabled),
                self._now(),
            ),
        )
        self.connection.commit()

    def start_session(self, source: str, device_name: str) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO capture_sessions(source, device_name, started_at, status)
            VALUES(?, ?, ?, 'running')
            """,
            (source, device_name, self._now()),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def record_samples(self, session_id: int, samples: Iterable[PoseSample]) -> None:
        rows = []
        for sample in samples:
            rx, ry, rz = sample.euler_xyz_degrees
            x, y, z = sample.position_mm
            rows.append(
                (
                    session_id,
                    sample.timestamp_utc.isoformat(),
                    sample.source,
                    sample.tool_id,
                    sample.frame_number,
                    int(sample.valid),
                    sample.quality,
                    x,
                    y,
                    z,
                    rx,
                    ry,
                    rz,
                )
            )
        if not rows:
            return
        self.connection.executemany(
            """
            INSERT INTO pose_samples(
                session_id, timestamp_utc, source, tool_id, frame_number, valid, quality,
                x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        if time.monotonic() - self._last_commit >= 0.5:
            self.connection.commit()
            self._last_commit = time.monotonic()

    def end_session(
        self,
        session_id: int,
        *,
        status: str = "completed",
        export_path: Path | None = None,
    ) -> None:
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError("会话结束状态无效")
        self.connection.execute(
            """
            UPDATE capture_sessions
            SET ended_at=?, export_path=?, status=?
            WHERE id=?
            """,
            (self._now(), None if export_path is None else str(export_path), status, session_id),
        )
        self.connection.commit()

    def sample_count(self, session_id: int) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM pose_samples WHERE session_id=?", (session_id,)
        ).fetchone()
        return int(row["count"])

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def __enter__(self) -> AppDatabase:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()
