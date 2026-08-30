from __future__ import annotations

import sqlite3
import subprocess
import time
from collections import deque
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QDialog,
    QDialogButtonBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from motion_capture.calibration import apply_reference_pose, average_initial_pose
from motion_capture.config import AppConfig, save_ndi_config, save_realsense_config, save_source_mode
from motion_capture.models import PoseSample, TrackingPacket
from motion_capture.recorder import TrajectoryRecorder
from motion_capture.storage import AppDatabase, TagMonitorProfile
from motion_capture.ui.monitor_widgets import CameraView, MetricCard, TagMonitorCard, app_font
from motion_capture.ui.open3d_view import Open3DViewWidget
from motion_capture.ui.settings_panel import SettingsPanel
from motion_capture.video_recorder import FFmpegVideoRecorder, VideoRecordingResult
from motion_capture.worker import TrackingWorker


class MainWindow(QMainWindow):
    SOURCE_NAMES = {
        "realsense": "Intel RealSense D435i · USB 3.2",
        "ndi": "NDI 光学定位 · IP / ROM",
        "simulator": "模拟数据源 · 本地验证",
    }

    def __init__(self, config: AppConfig, database: AppDatabase | None = None) -> None:
        super().__init__()
        self.config = config
        self.database = database or AppDatabase(config.database_path)
        self.worker: TrackingWorker | None = None
        self.recorder = TrajectoryRecorder(config.record_dir)
        self.video_recorder = FFmpegVideoRecorder(config.record_dir, config.ffmpeg)
        self._session_id: int | None = None
        self._session_device_name = ""
        self._session_failed = False
        self._last_export_path: Path | None = None
        self._last_data_path: Path | None = None
        self._last_video_result: VideoRecordingResult | None = None
        self._last_video_shape: tuple[int, int, int] | None = None
        self._frame_times: deque[float] = deque(maxlen=90)
        self._last_packet_at: float | None = None
        self._last_samples: dict[str, PoseSample] = {}
        self._tag_cards: dict[int, TagMonitorCard] = {}
        self._tag_profiles: dict[int, TagMonitorProfile] = {}
        self._tag_histories: dict[int, deque[tuple[float, np.ndarray]]] = {}
        self._tag_reference_positions: dict[int, np.ndarray] = {}
        self._tag_tones: dict[int, str] = {}
        self._tag_direction_ids: frozenset[int] = frozenset()
        self._rs_calibration_samples: list[PoseSample] | None = None
        self._rs_calibration_frames: set[int] = set()
        self._record_started_at: float | None = None
        self.spatial_dialog: QDialog | None = None
        self.spatial_view: Open3DViewWidget | None = None
        self._closed = False

        self.setWindowTitle("姿态捕捉系统")
        self.resize(1440, 1024)
        self.setMinimumSize(1120, 760)
        self._build_ui()
        self._apply_style()
        # 每次启动全部 Tag 重新开始监控，警告状态由本会话数据重新判读，不沿用上次保存的开关
        self.database.enable_all_tag_monitors()
        self._reload_tag_cards()
        self._sync_source_ui()

        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(1000)
        self._ui_timer.timeout.connect(self._update_clock_and_recording)
        self._ui_timer.start()
        self._update_clock_and_recording()

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("appRoot")
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QFrame()
        header.setObjectName("appHeader")
        header.setFixedHeight(64)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 0, 24, 0)
        header_layout.setSpacing(16)
        title = QLabel("姿态捕捉系统")
        title.setObjectName("appTitle")
        header_layout.addWidget(title)
        separator = QFrame()
        separator.setObjectName("headerSeparator")
        separator.setFixedSize(1, 24)
        header_layout.addWidget(separator)
        self.source_switcher = QComboBox()
        self.source_switcher.setObjectName("sourceSwitcher")
        self.source_switcher.addItem("RealSense 相机", "realsense")
        self.source_switcher.addItem("NDI 光学定位", "ndi")
        self.source_switcher.addItem("模拟器模式", "simulator")
        self.source_switcher.currentIndexChanged.connect(self._on_switcher_changed)
        header_layout.addWidget(self.source_switcher)
        self.device_label = QLabel()
        self.device_label.setObjectName("deviceLabel")
        header_layout.addWidget(self.device_label, 1)
        self.connect_button = QPushButton("连接设备")
        self.connect_button.setObjectName("headerConnectButton")
        self.connect_button.setCursor(Qt.PointingHandCursor)
        self.connect_button.clicked.connect(self.toggle_connection)
        header_layout.addWidget(self.connect_button)
        self.device_status = QLabel("● 设备离线")
        self.device_status.setObjectName("deviceStatus")
        self.device_status.setProperty("state", "offline")
        header_layout.addWidget(self.device_status)
        self.clock_label = QLabel()
        self.clock_label.setObjectName("clockLabel")
        header_layout.addWidget(self.clock_label)
        root.addWidget(header)

        workspace = QFrame()
        workspace.setObjectName("workspace")
        workspace_layout = QHBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(0)

        main_view = QFrame()
        main_view.setObjectName("mainView")
        main_layout = QVBoxLayout(main_view)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        asset_root = Path(__file__).resolve().parent / "assets"
        self.camera_view = CameraView(asset_root)
        self.camera_view.record_requested.connect(self.toggle_recording)
        main_layout.addWidget(self.camera_view, 1)

        metric_bar = QFrame()
        metric_bar.setObjectName("metricBar")
        metric_bar.setFixedHeight(84)
        metric_layout = QHBoxLayout(metric_bar)
        metric_layout.setContentsMargins(16, 8, 16, 8)
        metric_layout.setSpacing(12)
        self.resolution_card = MetricCard("分辨率", "1280 × 720")
        self.fps_card = MetricCard("帧率", "0 FPS")
        self.latency_card = MetricCard("响应时间", "— ms")
        self.roll_pitch_card = MetricCard("IMU · Roll / Pitch", "—° / —°")
        self.yaw_card = MetricCard("IMU · Yaw", "—°")
        for card in (
            self.resolution_card,
            self.fps_card,
            self.latency_card,
            self.roll_pitch_card,
            self.yaw_card,
        ):
            metric_layout.addWidget(card)
        main_layout.addWidget(metric_bar)
        workspace_layout.addWidget(main_view, 1)

        sidebar = QFrame()
        sidebar.setObjectName("rightSidebar")
        sidebar.setFixedWidth(400)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(16, 16, 16, 16)
        sidebar_layout.setSpacing(0)
        self.sidebar_tabs = QTabWidget()
        self.sidebar_tabs.setObjectName("sidebarTabs")
        self.sidebar_tabs.setDocumentMode(True)

        monitor_page = QWidget()
        monitor_layout = QVBoxLayout(monitor_page)
        monitor_layout.setContentsMargins(0, 12, 0, 0)
        monitor_layout.setSpacing(0)
        self.monitor_scroll = QScrollArea()
        self.monitor_scroll.setObjectName("monitorScroll")
        self.monitor_scroll.setWidgetResizable(True)
        self.monitor_scroll.setFrameShape(QFrame.NoFrame)
        self.monitor_content = QWidget()
        self.monitor_cards_layout = QVBoxLayout(self.monitor_content)
        self.monitor_cards_layout.setContentsMargins(0, 0, 0, 0)
        self.monitor_cards_layout.setSpacing(12)
        self.monitor_cards_layout.addStretch()
        self.monitor_scroll.setWidget(self.monitor_content)
        monitor_layout.addWidget(self.monitor_scroll)

        settings_scroll = QScrollArea()
        settings_scroll.setObjectName("settingsScroll")
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.NoFrame)
        settings_host = QWidget()
        settings_host_layout = QVBoxLayout(settings_host)
        settings_host_layout.setContentsMargins(0, 12, 0, 0)
        self.settings_panel = SettingsPanel(self.config)
        self.settings_panel.source_changed.connect(self._on_source_changed)
        self.settings_panel.save_requested.connect(self._save_settings)
        self.settings_panel.calibration_requested.connect(self._toggle_realsense_calibration)
        self.settings_panel.clear_calibration_requested.connect(self._clear_realsense_calibration)
        self.settings_panel.spatial_view_requested.connect(self._open_spatial_view)
        settings_host_layout.addWidget(self.settings_panel)
        settings_scroll.setWidget(settings_host)

        self.sidebar_tabs.addTab(monitor_page, "监控功能")
        self.sidebar_tabs.addTab(settings_scroll, "设置")
        sidebar_layout.addWidget(self.sidebar_tabs)
        workspace_layout.addWidget(sidebar)
        root.addWidget(workspace, 1)
        self.setCentralWidget(central)
        self.statusBar().hide()

        self.record_button = self.camera_view.record_button

    def _update_clock_and_recording(self) -> None:
        self.clock_label.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if self.recorder.active and self._record_started_at is not None:
            elapsed = int(time.monotonic() - self._record_started_at)
            self.camera_view.set_recording(True, elapsed)
        else:
            self.camera_view.set_recording(False, 0)

    def _reload_tag_cards(self) -> None:
        tag_ids = self.config.realsense.tag_ids
        self.database.ensure_tag_monitors(tag_ids, self.config.realsense.tag_size_m * 1000.0)
        profiles = self.database.list_tag_monitors(tag_ids)
        self._tag_profiles = {profile.tag_id: profile for profile in profiles}
        while self.monitor_cards_layout.count() > 1:
            item = self.monitor_cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._tag_cards.clear()
        for profile in profiles:
            card = TagMonitorCard(profile)
            card.monitor_toggled.connect(self._on_monitor_toggled)
            card.settings_requested.connect(self._edit_tag_profile)
            card.threshold_changed.connect(self._on_tag_threshold_changed)
            self.monitor_cards_layout.insertWidget(self.monitor_cards_layout.count() - 1, card)
            self._tag_cards[profile.tag_id] = card
            self._tag_histories.setdefault(profile.tag_id, deque())
        if not profiles:
            empty = QLabel("未配置 AprilTag ID")
            empty.setObjectName("emptyState")
            empty.setAlignment(Qt.AlignCenter)
            self.monitor_cards_layout.insertWidget(0, empty)

    def _sync_source_ui(self) -> None:
        self.source_switcher.blockSignals(True)
        self.source_switcher.setCurrentIndex(
            max(0, self.source_switcher.findData(self.config.source))
        )
        self.source_switcher.blockSignals(False)
        self.device_label.setText(self.SOURCE_NAMES[self.config.source])
        self.settings_panel.load_config(self.config)
        if self.config.source == "realsense":
            self.resolution_card.set_value(
                f"{self.config.realsense.width} × {self.config.realsense.height}"
            )
        elif self.config.source == "simulator":
            self.resolution_card.set_value("1280 × 720")
        else:
            self.resolution_card.set_value("位置数据")

    def _on_switcher_changed(self) -> None:
        self._on_source_changed(str(self.source_switcher.currentData()))

    def _on_source_changed(self, source: str) -> None:
        if source == self.config.source:
            return
        self.config = replace(self.config, source=source)
        try:
            save_source_mode(self.config.root, source)
            self.database.save_config(self.config)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "数据源保存失败", str(exc))
        self._reset_live_state()
        self._sync_source_ui()

    def _save_settings(self, notify: bool = True) -> bool:
        try:
            updated = self.settings_panel.to_config(self.config)
            save_source_mode(updated.root, updated.source)
            save_realsense_config(updated.root, updated.realsense)
            if updated.ndi.rom_files:
                save_ndi_config(updated.root, updated.ndi)
            self.database.save_config(updated)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "设置无效", str(exc))
            return False
        self.config = updated
        self.recorder.directory = updated.record_dir
        self.video_recorder.directory = updated.record_dir
        self.video_recorder.config = updated.ffmpeg
        self.settings_panel.load_config(updated)
        self._reload_tag_cards()
        self._sync_source_ui()
        if notify:
            QMessageBox.information(self, "设置已保存", f"本地 SQLite：{updated.database_path}")
        return True

    def toggle_connection(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self._disconnect()
        else:
            self._connect()

    def _connect(self) -> None:
        if not self._save_settings(notify=False):
            return
        self._reset_live_state()
        self.connect_button.setEnabled(False)
        self.connect_button.setText("连接中…")
        self.source_switcher.setEnabled(False)
        self.camera_view.set_record_enabled(False)
        self._set_device_status("connecting", "● 正在连接")
        self._session_failed = False
        tag_sizes_mm = {
            tag_id: profile.tag_size_mm for tag_id, profile in self._tag_profiles.items()
        }
        self.worker = TrackingWorker(
            self.config.source,
            self.config,
            self,
            tag_sizes_mm=tag_sizes_mm,
        )
        self.worker.connected.connect(self._on_connected)
        self.worker.packet_ready.connect(self._on_packet)
        self.worker.failed.connect(self._on_failed)
        self.worker.stopped.connect(self._on_stopped)
        self.worker.start()

    def _disconnect(self) -> None:
        if self.recorder.active:
            self.toggle_recording()
        if self.worker is not None:
            self.connect_button.setEnabled(False)
            self.connect_button.setText("断开中…")
            self.worker.requestInterruption()

    def _on_connected(self, name: str) -> None:
        self._session_device_name = name
        self._session_id = self.database.start_session(self.config.source, name)
        self._set_device_status("online", "● 设备在线")
        self.device_label.setText(name)
        self.connect_button.setEnabled(True)
        self.connect_button.setText("断开设备")
        self.settings_panel.set_connected(True)
        self.camera_view.set_record_enabled(True)

    def _on_packet(self, packet: TrackingPacket) -> None:
        raw_samples = packet.samples
        if any(sample.source == "realsense" for sample in raw_samples):
            self._collect_realsense_calibration(raw_samples)
            samples = tuple(
                apply_reference_pose(sample, self.config.realsense.reference_transform)
                for sample in raw_samples
            )
            packet = TrackingPacket(samples, packet.image_rgb, packet.tag_detections)

        now = time.monotonic()
        self._frame_times.append(now)
        if len(self._frame_times) > 1:
            fps = (len(self._frame_times) - 1) / max(
                1e-6, self._frame_times[-1] - self._frame_times[0]
            )
            self.fps_card.set_value(f"{fps:.0f} FPS")
        if self._last_packet_at is not None:
            self.latency_card.set_value(f"{(now - self._last_packet_at) * 1000.0:.0f} ms")
        self._last_packet_at = now

        for sample in packet.samples:
            self._last_samples[sample.tool_id] = sample
        if self.spatial_view is not None:
            self.spatial_view.add_samples(packet.samples)
        display_sample = next(
            (sample for sample in packet.samples if sample.tool_id == "apriltag_board"),
            packet.samples[0] if packet.samples else None,
        )
        if display_sample is not None:
            roll, pitch, yaw = display_sample.euler_xyz_degrees
            self.roll_pitch_card.set_value(f"{roll:.1f}° / {pitch:.1f}°")
            self.yaw_card.set_value(f"{yaw:.1f}°")

        self._update_tag_monitors(packet.samples, now)
        self.camera_view.canvas.set_frame(
            packet.image_rgb,
            packet.tag_detections,
            self._tag_tones,
            self._tag_direction_ids,
        )
        if packet.image_rgb is not None:
            self._last_video_shape = (
                int(packet.image_rgb.shape[0]),
                int(packet.image_rgb.shape[1]),
                int(packet.image_rgb.shape[2]),
            )
        if self._session_id is not None:
            self.database.record_samples(self._session_id, packet.samples)
        if self.recorder.active:
            if self.config.source in {"realsense", "simulator"}:
                recorded = tuple(
                    sample for sample in packet.samples if sample.tool_id == "apriltag_board"
                )
            else:
                recorded = packet.samples
            self.recorder.write(recorded)
        if self.video_recorder.active and packet.image_rgb is not None:
            try:
                self.video_recorder.write_frame(packet.image_rgb)
            except (OSError, RuntimeError, ValueError) as exc:
                self._stop_recording(show_dialog=False)
                QMessageBox.critical(self, "视频录制已停止", str(exc))

    def _update_tag_monitors(self, samples: tuple[PoseSample, ...], now: float) -> None:
        by_tag: dict[int, PoseSample] = {}
        for sample in samples:
            if not sample.tool_id.startswith("tag_"):
                continue
            try:
                tag_id = int(sample.tool_id.split("_", 1)[1])
            except ValueError:
                continue
            by_tag[tag_id] = sample

        direction_ids: set[int] = set()
        for tag_id, card in self._tag_cards.items():
            if not card.running:
                card.set_state(False, "danger")
                self._tag_tones[tag_id] = "danger"
                continue
            sample = by_tag.get(tag_id)
            if sample is None:
                card.update_values(None, None, None, "danger")
                self._tag_tones[tag_id] = "danger"
                continue
            position = np.asarray(sample.position_mm, dtype=float)
            reference = self._tag_reference_positions.get(tag_id)
            if reference is None and sample.valid:
                reference = position.copy()
                self._tag_reference_positions[tag_id] = reference
            offset_mm = (
                float(np.linalg.norm(position - reference)) if reference is not None else None
            )
            history = self._tag_histories.setdefault(tag_id, deque())
            history.append((now, position.copy()))
            while history and now - history[0][0] > 5.0:
                history.popleft()
            variance_axis = np.var([item[1] for item in history], axis=0)
            variance_mm = float(np.sqrt(np.mean(variance_axis)))
            # 状态判读用 5s 均值位置对基准的偏移（抗抖动），显示仍用实时偏移 offset_mm
            mean_position = np.mean([item[1] for item in history], axis=0)
            judged_offset = (
                float(np.linalg.norm(mean_position - reference)) if reference is not None else 0.0
            )
            profile = self._tag_profiles[tag_id]
            quality = 1.0 if sample.quality is None else sample.quality
            camera_distance = float(np.linalg.norm(position))
            if camera_distance <= profile.warning_distance_mm:
                distance_tone = "danger"
            elif camera_distance <= profile.warning_distance_mm * 1.5:
                distance_tone = "proximity"
            else:
                distance_tone = "normal"
            if judged_offset >= profile.quality_threshold:
                offset_tone = "danger"
            elif judged_offset >= profile.quality_threshold * 0.5:
                offset_tone = "proximity"
            else:
                offset_tone = "normal"
            if not sample.valid:
                quality_tone = "danger"
            else:
                quality_tone = "normal"
            severity = {"normal": 0, "proximity": 1, "danger": 2}
            tone = max((distance_tone, quality_tone, offset_tone), key=severity.__getitem__)
            self._tag_tones[tag_id] = tone
            # 方向箭头与警告同源：相对初始基准的偏移超过卡片阈值才显示
            if sample.valid and judged_offset >= profile.quality_threshold:
                direction_ids.add(tag_id)
            card.update_values(sample.position_mm, offset_mm, variance_mm, tone)
        self._tag_direction_ids = frozenset(direction_ids)

    def _on_monitor_toggled(self, tag_id: int, running: bool) -> None:
        current = self._tag_profiles[tag_id]
        updated = replace(current, enabled=running)
        try:
            self.database.save_tag_monitor(updated)
        except sqlite3.Error as exc:
            QMessageBox.warning(self, "监控状态保存失败", f"本次运行内生效，但未能写入数据库：{exc}")
        self._tag_profiles[tag_id] = updated
        self._tag_reference_positions.pop(tag_id, None)
        # 基准位置重建时，画面内的初始像素基准也一并重建
        self.camera_view.canvas.motion_tracker.reset_tag(tag_id)
        if running:
            self._tag_histories[tag_id] = deque()
            sample = self._last_samples.get(f"tag_{tag_id}")
            if sample is not None and sample.valid:
                self._tag_reference_positions[tag_id] = np.asarray(
                    sample.position_mm, dtype=float
                ).copy()

    def _on_tag_threshold_changed(self, tag_id: int, value: float) -> None:
        current = self._tag_profiles.get(tag_id)
        if current is None or current.quality_threshold == value:
            return
        updated = replace(current, quality_threshold=value)
        try:
            self.database.save_tag_monitor(updated)
        except sqlite3.Error as exc:
            QMessageBox.warning(self, "阈值保存失败", f"本次运行内生效，但未能写入数据库：{exc}")
        self._tag_profiles[tag_id] = updated

    def _edit_tag_profile(self, tag_id: int) -> None:
        profile = self._tag_profiles[tag_id]
        dialog = QDialog(self)
        dialog.setWindowTitle(f"ID {tag_id:02d} 监控设置")
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(18, 18, 18, 18)
        form = QFormLayout()
        form.setSpacing(10)
        size_spin = QDoubleSpinBox()
        size_spin.setRange(1.0, 1000.0)
        size_spin.setDecimals(1)
        size_spin.setSuffix(" mm")
        size_spin.setValue(profile.tag_size_mm)
        threshold_spin = QDoubleSpinBox()
        threshold_spin.setRange(0.0, float("inf"))
        threshold_spin.setDecimals(2)
        threshold_spin.setSingleStep(0.01)
        threshold_spin.setSuffix(" mm")
        threshold_spin.setValue(profile.quality_threshold)
        form.addRow("Tag 尺寸", size_spin)
        form.addRow("监控阈值", threshold_spin)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("保存")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        updated = replace(
            profile,
            tag_size_mm=size_spin.value(),
            quality_threshold=threshold_spin.value(),
        )
        try:
            self.database.save_tag_monitor(updated)
        except sqlite3.Error as exc:
            QMessageBox.warning(self, "监控设置保存失败", f"本次运行内生效，但未能写入数据库：{exc}")
        self._tag_profiles[tag_id] = updated
        self._tag_cards[tag_id].set_profile(updated)
        if self.worker is not None and self.worker.isRunning():
            self.worker.set_tag_size(tag_id, updated.tag_size_mm)

    def toggle_recording(self) -> None:
        if self.worker is None or not self.worker.isRunning():
            return
        if self.recorder.active:
            self._stop_recording(show_dialog=True)
            return
        if self.config.source == "ndi":
            file_format = self.config.ndi.record_format
            orientation = self.config.ndi.record_orientation
        elif self.config.source == "realsense":
            file_format = self.config.realsense.record_format
            orientation = "position"
        else:
            file_format = "csv"
            orientation = "position"
        video_required = self.config.ffmpeg.enabled and self.config.source in {
            "realsense",
            "simulator",
        }
        if video_required and self._last_video_shape is None:
            QMessageBox.warning(self, "无法开始录制", "尚未收到 RGB 视频帧，请等待画面显示后重试。")
            return
        stem = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        try:
            if video_required:
                height, width, channels = self._last_video_shape
                if channels != 3:
                    raise ValueError(f"RGB 视频帧必须为 3 通道，当前为 {channels} 通道")
                fps = (
                    self.config.realsense.fps
                    if self.config.source == "realsense"
                    else self.config.poll_hz
                )
                self.video_recorder.start(width, height, fps, stem)
            self.recorder.configure(file_format, orientation)
            self.recorder.start(stem)
        except (OSError, RuntimeError, ValueError) as exc:
            if self.video_recorder.active:
                try:
                    self.video_recorder.stop()
                except RuntimeError:
                    pass
            QMessageBox.warning(self, "无法开始录制", str(exc))
            return
        self._last_data_path = None
        self._last_video_result = None
        self._record_started_at = time.monotonic()
        self.camera_view.set_recording(True, 0)

    def _stop_recording(self, *, show_dialog: bool) -> None:
        errors: list[str] = []
        data_path: Path | None = None
        video_result: VideoRecordingResult | None = None
        if self.recorder.active:
            try:
                data_path = self.recorder.stop()
            except (OSError, RuntimeError) as exc:
                errors.append(f"轨迹：{exc}")
        if self.video_recorder.active:
            try:
                video_result = self.video_recorder.stop()
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                errors.append(f"视频：{exc}")
        self._last_data_path = data_path
        self._last_video_result = video_result
        self._last_export_path = video_result.path if video_result is not None else data_path
        self._record_started_at = None
        self.camera_view.set_recording(False, 0)
        if not show_dialog:
            return
        if errors:
            QMessageBox.warning(self, "录制停止时发生错误", "\n".join(errors))
            return
        saved = []
        if video_result is not None:
            saved.append(f"视频：{video_result.path}")
            saved.append(
                f"视频帧：{video_result.frames_written}，丢弃：{video_result.frames_dropped}"
            )
        if data_path is not None:
            saved.append(f"轨迹：{data_path}")
        if saved:
            QMessageBox.information(self, "录制已保存", "\n".join(saved))

    def _toggle_realsense_calibration(self) -> None:
        if self._rs_calibration_samples is not None:
            self._rs_calibration_samples = None
            self._rs_calibration_frames.clear()
            self.settings_panel.finish_calibration("初始标定已取消")
            self.camera_view.set_record_enabled(True)
            return
        if self.worker is None or not self.worker.isRunning() or self.config.source != "realsense":
            QMessageBox.warning(self, "无法标定", "请先连接 RealSense，再开始初始标定。")
            return
        if self.recorder.active:
            QMessageBox.warning(self, "无法标定", "请先停止当前数据录制，再执行初始标定。")
            return
        self._rs_calibration_samples = []
        self._rs_calibration_frames.clear()
        self.camera_view.set_record_enabled(False)
        target = self.config.realsense.calibration_samples
        self.settings_panel.set_calibration_progress(0, target)

    def _collect_realsense_calibration(self, samples: tuple[PoseSample, ...]) -> None:
        if self._rs_calibration_samples is None:
            return
        target = self.config.realsense.calibration_samples
        for sample in samples:
            if (
                sample.tool_id != "apriltag_board"
                or not sample.valid
                or sample.frame_number in self._rs_calibration_frames
            ):
                continue
            if sample.quality is not None and sample.quality < 0.999:
                self.settings_panel.set_calibration_progress(len(self._rs_calibration_samples), target)
                return
            self._rs_calibration_frames.add(sample.frame_number)
            self._rs_calibration_samples.append(sample)
            if len(self._rs_calibration_samples) >= target:
                break
        current = len(self._rs_calibration_samples)
        self.settings_panel.set_calibration_progress(current, target)
        if current < target:
            return

        result = average_initial_pose(self._rs_calibration_samples)
        self._rs_calibration_samples = None
        self._rs_calibration_frames.clear()
        rs = self.config.realsense
        if (
            result.position_rms_mm > rs.calibration_max_std_mm
            or result.max_angle_deviation_deg > rs.calibration_max_angle_deg
        ):
            message = (
                f"标定未保存：位置波动 {result.position_rms_mm:.2f} mm，"
                f"角度波动 {result.max_angle_deviation_deg:.2f}°"
            )
            self.settings_panel.finish_calibration(message)
            self.camera_view.set_record_enabled(True)
            QMessageBox.warning(self, "标定板不稳定", message)
            return

        reference = tuple(float(value) for value in result.reference_transform.reshape(-1))
        calibrated = replace(rs, reference_transform=reference)
        try:
            save_realsense_config(self.config.root, calibrated)
            self.config = replace(self.config, realsense=calibrated)
            self.database.save_config(self.config)
        except (OSError, ValueError) as exc:
            self.settings_panel.finish_calibration("初始零点保存失败")
            self.camera_view.set_record_enabled(True)
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        self.settings_panel.load_config(self.config)
        self.settings_panel.set_connected(True)
        self.settings_panel.finish_calibration(
            f"● 标定完成 · RMS {result.position_rms_mm:.2f} mm / {result.max_angle_deviation_deg:.2f}°"
        )
        self._last_samples.clear()
        self.camera_view.set_record_enabled(True)

    def _clear_realsense_calibration(self) -> None:
        if self.recorder.active:
            QMessageBox.warning(self, "无法清除", "请先停止当前录制，再清除初始零点。")
            return
        identity = tuple(float(value) for value in np.eye(4).reshape(-1))
        realsense = replace(self.config.realsense, reference_transform=identity)
        try:
            save_realsense_config(self.config.root, realsense)
            self.config = replace(self.config, realsense=realsense)
            self.database.save_config(self.config)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "清除失败", str(exc))
            return
        self.settings_panel.load_config(self.config)
        self._last_samples.clear()

    def _on_failed(self, message: str) -> None:
        self._session_failed = True
        self._set_device_status("danger", "● 连接失败")
        QMessageBox.critical(self, "设备错误", message)

    def _on_stopped(self) -> None:
        if self.recorder.active or self.video_recorder.active:
            self._stop_recording(show_dialog=False)
        if self._session_id is not None:
            self.database.end_session(
                self._session_id,
                status="failed" if self._session_failed else "completed",
                export_path=self._last_export_path,
            )
            self._session_id = None
        self.worker = None
        self._record_started_at = None
        self._rs_calibration_samples = None
        self._rs_calibration_frames.clear()
        self._set_device_status("offline", "● 设备离线")
        self.device_label.setText(self.SOURCE_NAMES[self.config.source])
        self.settings_panel.set_connected(False)
        self.source_switcher.setEnabled(True)
        self.settings_panel.finish_calibration()
        self.camera_view.set_record_enabled(False)
        self.camera_view.set_recording(False, 0)
        self.connect_button.setEnabled(True)
        self.connect_button.setText("连接设备")

    def _set_device_status(self, state: str, text: str) -> None:
        self.device_status.setText(text)
        self.device_status.setProperty("state", state)
        self.device_status.style().unpolish(self.device_status)
        self.device_status.style().polish(self.device_status)

    def _open_spatial_view(self) -> None:
        if self.spatial_dialog is None:
            dialog = QDialog(self)
            dialog.setWindowTitle("Open3D 三维空间")
            dialog.resize(900, 620)
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(12, 12, 12, 12)
            self.spatial_view = Open3DViewWidget(self.config.open3d)
            layout.addWidget(self.spatial_view)
            dialog.finished.connect(self._close_spatial_view)
            self.spatial_dialog = dialog
        self.spatial_dialog.show()
        self.spatial_dialog.raise_()
        self.spatial_dialog.activateWindow()

    def _close_spatial_view(self) -> None:
        if self.spatial_view is not None:
            self.spatial_view.shutdown()
        self.spatial_view = None
        self.spatial_dialog = None

    def _reset_live_state(self) -> None:
        self._frame_times.clear()
        self._last_packet_at = None
        self._last_samples.clear()
        self._last_video_shape = None
        self._tag_tones.clear()
        self._tag_direction_ids = frozenset()
        self._tag_reference_positions.clear()
        for history in self._tag_histories.values():
            history.clear()
        self.fps_card.set_value("0 FPS")
        self.latency_card.set_value("— ms")
        self.roll_pitch_card.set_value("—° / —°")
        self.yaw_card.set_value("—°")
        self.camera_view.canvas.set_frame(None, (), {})
        if self.spatial_view is not None:
            self.spatial_view.clear()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._closed:
            event.accept()
            return
        self._closed = True
        if self.recorder.active or self.video_recorder.active:
            self._stop_recording(show_dialog=False)
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(2500)
        if self._session_id is not None:
            self.database.end_session(
                self._session_id,
                status="failed" if self._session_failed else "completed",
                export_path=self._last_export_path,
            )
            self._session_id = None
        if self.spatial_view is not None:
            self.spatial_view.shutdown()
        self.database.close()
        event.accept()

    def _apply_style(self) -> None:
        self.setFont(app_font(10))
        self.setStyleSheet(
            """
            QMainWindow, #appRoot, #workspace, #mainView { background: #F7F9FC; color: #0B121B; }
            QLabel { background: transparent; }
            #appHeader { background: #FFFFFF; border-bottom: 1px solid #D9E0E8; }
            #appTitle { color: #003A99; font-size: 18px; font-weight: 700; }
            #headerSeparator { background: #D9E0E8; }
            #deviceLabel { color: #0B121B; font-size: 13px; font-weight: 500; }
            #sourceSwitcher {
                min-width: 150px; max-height: 36px; padding: 0 10px;
                font-size: 13px; font-weight: 600; color: #003A99;
            }
            #sourceSwitcher:disabled { color: #98A2B3; background: #F2F4F7; }
            #clockLabel { color: #3F4D5A; font-family: "Helvetica Neue"; font-size: 12px; }
            #deviceStatus { color: #5C6A78; font-size: 12px; font-weight: 500; }
            #headerConnectButton {
                background: #0049C0; color: #FFFFFF; border: 0; border-radius: 8px;
                min-height: 34px; padding: 0 18px; font-size: 13px; font-weight: 600;
            }
            #headerConnectButton:hover { background: #003A99; }
            #headerConnectButton:disabled { background: #7FA3D9; color: #FFFFFF; }
            #deviceStatus[state="online"] { color: #12B76A; }
            #deviceStatus[state="connecting"] { color: #FFB020; }
            #deviceStatus[state="danger"] { color: #E5484D; }

            #cameraView { background: #06101B; border: 0; }
            #glassButton, #recordControl, #recordTimer, #layerPanel {
                background: rgba(9, 14, 20, 175); color: #FFFFFF;
                border: 1px solid rgba(255, 255, 255, 62); border-radius: 8px;
            }
            #glassButton { padding: 11px; }
            #layerPanel { border-radius: 8px; }
            QPushButton#layerToggle {
                background: rgba(9, 14, 20, 175); color: #FFFFFF;
                border: 1px solid rgba(255, 255, 255, 62); border-radius: 6px;
                padding: 4px 10px; text-align: left; font-size: 13px;
            }
            QPushButton#layerToggle:checked { color: #FFFFFF; }
            QPushButton#layerToggle:!checked { color: #7F8A96; }
            #recordTimer { font-size: 14px; font-weight: 500; }
            #recordControl { font-size: 15px; font-weight: 600; }
            #recordControl:disabled { color: #7F8A96; background: rgba(9, 14, 20, 120); }
            #recordControl[active="true"] { color: #FFFFFF; }
            #recordTimer { color: #FFFFFF; }

            #metricBar { background: #FFFFFF; border-top: 1px solid #D9E0E8; }
            #metricCard { background: #FFFFFF; border: 1px solid #D9E0E8; border-radius: 12px; }
            #metricLabel { color: #3F4D5A; font-size: 12px; font-weight: 500; }
            #metricValue { color: #0B121B; font-family: "Helvetica Neue"; font-size: 22px; font-weight: 600; }

            #rightSidebar { background: #FFFFFF; border-left: 1px solid #D9E0E8; }
            QTabWidget#sidebarTabs::pane { border: 0; background: #FFFFFF; }
            QTabBar::tab {
                background: #FFFFFF; color: #3F4D5A; border: 0;
                border-bottom: 1px solid #D9E0E8; min-width: 170px; height: 43px;
                font-size: 14px; font-weight: 500;
            }
            QTabBar::tab:selected { color: #003A99; border-bottom: 3px solid #0049C0; }
            #monitorScroll, #settingsScroll, #monitorScroll > QWidget > QWidget,
            #settingsScroll > QWidget > QWidget { background: #FFFFFF; }
            QScrollBar:vertical { background: #FFFFFF; width: 8px; margin: 0; }
            QScrollBar::handle:vertical { background: #D9E0E8; border-radius: 4px; min-height: 28px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

            #tagMonitorCard { background: #FFFFFF; border: 1px solid #D9E0E8; border-radius: 12px; }
            #tagId { color: #0B121B; font-size: 16px; font-weight: 700; }
            #tagConfig { color: #5C6A78; font-size: 12px; font-weight: 500; }
            #tagCoordinates { color: #0B121B; font-size: 14px; font-weight: 500; }
            #tagOffset { color: #0B121B; font-size: 12px; font-weight: 500; }
            #tagStatus { background: #EEF2F7; color: #12B76A; border-radius: 10px; padding: 4px 8px; font-size: 12px; font-weight: 500; }
            #tagThresholdSpin { min-height: 24px; max-height: 26px; min-width: 96px; padding: 0 4px; font-size: 12px; }
            #tagThresholdSpin::up-button, #tagThresholdSpin::down-button { width: 18px; }
            #tagAction { background: #E5484D; color: #FFFFFF; border: 0; border-radius: 8px; font-size: 14px; }
            #tagMonitorCard[tone="proximity"] #tagOffset,
            #tagMonitorCard[tone="proximity"] #tagStatus { color: #FFB020; }
            #tagMonitorCard[tone="danger"] #tagOffset,
            #tagMonitorCard[tone="danger"] #tagStatus { color: #E5484D; }
            #tagMonitorCard[running="false"] #tagAction { background: #0049C0; }

            #settingsPanel { background: #FFFFFF; }
            #settingsSectionTitle { color: #0B121B; font-size: 14px; font-weight: 700; }
            #settingsHint, #databasePath, #calibrationStatus { color: #5C6A78; font-size: 12px; }
            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
                background: #FFFFFF; color: #0B121B; border: 1px solid #D9E0E8;
                border-radius: 6px; min-height: 34px; padding: 0 8px;
            }
            QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #0049C0; }
            QPushButton { background: #EEF2F7; color: #3F4D5A; border: 0; border-radius: 8px; padding: 8px 12px; }
            QPushButton:hover { background: #E4EAF2; }
            QPushButton:disabled { color: #98A2B3; background: #F2F4F7; }
            #primaryAction { background: #0049C0; color: #FFFFFF; font-weight: 500; }
            #primaryAction:hover { background: #003A99; }
            #secondaryAction { background: #EEF2F7; color: #3F4D5A; }
            #emptyState { color: #5C6A78; padding: 32px; }
            """
        )
