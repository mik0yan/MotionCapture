from __future__ import annotations

import time
from collections import deque
from dataclasses import replace

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from motion_capture.calibration import apply_reference_pose, average_initial_pose, is_identity_reference
from motion_capture.config import AppConfig, save_ndi_config, save_realsense_config, save_source_mode
from motion_capture.models import PoseSample, TrackingPacket
from motion_capture.recorder import TrajectoryRecorder
from motion_capture.ui.ndi_config_form import NDIConfigForm
from motion_capture.ui.mode_switcher import WorkModeSwitcher
from motion_capture.ui.open3d_view import Open3DViewWidget
from motion_capture.ui.realsense_config_form import RealSenseConfigForm
from motion_capture.worker import TrackingWorker


class MetricCard(QFrame):
    def __init__(self, title: str, value: str) -> None:
        super().__init__()
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricValue")
        layout.addWidget(title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class MainWindow(QMainWindow):
    SOURCE_LABELS = {
        "模拟器（无需硬件）": "simulator",
        "Intel RealSense / AprilTag": "realsense",
        "NDI / ROM 反光球工具": "ndi",
    }
    SOURCE_DISPLAY_NAMES = {
        "ndi": "NDI 模式",
        "realsense": "RealSense",
        "simulator": "模拟器模式",
    }

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.worker: TrackingWorker | None = None
        self.recorder = TrajectoryRecorder(config.record_dir)
        self._frame_times: deque[float] = deque(maxlen=60)
        self._last_samples: dict[str, PoseSample] = {}
        self._rs_calibration_samples: list[PoseSample] | None = None
        self._rs_calibration_frames: set[int] = set()
        self.setWindowTitle("MotionCapture · RealSense / NDI")
        self.resize(1360, 920)
        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(16)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        title = QLabel("MotionCapture")
        title.setObjectName("pageTitle")
        subtitle = QLabel("RealSense AprilTag 标定板 · NDI 红外反光球 ROM 工具")
        subtitle.setObjectName("subtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles)
        header.addStretch()
        self.status_badge = QLabel("● 未连接")
        self.status_badge.setObjectName("statusBadge")
        header.addWidget(self.status_badge)
        root.addLayout(header)

        controls = QFrame()
        controls.setObjectName("controlBar")
        control_layout = QHBoxLayout(controls)
        control_layout.setContentsMargins(16, 12, 16, 12)
        control_layout.addWidget(QLabel("工作模式"))
        self.mode_switcher = WorkModeSwitcher(self.config.source)
        control_layout.addWidget(self.mode_switcher, 1)
        self.connect_button = QPushButton("连接设备")
        self.connect_button.setObjectName("primaryButton")
        self.connect_button.clicked.connect(self.toggle_connection)
        control_layout.addWidget(self.connect_button)
        self.record_button = QPushButton(self._idle_record_button_text())
        self.record_button.setEnabled(False)
        self.record_button.clicked.connect(self.toggle_recording)
        control_layout.addWidget(self.record_button)
        root.addWidget(controls)

        self.realsense_form = RealSenseConfigForm(self.config.realsense)
        self.realsense_form.save_requested.connect(self._save_realsense_form)
        self.realsense_form.calibration_requested.connect(self._toggle_realsense_calibration)
        self.realsense_form.clear_calibration_requested.connect(self._clear_realsense_calibration)

        self.ndi_form = NDIConfigForm(self.config.root, self.config.ndi)
        self.ndi_form.save_requested.connect(self._save_ndi_form)
        self.device_config_stack = QStackedWidget()
        self.device_config_stack.setObjectName("deviceConfigStack")
        self.device_config_stack.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.device_config_stack.addWidget(self.realsense_form)
        self.device_config_stack.addWidget(self.ndi_form)
        root.addWidget(self.device_config_stack)
        self.mode_switcher.mode_changed.connect(self._on_mode_changed)
        self._sync_source_panels()

        metrics = QHBoxLayout()
        self.source_card = MetricCard(
            "当前数据源", f"{self.SOURCE_DISPLAY_NAMES[self.config.source]} · 待连接"
        )
        self.tools_card = MetricCard("可见工具", "0")
        self.fps_card = MetricCard("更新频率", "0.0 Hz")
        self.frame_card = MetricCard("最新帧号", "—")
        for card in (self.source_card, self.tools_card, self.fps_card, self.frame_card):
            metrics.addWidget(card)
        root.addLayout(metrics)

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.video_label = QLabel("连接 RealSense 后显示视频\nNDI 模式显示位姿与轨迹")
        self.video_label.setObjectName("videoPanel")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(660, 250)
        left_layout.addWidget(self.video_label, 3)
        self.spatial_view = Open3DViewWidget(self.config.open3d)
        left_layout.addWidget(self.spatial_view, 2)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 0, 0, 0)
        section = QLabel("实时位姿")
        section.setObjectName("sectionTitle")
        right_layout.addWidget(section)
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(("工具", "状态", "X mm", "Y mm", "Z mm", "Rx °", "Ry °", "Rz °", "质量"))
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        right_layout.addWidget(self.table, 1)

        config_title = QLabel("当前配置")
        config_title.setObjectName("sectionTitle")
        right_layout.addWidget(config_title)
        self.config_label = QLabel(self._config_summary())
        self.config_label.setObjectName("configPanel")
        self.config_label.setWordWrap(True)
        self.config_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        right_layout.addWidget(self.config_label)
        splitter.addWidget(right)
        splitter.setSizes([850, 470])
        root.addWidget(splitter, 1)
        self.setCentralWidget(central)

    def _config_summary(self) -> str:
        rs = self.config.realsense
        ndi = self.config.ndi
        roms = ", ".join(path.name for path in ndi.rom_files) or "未配置"
        ndi_endpoint = (
            f"{ndi.ip_address}:{ndi.port}"
            if ndi.tracker_type == "vega"
            else (ndi.serial_port or "自动检测串口")
        )
        return (
            f"配置文件：{self.config.root / '.env'}\n"
            f"工作模式：{self.SOURCE_DISPLAY_NAMES[self._selected_source()]}\n"
            f"AprilTag：{rs.tag_family} · {rs.board_rows}×{rs.board_cols} · {rs.tag_size_m * 1000:.0f} mm\n"
            f"RealSense 位置：{rs.record_format.upper()} · 最少 {rs.min_visible_tags} Tag\n"
            f"RealSense 零点：{'已标定' if not is_identity_reference(rs.reference_transform) else '未标定'}\n"
            f"NDI：{ndi.tracker_type.upper()} · {ndi_endpoint}\n"
            f"ROM：{roms}\n"
            f"NDI 轨迹：{ndi.record_format.upper()} · "
            f"{'四元数' if ndi.record_orientation == 'quaternion' else '方向余弦矩阵'}\n"
            f"Open3D：{'启用' if self.config.open3d.enabled else '禁用'} · {self.config.open3d.render_hz} Hz\n"
            f"录制目录：{self.config.record_dir}"
        )

    def _selected_source(self) -> str:
        return self.mode_switcher.current_mode

    def _on_mode_changed(self, mode: str) -> None:
        self.config = replace(self.config, source=mode)
        try:
            env_path = save_source_mode(self.config.root, mode)
        except OSError as exc:
            QMessageBox.warning(self, "工作模式保存失败", str(exc))
        else:
            self.statusBar().showMessage(f"工作模式已保存：{env_path}", 4000)
        self._last_samples.clear()
        self._frame_times.clear()
        self.table.setRowCount(0)
        self.spatial_view.clear()
        self.video_label.clear()
        self.video_label.setText("连接 RealSense 后显示视频\nNDI 模式显示位姿与轨迹")
        self.source_card.set_value(f"{self.SOURCE_DISPLAY_NAMES[mode]} · 待连接")
        self.tools_card.set_value("0")
        self.fps_card.set_value("0.0 Hz")
        self.frame_card.set_value("—")
        self.record_button.setText(self._idle_record_button_text())
        self.config_label.setText(self._config_summary())
        self._sync_source_panels()

    def _sync_source_panels(self) -> None:
        source = self._selected_source()
        if source == "realsense":
            self.device_config_stack.setCurrentWidget(self.realsense_form)
            self.device_config_stack.setVisible(True)
        elif source == "ndi":
            self.device_config_stack.setCurrentWidget(self.ndi_form)
            self.device_config_stack.setVisible(True)
        else:
            self.device_config_stack.setVisible(False)

    def _idle_record_button_text(self) -> str:
        source = self._selected_source()
        if source == "ndi":
            return "开始轨迹记录"
        if source == "realsense":
            return "开始位置记录"
        return "开始录制"

    def _save_realsense_form(self, notify: bool = True) -> bool:
        try:
            realsense = self.realsense_form.to_config()
            env_path = save_realsense_config(self.config.root, realsense)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "RealSense 配置无效", str(exc))
            return False
        self.config = replace(self.config, realsense=realsense)
        self.realsense_form.load_config(realsense)
        self.config_label.setText(self._config_summary())
        if notify:
            self.statusBar().showMessage(f"RealSense 配置已保存：{env_path}", 6000)
        return True

    def _save_ndi_form(self, notify: bool = True) -> bool:
        try:
            ndi = self.ndi_form.to_config()
            env_path = save_ndi_config(self.config.root, ndi)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "NDI 配置无效", str(exc))
            return False
        self.config = replace(self.config, ndi=ndi)
        self.config_label.setText(self._config_summary())
        if notify:
            self.statusBar().showMessage(f"NDI 配置已保存：{env_path}", 6000)
        return True

    def toggle_connection(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self._disconnect()
        else:
            self._connect()

    def _connect(self) -> None:
        source = self._selected_source()
        if source == "realsense" and not self._save_realsense_form(notify=False):
            return
        if source == "ndi" and not self._save_ndi_form(notify=False):
            return
        self._frame_times.clear()
        self._last_samples.clear()
        self.spatial_view.clear()
        self.mode_switcher.setEnabled(False)
        self.realsense_form.set_connection_active(True, calibration_available=False)
        self.ndi_form.set_connection_active(True)
        self.connect_button.setEnabled(False)
        self.connect_button.setText("连接中…")
        self.status_badge.setText("● 正在连接")
        self.worker = TrackingWorker(self._selected_source(), self.config, self)
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
        self.status_badge.setText("● 已连接")
        self.status_badge.setProperty("connected", True)
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)
        self.source_card.set_value(name)
        self.connect_button.setText("断开设备")
        self.connect_button.setEnabled(True)
        self.record_button.setEnabled(True)
        if self._selected_source() == "realsense":
            self.realsense_form.set_connection_active(True, calibration_available=True)

    def _on_packet(self, packet: TrackingPacket) -> None:
        if packet.samples and packet.samples[0].source == "realsense":
            self._collect_realsense_calibration(packet.samples)
            packet = TrackingPacket(
                tuple(
                    apply_reference_pose(sample, self.config.realsense.reference_transform)
                    for sample in packet.samples
                ),
                packet.image_rgb,
            )
        now = time.monotonic()
        self._frame_times.append(now)
        if len(self._frame_times) > 1:
            fps = (len(self._frame_times) - 1) / max(1e-6, self._frame_times[-1] - self._frame_times[0])
            self.fps_card.set_value(f"{fps:.1f} Hz")
        self.tools_card.set_value(str(len(packet.samples)))
        if packet.samples:
            self.frame_card.set_value(str(max(sample.frame_number for sample in packet.samples)))
            for sample in packet.samples:
                self._last_samples[sample.tool_id] = sample
            self.spatial_view.add_samples(packet.samples)
        self._update_table()
        if packet.image_rgb is not None:
            self._show_image(packet.image_rgb)
        if self.recorder.active:
            self.recorder.write(packet.samples)

    def _show_image(self, image) -> None:
        height, width, channels = image.shape
        qimage = QImage(image.data, width, height, channels * width, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

    def _update_table(self) -> None:
        samples = list(self._last_samples.values())
        self.table.setRowCount(len(samples))
        for row, sample in enumerate(samples):
            euler = sample.euler_xyz_degrees
            quality = "—" if sample.quality is None else f"{sample.quality:.3f}"
            values = (
                sample.tool_id, "有效" if sample.valid else "丢失",
                *[f"{value:.2f}" for value in sample.position_mm],
                *[f"{value:.2f}" for value in euler], quality,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column >= 2:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(row, column, item)

    def toggle_recording(self) -> None:
        if self.recorder.active:
            path = self.recorder.stop()
            self.record_button.setText(self._idle_record_button_text())
            self.statusBar().showMessage(f"录制已保存：{path}", 8000)
        else:
            if self._selected_source() == "ndi":
                file_format = self.config.ndi.record_format
                orientation = self.config.ndi.record_orientation
            elif self._selected_source() == "realsense":
                file_format = self.config.realsense.record_format
                orientation = "position"
            else:
                file_format = "csv"
                orientation = "quaternion"
            try:
                self.recorder.configure(file_format, orientation)
                path = self.recorder.start()
            except (OSError, RuntimeError, ValueError) as exc:
                QMessageBox.warning(self, "无法开始录制", str(exc))
                return
            self.record_button.setText("停止录制")
            self.statusBar().showMessage(f"正在录制：{path}")

    def _toggle_realsense_calibration(self) -> None:
        if self._rs_calibration_samples is not None:
            self._rs_calibration_samples = None
            self._rs_calibration_frames.clear()
            self.realsense_form.finish_calibration("初始标定已取消")
            self.record_button.setEnabled(True)
            return
        if self.worker is None or not self.worker.isRunning() or self._selected_source() != "realsense":
            QMessageBox.warning(self, "无法标定", "请先连接 RealSense，再开始初始标定。")
            return
        if self.recorder.active:
            QMessageBox.warning(self, "无法标定", "请先停止当前数据录制，再执行初始标定。")
            return
        self._rs_calibration_samples = []
        self._rs_calibration_frames.clear()
        self.record_button.setEnabled(False)
        target = self.config.realsense.calibration_samples
        self.realsense_form.set_calibration_progress(0, target)

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
                self.realsense_form.set_calibration_progress(len(self._rs_calibration_samples), target)
                return
            self._rs_calibration_frames.add(sample.frame_number)
            self._rs_calibration_samples.append(sample)
            if len(self._rs_calibration_samples) >= target:
                break
        current = len(self._rs_calibration_samples)
        self.realsense_form.set_calibration_progress(current, target)
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
            self.realsense_form.finish_calibration(message)
            self.record_button.setEnabled(True)
            QMessageBox.warning(
                self,
                "标定板不稳定",
                message + "\n请固定相机和标定板，并确保所有 Tag 完整可见后重试。",
            )
            return

        reference = tuple(float(value) for value in result.reference_transform.reshape(-1))
        calibrated = replace(rs, reference_transform=reference)
        try:
            env_path = save_realsense_config(self.config.root, calibrated)
        except OSError as exc:
            self.realsense_form.finish_calibration("初始零点保存失败")
            self.record_button.setEnabled(True)
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        self.config = replace(self.config, realsense=calibrated)
        self.realsense_form.load_config(calibrated)
        self.realsense_form.set_connection_active(True, calibration_available=True)
        self.realsense_form.finish_calibration(
            f"● 标定完成 · RMS {result.position_rms_mm:.2f} mm / {result.max_angle_deviation_deg:.2f}°"
        )
        self.config_label.setText(self._config_summary())
        self._last_samples.clear()
        self.spatial_view.clear()
        self.statusBar().showMessage(f"初始零点已保存：{env_path}", 8000)
        self.record_button.setEnabled(True)

    def _clear_realsense_calibration(self) -> None:
        if self.recorder.active:
            QMessageBox.warning(self, "无法清除", "请先停止当前 CSV 录制，再清除初始零点。")
            return
        self._rs_calibration_samples = None
        self._rs_calibration_frames.clear()
        identity = tuple(float(value) for value in np.eye(4).reshape(-1))
        realsense = replace(self.config.realsense, reference_transform=identity)
        try:
            save_realsense_config(self.config.root, realsense)
        except OSError as exc:
            QMessageBox.warning(self, "清除失败", str(exc))
            return
        self.config = replace(self.config, realsense=realsense)
        self.realsense_form.load_config(realsense)
        self.realsense_form.set_connection_active(
            self.worker is not None and self.worker.isRunning(),
            calibration_available=self.worker is not None and self.worker.isRunning(),
        )
        self.config_label.setText(self._config_summary())
        self._last_samples.clear()
        self.spatial_view.clear()
        self.statusBar().showMessage("RealSense 初始零点已清除", 6000)

    def _on_failed(self, message: str) -> None:
        self.status_badge.setText("● 连接失败")
        QMessageBox.critical(self, "设备错误", message)

    def _on_stopped(self) -> None:
        self.worker = None
        self._rs_calibration_samples = None
        self._rs_calibration_frames.clear()
        self.status_badge.setProperty("connected", False)
        self.status_badge.setText("● 未连接")
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)
        self.mode_switcher.setEnabled(True)
        self.realsense_form.set_connection_active(False)
        self.realsense_form.finish_calibration()
        self.ndi_form.set_connection_active(False)
        self._sync_source_panels()
        self.connect_button.setText("连接设备")
        self.connect_button.setEnabled(True)
        self.record_button.setEnabled(False)
        self.record_button.setText(self._idle_record_button_text())
        self.source_card.set_value(
            f"{self.SOURCE_DISPLAY_NAMES[self._selected_source()]} · 待连接"
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self.recorder.active:
            self.recorder.stop()
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(2500)
        self.spatial_view.shutdown()
        event.accept()

    def _apply_style(self) -> None:
        self.setFont(QFont("Arial", 10))
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #0b121b; color: #dce7f2; }
            QLabel { background: transparent; }
            #pageTitle { font-size: 27px; font-weight: 700; color: #f4f8fb; }
            #subtitle, #metricTitle { color: #8293a6; }
            #statusBadge { background: #273443; color: #b6c1cc; padding: 8px 14px; border-radius: 14px; }
            #statusBadge[connected="true"] { background: #113e35; color: #59e0b9; }
            #controlBar, #metricCard, #configPanel, #ndiConfigPanel, #realsenseConfigPanel { background: #121d29; border: 1px solid #213043; border-radius: 9px; }
            #modeSwitcher { background: #0d1721; border: 1px solid #2b3e52; border-radius: 7px; }
            QPushButton#modeButton { background: transparent; color: #91a4b6; padding: 8px 14px; border-radius: 5px; font-weight: 600; }
            QPushButton#modeButton:hover { background: #1b2a39; color: #dce7f2; }
            QPushButton#modeButton:checked { background: #16a884; color: #0b121b; }
            QPushButton#modeButton:checked:disabled { background: #127e65; color: #0b121b; }
            #metricValue { font-size: 21px; font-weight: 650; color: #f0f6fa; }
            #sectionTitle { font-size: 15px; font-weight: 650; margin: 4px 0; }
            #configPanel { color: #9eafbf; padding: 13px; line-height: 1.5; }
            #ndiConfigTitle { color: #f0f6fa; font-size: 14px; font-weight: 650; }
            #ndiConfigHint { color: #8293a6; }
            #realsenseConfigTitle { color: #f0f6fa; font-size: 14px; font-weight: 650; }
            #realsenseExportNote { color: #8293a6; }
            #calibrationStatus { color: #8293a6; }
            #calibrationStatus[calibrated="true"] { color: #59e0b9; }
            #videoPanel, #open3dPanel { background: #101923; border: 1px solid #213043; border-radius: 9px; color: #718397; }
            #open3dTitle { color: #dce7f2; font-size: 14px; font-weight: 650; }
            #open3dSummary { color: #8293a6; }
            #open3dCanvas { background: #0a1017; border-radius: 6px; color: #718397; }
            QPushButton { background: #26374a; border: 0; padding: 9px 16px; border-radius: 6px; }
            QPushButton:hover { background: #31475e; }
            QPushButton:disabled { color: #627183; background: #1b2734; }
            #primaryButton { background: #16a884; color: #0b121b; font-weight: 650; }
            #primaryButton:hover { background: #1bbb94; }
            #calibrationButton { border: 1px solid #16a884; color: #59e0b9; background: #113e35; }
            #calibrationButton:disabled { border-color: #26374a; color: #627183; background: #1b2734; }
            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox { background: #0d1721; border: 1px solid #2b3e52; padding: 8px 12px; border-radius: 6px; }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #16a884; }
            QTableWidget { background: #101923; alternate-background-color: #142130; border: 1px solid #213043; gridline-color: #213043; border-radius: 7px; }
            QHeaderView::section { background: #182535; color: #9fb0c0; border: 0; border-bottom: 1px solid #2a3a4b; padding: 8px; }
            QSplitter::handle { background: #162331; width: 2px; }
            QStatusBar { color: #91a4b6; }
        """)
