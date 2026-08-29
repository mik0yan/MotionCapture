from __future__ import annotations

from dataclasses import replace

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from motion_capture.calibration import is_identity_reference
from motion_capture.config import AppConfig, build_ndi_config, build_realsense_config


class SettingsPanel(QFrame):
    source_changed = pyqtSignal(str)
    save_requested = pyqtSignal()
    connect_requested = pyqtSignal()
    calibration_requested = pyqtSignal()
    clear_calibration_requested = pyqtSignal()
    spatial_view_requested = pyqtSignal()

    def __init__(self, config: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self._connection_active = False
        self.setObjectName("settingsPanel")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)

        source_label = QLabel("数据源")
        source_label.setObjectName("settingsSectionTitle")
        outer.addWidget(source_label)
        self.source_combo = QComboBox()
        self.source_combo.addItem("Intel RealSense（USB 设备）", "realsense")
        self.source_combo.addItem("NDI（IP / 串口）", "ndi")
        self.source_combo.addItem("模拟器（无需硬件）", "simulator")
        self.source_combo.currentIndexChanged.connect(self._source_index_changed)
        outer.addWidget(self.source_combo)

        self.source_stack = QStackedWidget()
        self.realsense_page = self._build_realsense_page()
        self.ndi_page = self._build_ndi_page()
        simulator_page = QFrame()
        simulator_layout = QVBoxLayout(simulator_page)
        simulator_layout.setContentsMargins(0, 12, 0, 12)
        simulator_note = QLabel("使用本地模拟数据验证界面、SQLite 持久化与录制流程。")
        simulator_note.setWordWrap(True)
        simulator_note.setObjectName("settingsHint")
        simulator_layout.addWidget(simulator_note)
        simulator_layout.addStretch()
        self.source_stack.addWidget(self.realsense_page)
        self.source_stack.addWidget(self.ndi_page)
        self.source_stack.addWidget(simulator_page)
        outer.addWidget(self.source_stack)

        database_title = QLabel("本地数据")
        database_title.setObjectName("settingsSectionTitle")
        outer.addWidget(database_title)
        self.database_label = QLabel(str(config.database_path))
        self.database_label.setObjectName("databasePath")
        self.database_label.setWordWrap(True)
        self.database_label.setTextInteractionFlags(self.database_label.textInteractionFlags())
        outer.addWidget(self.database_label)
        self.spatial_button = QPushButton("打开 Open3D 三维轨迹")
        self.spatial_button.setObjectName("secondaryAction")
        self.spatial_button.clicked.connect(self.spatial_view_requested.emit)
        outer.addWidget(self.spatial_button)

        action_row = QHBoxLayout()
        self.save_button = QPushButton("保存设置")
        self.save_button.setObjectName("secondaryAction")
        self.save_button.clicked.connect(self.save_requested.emit)
        self.connect_button = QPushButton("连接设备")
        self.connect_button.setObjectName("primaryAction")
        self.connect_button.clicked.connect(self.connect_requested.emit)
        action_row.addWidget(self.save_button)
        action_row.addWidget(self.connect_button)
        outer.addLayout(action_row)
        outer.addStretch()
        self._settings_inputs = (
            self.serial_edit,
            self.width_spin,
            self.height_spin,
            self.fps_spin,
            self.family_combo,
            self.ids_edit,
            self.rows_spin,
            self.cols_spin,
            self.tag_size_spin,
            self.spacing_spin,
            self.min_visible_spin,
            self.record_format_combo,
            self.tracker_combo,
            self.ip_edit,
            self.port_spin,
            self.ndi_serial_edit,
            self.rom_edit,
            self.orientation_combo,
            self.ndi_format_combo,
        )
        self.load_config(config)

    def _build_realsense_page(self) -> QWidget:
        page = QFrame()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(10)
        title = QLabel("RealSense / AprilTag")
        title.setObjectName("settingsSectionTitle")
        layout.addWidget(title)
        form = QFormLayout()
        form.setSpacing(8)
        self.serial_edit = QLineEdit()
        self.serial_edit.setPlaceholderText("留空使用第一台设备")
        form.addRow("序列号", self.serial_edit)

        resolution = QWidget()
        resolution_layout = QHBoxLayout(resolution)
        resolution_layout.setContentsMargins(0, 0, 0, 0)
        resolution_layout.setSpacing(6)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(320, 4096)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(240, 2160)
        resolution_layout.addWidget(self.width_spin)
        resolution_layout.addWidget(QLabel("×"))
        resolution_layout.addWidget(self.height_spin)
        form.addRow("分辨率", resolution)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setSuffix(" FPS")
        form.addRow("帧率", self.fps_spin)
        self.family_combo = QComboBox()
        for family in ("tag36h11", "tag25h9", "tag16h5", "tagCircle21h7"):
            self.family_combo.addItem(family, family.lower())
        form.addRow("Tag Family", self.family_combo)
        self.ids_edit = QLineEdit()
        self.ids_edit.setPlaceholderText("0,1,2,3")
        form.addRow("Tag IDs", self.ids_edit)

        board = QWidget()
        board_layout = QHBoxLayout(board)
        board_layout.setContentsMargins(0, 0, 0, 0)
        board_layout.setSpacing(6)
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 20)
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 20)
        board_layout.addWidget(self.rows_spin)
        board_layout.addWidget(QLabel("×"))
        board_layout.addWidget(self.cols_spin)
        form.addRow("标定板", board)

        self.tag_size_spin = QDoubleSpinBox()
        self.tag_size_spin.setRange(1.0, 1000.0)
        self.tag_size_spin.setDecimals(2)
        self.tag_size_spin.setSuffix(" mm")
        form.addRow("Tag 边长", self.tag_size_spin)
        self.spacing_spin = QDoubleSpinBox()
        self.spacing_spin.setRange(0.0, 1000.0)
        self.spacing_spin.setDecimals(2)
        self.spacing_spin.setSuffix(" mm")
        form.addRow("Tag 间距", self.spacing_spin)
        self.min_visible_spin = QSpinBox()
        self.min_visible_spin.setRange(1, 400)
        form.addRow("最少可见", self.min_visible_spin)
        self.record_format_combo = QComboBox()
        self.record_format_combo.addItem("XLSX", "xlsx")
        self.record_format_combo.addItem("CSV", "csv")
        form.addRow("位置导出", self.record_format_combo)
        layout.addLayout(form)

        self.calibration_status = QLabel()
        self.calibration_status.setObjectName("calibrationStatus")
        self.calibration_status.setWordWrap(True)
        layout.addWidget(self.calibration_status)
        calibration_actions = QHBoxLayout()
        self.clear_button = QPushButton("清除零点")
        self.clear_button.clicked.connect(self.clear_calibration_requested.emit)
        self.calibration_button = QPushButton("开始初始标定")
        self.calibration_button.clicked.connect(self.calibration_requested.emit)
        calibration_actions.addWidget(self.clear_button)
        calibration_actions.addWidget(self.calibration_button)
        layout.addLayout(calibration_actions)
        return page

    def _build_ndi_page(self) -> QWidget:
        page = QFrame()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(10)
        title = QLabel("NDI / ROM 工具")
        title.setObjectName("settingsSectionTitle")
        layout.addWidget(title)
        form = QFormLayout()
        form.setSpacing(8)
        self.tracker_combo = QComboBox()
        self.tracker_combo.addItem("Vega（网络）", "vega")
        self.tracker_combo.addItem("Polaris（串口）", "polaris")
        self.tracker_combo.currentIndexChanged.connect(self._sync_ndi_fields)
        form.addRow("设备类型", self.tracker_combo)
        self.ip_edit = QLineEdit()
        form.addRow("IP 地址", self.ip_edit)
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        form.addRow("端口", self.port_spin)
        self.ndi_serial_edit = QLineEdit()
        form.addRow("串口", self.ndi_serial_edit)
        self.rom_edit = QLineEdit()
        self.rom_edit.setPlaceholderText("roms/tool_1.rom")
        form.addRow("ROM 文件", self.rom_edit)
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItem("四元数", "quaternion")
        self.orientation_combo.addItem("方向余弦矩阵", "matrix")
        form.addRow("姿态保存", self.orientation_combo)
        self.ndi_format_combo = QComboBox()
        self.ndi_format_combo.addItem("CSV", "csv")
        self.ndi_format_combo.addItem("XLSX", "xlsx")
        form.addRow("轨迹文件", self.ndi_format_combo)
        layout.addLayout(form)
        return page

    def _source_index_changed(self) -> None:
        source = str(self.source_combo.currentData())
        self._sync_source_stack(source)
        self.source_changed.emit(source)

    def _sync_source_stack(self, source: str) -> None:
        self.source_stack.setCurrentIndex({"realsense": 0, "ndi": 1, "simulator": 2}[source])

    def _sync_ndi_fields(self) -> None:
        is_vega = self.tracker_combo.currentData() == "vega"
        editable = not self._connection_active
        self.ip_edit.setEnabled(is_vega and editable)
        self.port_spin.setEnabled(is_vega and editable)
        self.ndi_serial_edit.setEnabled((not is_vega) and editable)

    @property
    def selected_source(self) -> str:
        return str(self.source_combo.currentData())

    def load_config(self, config: AppConfig) -> None:
        self.config = config
        self.source_combo.blockSignals(True)
        self.source_combo.setCurrentIndex(max(0, self.source_combo.findData(config.source)))
        self.source_combo.blockSignals(False)
        self._sync_source_stack(config.source)
        rs = config.realsense
        self.serial_edit.setText(rs.serial)
        self.width_spin.setValue(rs.width)
        self.height_spin.setValue(rs.height)
        self.fps_spin.setValue(rs.fps)
        self.family_combo.setCurrentIndex(max(0, self.family_combo.findData(rs.tag_family)))
        self.ids_edit.setText(",".join(str(tag_id) for tag_id in rs.tag_ids))
        self.rows_spin.setValue(rs.board_rows)
        self.cols_spin.setValue(rs.board_cols)
        self.tag_size_spin.setValue(rs.tag_size_m * 1000.0)
        self.spacing_spin.setValue(rs.tag_spacing_m * 1000.0)
        self.min_visible_spin.setValue(rs.min_visible_tags)
        self.record_format_combo.setCurrentIndex(
            max(0, self.record_format_combo.findData(rs.record_format))
        )
        ndi = config.ndi
        self.tracker_combo.setCurrentIndex(max(0, self.tracker_combo.findData(ndi.tracker_type)))
        self.ip_edit.setText(ndi.ip_address)
        self.port_spin.setValue(ndi.port)
        self.ndi_serial_edit.setText(ndi.serial_port)
        self.rom_edit.setText(",".join(str(path) for path in ndi.rom_files))
        self.orientation_combo.setCurrentIndex(
            max(0, self.orientation_combo.findData(ndi.record_orientation))
        )
        self.ndi_format_combo.setCurrentIndex(
            max(0, self.ndi_format_combo.findData(ndi.record_format))
        )
        self._sync_ndi_fields()
        self._set_reference_status()

    def to_config(self, current: AppConfig) -> AppConfig:
        source = self.selected_source
        realsense = current.realsense
        ndi = current.ndi
        if source in {"realsense", "simulator"}:
            realsense = build_realsense_config(
                serial=self.serial_edit.text(),
                width=self.width_spin.value(),
                height=self.height_spin.value(),
                fps=self.fps_spin.value(),
                tag_family=str(self.family_combo.currentData()),
                tag_ids=self.ids_edit.text().split(","),
                board_rows=self.rows_spin.value(),
                board_cols=self.cols_spin.value(),
                tag_size_m=self.tag_size_spin.value() / 1000.0,
                tag_spacing_m=self.spacing_spin.value() / 1000.0,
                min_visible_tags=self.min_visible_spin.value(),
                record_format=str(self.record_format_combo.currentData()),
                reference_transform=current.realsense.reference_transform,
                calibration_samples=current.realsense.calibration_samples,
                calibration_max_std_mm=current.realsense.calibration_max_std_mm,
                calibration_max_angle_deg=current.realsense.calibration_max_angle_deg,
            )
        if source == "ndi":
            ndi = build_ndi_config(
                root=current.root,
                tracker_type=str(self.tracker_combo.currentData()),
                ip_address=self.ip_edit.text(),
                port=self.port_spin.value(),
                serial_port=self.ndi_serial_edit.text(),
                rom_files=self.rom_edit.text().split(","),
                record_orientation=str(self.orientation_combo.currentData()),
                record_format=str(self.ndi_format_combo.currentData()),
            )
        return replace(current, source=source, realsense=realsense, ndi=ndi)

    def set_connection_active(self, active: bool) -> None:
        self._connection_active = active
        self.source_combo.setEnabled(not active)
        self.save_button.setEnabled(not active)
        for widget in self._settings_inputs:
            widget.setEnabled(not active)
        self._sync_ndi_fields()
        self.connect_button.setText("断开设备" if active else "连接设备")
        self.calibration_button.setEnabled(active and self.selected_source == "realsense")

    def set_connecting(self) -> None:
        self.connect_button.setEnabled(False)
        self.connect_button.setText("连接中…")

    def set_connected(self, active: bool) -> None:
        self.connect_button.setEnabled(True)
        self.set_connection_active(active)

    def set_calibration_progress(self, current: int, target: int) -> None:
        self.calibration_status.setText(f"标定中：保持标定板静止 · {current}/{target}")
        self.calibration_button.setText("取消标定")

    def finish_calibration(self, message: str | None = None) -> None:
        self.calibration_button.setText("开始初始标定")
        if message:
            self.calibration_status.setText(message)
        else:
            self._set_reference_status()

    def _set_reference_status(self) -> None:
        calibrated = not is_identity_reference(self.config.realsense.reference_transform)
        self.calibration_status.setText(
            "● 已保存初始零点" if calibrated else "○ 未标定 · 连接相机后保持标定板静止"
        )
        self.clear_button.setEnabled(calibrated)
