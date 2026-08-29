from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from motion_capture.calibration import is_identity_reference
from motion_capture.config import RealSenseConfig, build_realsense_config


class RealSenseConfigForm(QFrame):
    """RealSense stream/AprilTag editor with initial-reference controls."""

    save_requested = pyqtSignal()
    calibration_requested = pyqtSignal()
    clear_calibration_requested = pyqtSignal()

    def __init__(self, config: RealSenseConfig, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._connected = False
        self.setObjectName("realsenseConfigPanel")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 10, 14, 12)
        outer.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("RealSense 配置与初始标定")
        title.setObjectName("realsenseConfigTitle")
        header.addWidget(title)
        self.status_label = QLabel()
        self.status_label.setObjectName("calibrationStatus")
        header.addWidget(self.status_label)
        header.addStretch()
        self.clear_button = QPushButton("清除零点")
        self.clear_button.clicked.connect(lambda _checked=False: self.clear_calibration_requested.emit())
        header.addWidget(self.clear_button)
        self.calibration_button = QPushButton("开始初始标定")
        self.calibration_button.setObjectName("calibrationButton")
        self.calibration_button.clicked.connect(lambda _checked=False: self.calibration_requested.emit())
        header.addWidget(self.calibration_button)
        self.save_button = QPushButton("保存配置")
        self.save_button.clicked.connect(lambda _checked=False: self.save_requested.emit())
        header.addWidget(self.save_button)
        outer.addLayout(header)

        fields = QGridLayout()
        fields.setHorizontalSpacing(9)
        fields.setVerticalSpacing(8)

        fields.addWidget(QLabel("设备序列号"), 0, 0)
        self.serial_edit = QLineEdit()
        self.serial_edit.setPlaceholderText("留空使用第一台 RealSense")
        fields.addWidget(self.serial_edit, 0, 1)

        fields.addWidget(QLabel("彩色分辨率"), 0, 2)
        resolution = QWidget()
        resolution_layout = QHBoxLayout(resolution)
        resolution_layout.setContentsMargins(0, 0, 0, 0)
        resolution_layout.setSpacing(5)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(320, 4096)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(240, 2160)
        resolution_layout.addWidget(self.width_spin)
        resolution_layout.addWidget(QLabel("×"))
        resolution_layout.addWidget(self.height_spin)
        fields.addWidget(resolution, 0, 3)

        fields.addWidget(QLabel("FPS"), 0, 4)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        fields.addWidget(self.fps_spin, 0, 5)

        fields.addWidget(QLabel("Tag Family"), 0, 6)
        self.family_combo = QComboBox()
        for family in ("tag36h11", "tag25h9", "tag16h5", "tagCircle21h7"):
            self.family_combo.addItem(family, family.lower())
        fields.addWidget(self.family_combo, 0, 7)

        fields.addWidget(QLabel("Tag IDs"), 1, 0)
        self.ids_edit = QLineEdit()
        self.ids_edit.setPlaceholderText("0,1,2,3")
        fields.addWidget(self.ids_edit, 1, 1)

        fields.addWidget(QLabel("标定板行列"), 1, 2)
        board_shape = QWidget()
        board_layout = QHBoxLayout(board_shape)
        board_layout.setContentsMargins(0, 0, 0, 0)
        board_layout.setSpacing(5)
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 20)
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 20)
        board_layout.addWidget(self.rows_spin)
        board_layout.addWidget(QLabel("×"))
        board_layout.addWidget(self.cols_spin)
        fields.addWidget(board_shape, 1, 3)

        fields.addWidget(QLabel("Tag 边长 mm"), 1, 4)
        self.tag_size_spin = QDoubleSpinBox()
        self.tag_size_spin.setRange(1.0, 1000.0)
        self.tag_size_spin.setDecimals(2)
        fields.addWidget(self.tag_size_spin, 1, 5)

        fields.addWidget(QLabel("间距 mm"), 1, 6)
        self.spacing_spin = QDoubleSpinBox()
        self.spacing_spin.setRange(0.0, 1000.0)
        self.spacing_spin.setDecimals(2)
        fields.addWidget(self.spacing_spin, 1, 7)

        fields.addWidget(QLabel("最少可见 Tag"), 2, 0)
        self.min_visible_spin = QSpinBox()
        self.min_visible_spin.setRange(1, 400)
        self.min_visible_spin.setToolTip("可见的标定板 Tag 少于该数量时，不输出位置数据")
        fields.addWidget(self.min_visible_spin, 2, 1)

        fields.addWidget(QLabel("位置导出"), 2, 2)
        self.record_format_combo = QComboBox()
        self.record_format_combo.addItem("XLSX", "xlsx")
        self.record_format_combo.addItem("CSV", "csv")
        self.record_format_combo.setToolTip("RealSense 记录只包含公共字段和 XYZ 位置")
        fields.addWidget(self.record_format_combo, 2, 3)

        export_note = QLabel("经初始零点转换后的 XYZ · 数值单位 mm")
        export_note.setObjectName("realsenseExportNote")
        fields.addWidget(export_note, 2, 4, 1, 4)

        fields.setColumnStretch(1, 3)
        fields.setColumnStretch(3, 2)
        fields.setColumnStretch(7, 2)
        outer.addLayout(fields)

        self._settings_widgets = (
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
            self.save_button,
        )
        self.load_config(config)
        self.set_connection_active(False)

    def load_config(self, config: RealSenseConfig) -> None:
        self._config = config
        self.serial_edit.setText(config.serial)
        self.width_spin.setValue(config.width)
        self.height_spin.setValue(config.height)
        self.fps_spin.setValue(config.fps)
        family_index = self.family_combo.findData(config.tag_family)
        self.family_combo.setCurrentIndex(max(0, family_index))
        self.ids_edit.setText(",".join(str(tag_id) for tag_id in config.tag_ids))
        self.rows_spin.setValue(config.board_rows)
        self.cols_spin.setValue(config.board_cols)
        self.tag_size_spin.setValue(config.tag_size_m * 1000.0)
        self.spacing_spin.setValue(config.tag_spacing_m * 1000.0)
        self.min_visible_spin.setValue(config.min_visible_tags)
        format_index = self.record_format_combo.findData(config.record_format)
        self.record_format_combo.setCurrentIndex(max(0, format_index))
        self._set_reference_status()

    def to_config(self) -> RealSenseConfig:
        return build_realsense_config(
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
            reference_transform=self._config.reference_transform,
            calibration_samples=self._config.calibration_samples,
            calibration_max_std_mm=self._config.calibration_max_std_mm,
            calibration_max_angle_deg=self._config.calibration_max_angle_deg,
        )

    def set_connection_active(self, active: bool, calibration_available: bool = True) -> None:
        self._connected = active
        for widget in self._settings_widgets:
            widget.setEnabled(not active)
        self.calibration_button.setEnabled(active and calibration_available)
        self.clear_button.setEnabled(not is_identity_reference(self._config.reference_transform))
        if not active:
            self.calibration_button.setText("开始初始标定")
            self._set_reference_status()

    def set_calibration_progress(self, current: int, target: int) -> None:
        self.status_label.setText(f"标定中：保持标定板静止 · {current}/{target}")
        self.status_label.setProperty("calibrated", False)
        self.calibration_button.setText("取消标定")

    def finish_calibration(self, message: str | None = None) -> None:
        self.calibration_button.setText("开始初始标定")
        if message:
            self.status_label.setText(message)
        else:
            self._set_reference_status()

    def _set_reference_status(self) -> None:
        calibrated = not is_identity_reference(self._config.reference_transform)
        if calibrated:
            self.status_label.setText("● 已保存初始零点")
        else:
            self.status_label.setText("○ 未标定 · 连接相机后保持标定板静止")
        self.status_label.setProperty("calibrated", calibrated)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
