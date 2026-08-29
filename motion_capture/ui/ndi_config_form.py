from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from motion_capture.config import NDIConfig, build_ndi_config
from motion_capture.rom_importer import import_rom_files


class NDIConfigForm(QFrame):
    """Compact editor for Vega/Polaris connection and ROM settings."""

    save_requested = pyqtSignal()

    def __init__(self, root: Path, config: NDIConfig, parent=None) -> None:
        super().__init__(parent)
        self.root = root.resolve()
        self.setObjectName("ndiConfigPanel")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 10, 14, 12)
        outer.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("NDI 连接配置")
        title.setObjectName("ndiConfigTitle")
        self.hint_label = QLabel("连接前自动保存到根目录 .env")
        self.hint_label.setObjectName("ndiConfigHint")
        header.addWidget(title)
        header.addWidget(self.hint_label)
        header.addStretch()
        header.addWidget(QLabel("姿态保存"))
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItem("四元数", "quaternion")
        self.orientation_combo.addItem("方向余弦矩阵", "matrix")
        header.addWidget(self.orientation_combo)
        header.addWidget(QLabel("文件"))
        self.format_combo = QComboBox()
        self.format_combo.addItem("CSV", "csv")
        self.format_combo.addItem("XLSX", "xlsx")
        header.addWidget(self.format_combo)
        self.save_button = QPushButton("保存配置")
        self.save_button.setObjectName("secondaryButton")
        self.save_button.clicked.connect(lambda _checked=False: self.save_requested.emit())
        header.addWidget(self.save_button)
        outer.addLayout(header)

        fields = QGridLayout()
        fields.setHorizontalSpacing(10)
        fields.setVerticalSpacing(8)

        fields.addWidget(QLabel("设备类型"), 0, 0)
        self.tracker_combo = QComboBox()
        self.tracker_combo.addItem("Vega（网络）", "vega")
        self.tracker_combo.addItem("Polaris（串口）", "polaris")
        fields.addWidget(self.tracker_combo, 0, 1)

        self.ip_label = QLabel("IP 地址")
        fields.addWidget(self.ip_label, 0, 2)
        self.ip_edit = QLineEdit()
        self.ip_edit.setPlaceholderText("例如 192.168.2.17")
        fields.addWidget(self.ip_edit, 0, 3)

        self.port_label = QLabel("端口")
        fields.addWidget(self.port_label, 0, 4)
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        fields.addWidget(self.port_spin, 0, 5)

        self.serial_label = QLabel("串口")
        fields.addWidget(self.serial_label, 0, 6)
        self.serial_edit = QLineEdit()
        self.serial_edit.setPlaceholderText("COM3 或 /dev/cu.usbserial…")
        fields.addWidget(self.serial_edit, 0, 7)

        fields.addWidget(QLabel("ROM 工具文件"), 1, 0)
        self.rom_edit = QLineEdit()
        self.rom_edit.setPlaceholderText("roms/tool_1.rom,roms/tool_2.rom")
        fields.addWidget(self.rom_edit, 1, 1, 1, 6)
        self.browse_button = QPushButton("导入 ROM…")
        self.browse_button.setMaximumWidth(170)
        self.browse_button.clicked.connect(self._import_rom_files)
        fields.addWidget(self.browse_button, 1, 7)
        fields.setColumnStretch(3, 2)
        fields.setColumnStretch(7, 1)
        outer.addLayout(fields)

        self.tracker_combo.currentIndexChanged.connect(self._sync_tracker_fields)
        self.load_config(config)

    def load_config(self, config: NDIConfig) -> None:
        index = self.tracker_combo.findData(config.tracker_type)
        self.tracker_combo.setCurrentIndex(max(0, index))
        self.ip_edit.setText(config.ip_address)
        self.port_spin.setValue(config.port)
        self.serial_edit.setText(config.serial_port)
        self.rom_edit.setText(",".join(self._display_path(path) for path in config.rom_files))
        orientation_index = self.orientation_combo.findData(config.record_orientation)
        self.orientation_combo.setCurrentIndex(max(0, orientation_index))
        format_index = self.format_combo.findData(config.record_format)
        self.format_combo.setCurrentIndex(max(0, format_index))
        self._sync_tracker_fields()

    def to_config(self) -> NDIConfig:
        return build_ndi_config(
            root=self.root,
            tracker_type=str(self.tracker_combo.currentData()),
            ip_address=self.ip_edit.text(),
            port=self.port_spin.value(),
            serial_port=self.serial_edit.text(),
            rom_files=self.rom_edit.text().split(","),
            record_orientation=str(self.orientation_combo.currentData()),
            record_format=str(self.format_combo.currentData()),
        )

    def set_connection_active(self, active: bool) -> None:
        self.setEnabled(not active)

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def _import_rom_files(self) -> None:
        initial_dir = self.root / "roms"
        if not initial_dir.is_dir():
            initial_dir = self.root
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "导入 NDI ROM 工具文件",
            str(initial_dir),
            "NDI ROM 文件 (*.rom);;所有文件 (*)",
        )
        if paths:
            try:
                result = import_rom_files(self.root, paths)
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "ROM 导入失败", str(exc))
                return
            current = [value.strip() for value in self.rom_edit.text().split(",") if value.strip()]
            imported = [self._display_path(path) for path in result.paths]
            merged = list(dict.fromkeys((*current, *imported)))
            self.rom_edit.setText(",".join(merged))
            self.hint_label.setText(
                f"ROM 已导入：新增 {result.copied_count}，复用 {result.reused_count}"
            )
            self.save_requested.emit()

    def _sync_tracker_fields(self) -> None:
        is_vega = self.tracker_combo.currentData() == "vega"
        for widget in (self.ip_label, self.ip_edit, self.port_label, self.port_spin):
            widget.setVisible(is_vega)
        for widget in (self.serial_label, self.serial_edit):
            widget.setVisible(not is_vega)
