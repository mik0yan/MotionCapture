from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QPushButton, QSizePolicy


class WorkModeSwitcher(QFrame):
    """Exclusive three-state button switcher for tracking backends."""

    mode_changed = pyqtSignal(str)
    MODES = (
        ("ndi", "NDI 模式"),
        ("realsense", "RealSense"),
        ("simulator", "模拟器模式"),
    )

    def __init__(self, initial_mode: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("modeSwitcher")
        self._buttons: dict[str, QPushButton] = {}
        self._current_mode = ""
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(3)
        for mode, label in self.MODES:
            button = QPushButton(label)
            button.setObjectName("modeButton")
            button.setCheckable(True)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.setAccessibleName(f"工作模式：{label}")
            button.clicked.connect(
                lambda checked, selected_mode=mode: self._on_clicked(selected_mode, checked)
            )
            self._group.addButton(button)
            self._buttons[mode] = button
            layout.addWidget(button)
        self.set_mode(initial_mode)

    @property
    def current_mode(self) -> str:
        if not self._current_mode:
            raise RuntimeError("工作模式切换器没有选中项")
        return self._current_mode

    def set_mode(self, mode: str) -> None:
        if mode not in self._buttons:
            raise ValueError(f"不支持的工作模式：{mode}")
        self._buttons[mode].setChecked(True)
        self._current_mode = mode

    def button(self, mode: str) -> QPushButton:
        return self._buttons[mode]

    def _on_clicked(self, mode: str, checked: bool) -> None:
        if checked and mode != self._current_mode:
            self._current_mode = mode
            self.mode_changed.emit(mode)
