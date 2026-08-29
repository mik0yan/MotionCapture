from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QFontDatabase, QIcon, QImage, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from motion_capture.models import TagDetection
from motion_capture.storage import TagMonitorProfile


COLORS = {
    "canvas": "#F7F9FC",
    "surface": "#FFFFFF",
    "primary": "#0B121B",
    "secondary": "#3F4D5A",
    "muted": "#5C6A78",
    "brand": "#0049C0",
    "brand_text": "#003A99",
    "border": "#D9E0E8",
    "subtle": "#EEF2F7",
    "success": "#12B76A",
    "proximity": "#FFB020",
    "danger": "#E5484D",
}


@lru_cache(maxsize=1)
def preferred_font_family() -> str:
    families = QFontDatabase().families()
    return "Noto Sans SC" if "Noto Sans SC" in families else "PingFang SC"


def app_font(size: int, weight: int = QFont.Medium) -> QFont:
    font = QFont(preferred_font_family(), size)
    font.setStyleHint(QFont.SansSerif)
    font.setWeight(weight)
    return font


class MetricCard(QFrame):
    def __init__(self, label: str, value: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("metricCard")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(68)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 9, 16, 9)
        layout.setSpacing(2)
        label_widget = QLabel(label)
        label_widget.setObjectName("metricLabel")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricValue")
        layout.addWidget(label_widget)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


@dataclass(frozen=True)
class CameraFrame:
    image: np.ndarray | None
    detections: tuple[TagDetection, ...]
    tones: dict[int, str]


class CameraCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.frame = CameraFrame(None, (), {})
        self.show_rgb = True
        self.show_depth = True
        self.show_tags = True
        self.setMinimumSize(640, 420)

    def set_frame(
        self,
        image: np.ndarray | None,
        detections: tuple[TagDetection, ...],
        tones: dict[int, str],
    ) -> None:
        self.frame = CameraFrame(None if image is None else np.asarray(image).copy(), detections, tones)
        self.update()

    def set_layer(self, layer: str, visible: bool) -> None:
        if layer == "rgb":
            self.show_rgb = visible
        elif layer == "depth":
            self.show_depth = visible
        elif layer == "tags":
            self.show_tags = visible
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        target = QRectF(self.rect())
        painter.fillRect(target, QColor("#06101B"))
        source_size = (1280.0, 720.0)
        draw_rect = target
        image = self.frame.image
        if self.show_rgb and image is not None and image.ndim == 3:
            height, width, channels = image.shape
            source_size = (float(width), float(height))
            qimage = QImage(
                image.data,
                width,
                height,
                channels * width,
                QImage.Format_RGB888,
            ).copy()
            scale = max(target.width() / width, target.height() / height)
            drawn_width = width * scale
            drawn_height = height * scale
            draw_rect = QRectF(
                (target.width() - drawn_width) / 2.0,
                (target.height() - drawn_height) / 2.0,
                drawn_width,
                drawn_height,
            )
            painter.drawImage(draw_rect, qimage)
        else:
            gradient_color = QColor("#0A1C2E")
            painter.fillRect(target, gradient_color)

        grid_pen = QPen(QColor(217, 224, 232, 42), 1)
        painter.setPen(grid_pen)
        for column in range(1, 8):
            x = target.width() * column / 8.0
            painter.drawLine(QPointF(x, 0), QPointF(x, target.height()))
        for row in range(1, 6):
            y = target.height() * row / 6.0
            painter.drawLine(QPointF(0, y), QPointF(target.width(), y))

        if self.show_tags:
            scale_x = draw_rect.width() / source_size[0]
            scale_y = draw_rect.height() / source_size[1]
            for detection in self.frame.detections:
                points = [
                    QPointF(draw_rect.x() + x * scale_x, draw_rect.y() + y * scale_y)
                    for x, y in detection.corners_px
                ]
                if len(points) != 4:
                    continue
                tone = self.frame.tones.get(detection.tag_id, "normal")
                color = QColor(
                    COLORS[
                        "danger"
                        if tone == "danger"
                        else "proximity"
                        if tone == "proximity"
                        else "success"
                    ]
                )
                painter.setPen(QPen(color, 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawPolygon(QPolygonF(points))
                top_left = min(points, key=lambda point: point.y())
                painter.setFont(app_font(10, QFont.DemiBold))
                painter.setPen(color)
                painter.drawText(
                    QPointF(top_left.x(), max(16.0, top_left.y() - 8.0)),
                    f"ID {detection.tag_id:02d}  ·  {detection.distance_mm:.1f} mm",
                )
                if self.show_depth:
                    center = QPointF(
                        sum(point.x() for point in points) / 4.0,
                        sum(point.y() for point in points) / 4.0,
                    )
                    painter.setPen(QColor("#FFFFFF"))
                    painter.setFont(app_font(9, QFont.DemiBold))
                    painter.drawText(
                        QRectF(center.x() - 42, center.y() - 10, 84, 20),
                        Qt.AlignCenter,
                        f"D  {detection.distance_mm:.1f}",
                    )
        painter.end()


class CameraView(QFrame):
    record_requested = pyqtSignal()

    def __init__(self, asset_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("cameraView")
        self.canvas = CameraCanvas(self)
        self.layer_button = QToolButton(self)
        self.layer_button.setObjectName("glassButton")
        self.layer_button.setCheckable(True)
        self.layer_button.setChecked(True)
        self.layer_button.setIcon(QIcon(str(asset_root / "icon_layers.svg")))
        self.layer_button.setIconSize(self.layer_button.sizeHint())
        self.layer_button.clicked.connect(self._sync_layer_panel)

        self.layer_panel = QFrame(self)
        self.layer_panel.setObjectName("layerPanel")
        panel_layout = QVBoxLayout(self.layer_panel)
        panel_layout.setContentsMargins(8, 8, 8, 8)
        panel_layout.setSpacing(4)
        self.layer_toggles: dict[str, QPushButton] = {}
        for key, label in (("rgb", "RGB"), ("depth", "D"), ("tags", "Tag")):
            button = QPushButton(label)
            button.setObjectName("layerToggle")
            button.setCheckable(True)
            button.setChecked(True)
            button.setFixedHeight(32)
            button.toggled.connect(lambda checked, layer=key: self.canvas.set_layer(layer, checked))
            panel_layout.addWidget(button)
            self.layer_toggles[key] = button

        self.timer_label = QLabel("●  00:00:00", self)
        self.timer_label.setObjectName("recordTimer")
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.hide()
        self.record_button = QPushButton("●", self)
        self.record_button.setObjectName("recordControl")
        self.record_button.setEnabled(False)
        self.record_button.clicked.connect(self.record_requested.emit)
        self.set_recording(False, 0)

    def _sync_layer_panel(self) -> None:
        self.layer_panel.setVisible(self.layer_button.isChecked())

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self.canvas.setGeometry(self.rect())
        self.layer_button.setGeometry(16, 16, 44, 44)
        self.layer_panel.setGeometry(16, 68, 124, 124)
        self.record_button.setGeometry(max(16, self.width() - 60), 16, 44, 44)
        self.timer_label.setGeometry(max(16, self.width() - 176), 16, 108, 44)

    def set_record_enabled(self, enabled: bool) -> None:
        self.record_button.setEnabled(enabled)

    def set_recording(self, active: bool, elapsed_seconds: int) -> None:
        hours, remainder = divmod(max(0, elapsed_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        self.timer_label.setText(f"●  {hours:02d}:{minutes:02d}:{seconds:02d}")
        self.timer_label.setVisible(active)
        self.record_button.setText("■" if active else "●")
        self.record_button.setProperty("active", active)
        self.record_button.style().unpolish(self.record_button)
        self.record_button.style().polish(self.record_button)


class TagMonitorCard(QFrame):
    monitor_toggled = pyqtSignal(int, bool)
    settings_requested = pyqtSignal(int)
    threshold_changed = pyqtSignal(int, float)

    def __init__(self, profile: TagMonitorProfile, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.profile = profile
        self.running = profile.enabled
        self.setObjectName("tagMonitorCard")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(92)
        self.setCursor(Qt.PointingHandCursor)

        self._long_press_timer = QTimer(self)
        self._long_press_timer.setSingleShot(True)
        self._long_press_timer.setInterval(500)
        self._long_press_timer.timeout.connect(self._emit_settings_requested)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 12, 8, 12)
        outer.setSpacing(12)
        data_layout = QVBoxLayout()
        data_layout.setSpacing(4)
        header = QHBoxLayout()
        header.setSpacing(8)
        self.id_label = QLabel(f"ID {profile.tag_id:02d}")
        self.id_label.setObjectName("tagId")
        self.config_label = QLabel(f"{profile.tag_size_mm:g} × {profile.tag_size_mm:g} mm")
        self.config_label.setObjectName("tagConfig")
        header.addWidget(self.id_label)
        header.addWidget(self.config_label)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setObjectName("tagThresholdSpin")
        self.threshold_spin.setRange(0.0, float("inf"))
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setPrefix("阈值 ")
        self.threshold_spin.setSuffix(" mm")
        self.threshold_spin.setFixedWidth(148)
        self.threshold_spin.setAlignment(Qt.AlignRight)
        self.threshold_spin.setValue(profile.quality_threshold)
        self.threshold_spin.valueChanged.connect(self._threshold_value_changed)
        header.addWidget(self.threshold_spin)
        header.addStretch()
        data_layout.addLayout(header)
        self.coordinates_label = QLabel("X —  ·  Y —  ·  Z —")
        self.coordinates_label.setObjectName("tagCoordinates")
        data_layout.addWidget(self.coordinates_label)
        self.offset_label = QLabel("偏移量：— mm ± —")
        self.offset_label.setObjectName("tagOffset")
        data_layout.addWidget(self.offset_label)
        outer.addLayout(data_layout, 1)

        action_layout = QVBoxLayout()
        action_layout.setSpacing(8)
        action_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.status_label = QLabel("● 监控中")
        self.status_label.setObjectName("tagStatus")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.toggle_button = QPushButton("■")
        self.toggle_button.setObjectName("tagAction")
        self.toggle_button.setFixedSize(36, 36)
        self.toggle_button.clicked.connect(self._toggle)
        action_layout.addWidget(self.status_label)
        action_layout.addWidget(self.toggle_button)
        outer.addLayout(action_layout)
        for label in (
            self.id_label,
            self.config_label,
            self.coordinates_label,
            self.offset_label,
            self.status_label,
        ):
            label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.set_state(self.running, "normal")

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.LeftButton:
            self._long_press_timer.start()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._long_press_timer.stop()
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._long_press_timer.stop()
        super().leaveEvent(event)

    def _emit_settings_requested(self) -> None:
        self.settings_requested.emit(self.profile.tag_id)

    def _toggle(self) -> None:
        self.set_state(not self.running, "normal" if not self.running else "danger")
        self.monitor_toggled.emit(self.profile.tag_id, self.running)

    def set_state(self, running: bool, tone: str) -> None:
        self.running = running
        if not running:
            tone = "danger"
        self.setProperty("tone", tone)
        self.setProperty("running", running)
        self.status_label.setText(
            "● 已停止"
            if not running
            else "● 接近中"
            if tone == "proximity"
            else "● 警告"
            if tone == "danger"
            else "● 监控中"
        )
        self.toggle_button.setText("■" if running else "▶")
        for widget in (
            self,
            self.status_label,
            self.offset_label,
            self.toggle_button,
        ):
            widget.style().unpolish(widget)
            widget.style().polish(widget)

    def update_values(
        self,
        position: tuple[float, float, float] | None,
        offset_mm: float | None,
        variance_mm: float | None,
        tone: str,
    ) -> None:
        self.set_state(self.running, tone)
        if position is None:
            self.coordinates_label.setText("X —  ·  Y —  ·  Z —")
            self.offset_label.setText("偏移量：— mm ± —")
            return
        x, y, z = position
        self.coordinates_label.setText(f"X {x:+.1f}  ·  Y {y:+.1f}  ·  Z {z:+.1f}")
        if offset_mm is None or variance_mm is None:
            self.offset_label.setText("偏移量：— mm ± —")
        else:
            self.offset_label.setText(f"偏移量：{offset_mm:.1f} mm ± {variance_mm:.2f}")

    def _threshold_value_changed(self, value: float) -> None:
        self.profile = replace(self.profile, quality_threshold=value)
        self.threshold_changed.emit(self.profile.tag_id, value)

    def set_profile(self, profile: TagMonitorProfile) -> None:
        self.profile = profile
        self.config_label.setText(f"{profile.tag_size_mm:g} × {profile.tag_size_mm:g} mm")
        self.threshold_spin.blockSignals(True)
        self.threshold_spin.setValue(profile.quality_threshold)
        self.threshold_spin.blockSignals(False)
