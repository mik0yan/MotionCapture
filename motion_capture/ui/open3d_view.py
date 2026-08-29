from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from importlib import import_module
from typing import Iterable

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout

from motion_capture.config import Open3DConfig
from motion_capture.models import PoseSample


def pose_transform_mm(sample: PoseSample) -> np.ndarray:
    """Return a homogeneous tool-to-world transform using millimetres."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(sample.rotation_matrix, dtype=np.float64)
    transform[:3, 3] = np.asarray(sample.position_mm, dtype=np.float64)
    return transform


def trajectory_segments(point_count: int) -> np.ndarray:
    """Return consecutive line indices for an Open3D LineSet."""
    if point_count < 2:
        return np.empty((0, 2), dtype=np.int32)
    return np.column_stack((np.arange(point_count - 1), np.arange(1, point_count))).astype(np.int32)


class Open3DViewWidget(QFrame):
    """Render Open3D geometry in a hidden context and present it inside Qt."""

    _PALETTE = (
        (0.13, 0.83, 0.65),
        (0.25, 0.65, 0.96),
        (0.98, 0.65, 0.24),
        (0.77, 0.48, 0.95),
        (0.96, 0.42, 0.55),
    )

    def __init__(self, config: Open3DConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self.setObjectName("open3dPanel")
        self.setMinimumHeight(200)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(7)
        header = QHBoxLayout()
        title = QLabel("Open3D 三维空间 · XYZ / mm")
        title.setObjectName("open3dTitle")
        header.addWidget(title)
        header.addStretch()
        self._summary_label = QLabel("等待位姿数据")
        self._summary_label.setObjectName("open3dSummary")
        header.addWidget(self._summary_label)
        layout.addLayout(header)

        self._render_label = QLabel("正在初始化 Open3D…" if config.enabled else "Open3D 已通过 .env 禁用")
        self._render_label.setObjectName("open3dCanvas")
        self._render_label.setAlignment(Qt.AlignCenter)
        self._render_label.setMinimumHeight(145)
        layout.addWidget(self._render_label, 1)

        self._latest: dict[str, PoseSample] = {}
        self._trails: dict[str, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.config.trail_points)
        )
        self._dynamic_geometries: list[object] = []
        self._visualizer = None
        self._o3d = None
        self._last_image: QImage | None = None
        self._disabled_reason: str | None = None
        self._dirty = True
        self._render_failures = 0

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.timeout.connect(self._render_if_dirty)
        if config.enabled:
            self._timer.start(max(16, round(1000 / config.render_hz)))

    def add_samples(self, samples: Iterable[PoseSample]) -> None:
        changed = False
        for sample in samples:
            position = np.asarray(sample.position_mm, dtype=np.float64)
            rotation = np.asarray(sample.rotation_matrix, dtype=np.float64)
            if not sample.valid or position.shape != (3,) or rotation.shape != (3, 3):
                continue
            if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
                continue
            self._latest[sample.tool_id] = sample
            self._trails[sample.tool_id].append(position.copy())
            changed = True
        if changed:
            self._dirty = True

    def clear(self) -> None:
        self._latest.clear()
        self._trails.clear()
        self._dirty = True
        self._summary_label.setText("等待位姿数据")

    def shutdown(self) -> None:
        self._timer.stop()
        self._dynamic_geometries.clear()
        if self._visualizer is not None:
            self._visualizer.destroy_window()
        self._visualizer = None
        self._o3d = None

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if self._last_image is not None:
            self._set_qimage(self._last_image)

    def _ensure_visualizer(self) -> bool:
        if self._visualizer is not None:
            return True
        if self._disabled_reason is not None or not self.config.enabled:
            return False
        try:
            self._o3d = import_module("open3d")
            visualizer = self._o3d.visualization.Visualizer()
            created = visualizer.create_window(
                window_name="MotionCapture Open3D Renderer",
                width=self.config.width,
                height=self.config.height,
                visible=False,
            )
            if not created:
                raise RuntimeError("Open3D 无法创建隐藏 GLFW 渲染窗口")
            self._visualizer = visualizer
            options = visualizer.get_render_option()
            options.background_color = np.array((0.039, 0.063, 0.090), dtype=np.float64)
            options.line_width = 2.0
            options.point_size = 6.0
            options.show_coordinate_frame = True
            visualizer.add_geometry(self._make_grid(), reset_bounding_box=True)
            view = visualizer.get_view_control()
            current_fov = view.get_field_of_view()
            view.change_field_of_view((self.config.camera_fov_deg - current_fov) / 5.0)
            view.set_lookat((0.0, 0.0, 450.0))
            view.set_front((-0.55, 0.70, -0.45))
            view.set_up((0.0, 0.0, 1.0))
            view.set_zoom(0.55)
            return True
        except Exception as exc:
            self._disabled_reason = f"{type(exc).__name__}: {exc}"
            self._render_label.setText(
                "Open3D 初始化失败\n"
                f"{self._disabled_reason}\n"
                "请确认已安装 open3d，并检查 GLFW/图形驱动。"
            )
            self._summary_label.setText("Open3D 不可用")
            self._timer.stop()
            return False

    def _make_grid(self):
        extent = 1000.0
        step = 100.0
        values = np.arange(-extent, extent + step, step, dtype=np.float64)
        points: list[tuple[float, float, float]] = []
        lines: list[tuple[int, int]] = []
        for value in values:
            start = len(points)
            points.extend(((-extent, value, 0.0), (extent, value, 0.0)))
            lines.append((start, start + 1))
            start = len(points)
            points.extend(((value, -extent, 0.0), (value, extent, 0.0)))
            lines.append((start, start + 1))
        grid = self._o3d.geometry.LineSet()
        grid.points = self._o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        grid.lines = self._o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        grid.colors = self._o3d.utility.Vector3dVector(
            np.tile(np.array((0.15, 0.22, 0.29), dtype=np.float64), (len(lines), 1))
        )
        return grid

    def _tool_color(self, tool_id: str) -> tuple[float, float, float]:
        digest = hashlib.sha1(tool_id.encode("utf-8"), usedforsecurity=False).digest()
        return self._PALETTE[digest[0] % len(self._PALETTE)]

    def _replace_dynamic_geometry(self) -> None:
        for geometry in self._dynamic_geometries:
            self._visualizer.remove_geometry(geometry, reset_bounding_box=False)
        self._dynamic_geometries.clear()

        for tool_id, sample in sorted(self._latest.items()):
            color = self._tool_color(tool_id)

            trail_points = np.asarray(self._trails[tool_id], dtype=np.float64)
            segments = trajectory_segments(len(trail_points))
            if len(segments):
                trail = self._o3d.geometry.LineSet()
                trail.points = self._o3d.utility.Vector3dVector(trail_points)
                trail.lines = self._o3d.utility.Vector2iVector(segments)
                trail.colors = self._o3d.utility.Vector3dVector(
                    np.tile(np.asarray(color, dtype=np.float64), (len(segments), 1))
                )
                self._visualizer.add_geometry(trail, reset_bounding_box=False)
                self._dynamic_geometries.append(trail)

            frame = self._o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=self.config.axis_size_mm,
                origin=(0.0, 0.0, 0.0),
            )
            frame.transform(pose_transform_mm(sample))
            self._visualizer.add_geometry(frame, reset_bounding_box=False)
            self._dynamic_geometries.append(frame)

            marker = self._o3d.geometry.TriangleMesh.create_sphere(
                radius=max(4.0, self.config.axis_size_mm * 0.11)
            )
            marker.compute_vertex_normals()
            marker.paint_uniform_color(color)
            marker.translate(np.asarray(sample.position_mm, dtype=np.float64))
            self._visualizer.add_geometry(marker, reset_bounding_box=False)
            self._dynamic_geometries.append(marker)

    def _render_if_dirty(self) -> None:
        if not self._dirty or not self._ensure_visualizer():
            return
        try:
            self._replace_dynamic_geometry()
            self._visualizer.poll_events()
            self._visualizer.update_renderer()
            image = np.asarray(self._visualizer.capture_screen_float_buffer(do_render=True))
            if image.ndim != 3 or image.shape[2] != 3:
                raise RuntimeError(f"Open3D 返回了不支持的图像形状 {image.shape}")
            image = np.ascontiguousarray(np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8))
            qimage = QImage(
                image.data,
                image.shape[1],
                image.shape[0],
                image.strides[0],
                QImage.Format_RGB888,
            ).copy()
            self._last_image = qimage
            self._set_qimage(qimage)
            trail_count = sum(len(points) for points in self._trails.values())
            self._summary_label.setText(f"{len(self._latest)} 工具 · {trail_count} 轨迹点")
            self._dirty = False
            self._render_failures = 0
        except Exception as exc:
            self._render_failures += 1
            self._render_label.setText(f"Open3D 渲染失败（{self._render_failures}/3）\n{type(exc).__name__}: {exc}")
            if self._render_failures >= 3:
                self._disabled_reason = f"{type(exc).__name__}: {exc}"
                self._summary_label.setText("Open3D 已暂停")
                self._timer.stop()

    def _set_qimage(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image).scaled(
            self._render_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._render_label.setPixmap(pixmap)
