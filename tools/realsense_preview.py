from __future__ import annotations

import argparse
import math
import os
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.check_realsense import capture_devices, realsense_capture_device, realsense_usb_devices, uvc_backend


# D435i 彩色相机标称水平 FOV，用于 UVC 模式下反推焦距。
D435I_RGB_HFOV_DEG = 69.4


def approximate_intrinsics(width: int, height: int, hfov_deg: float = D435I_RGB_HFOV_DEG) -> np.ndarray:
    """按标称 FOV 反推针孔内参。仅供 UVC 模式兜底，精度远不如出厂标定。"""
    focal = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def sdk_status() -> tuple[bool, str]:
    """pyrealsense2 是否真的能用。macOS 上装了也不代表非 root 能访问设备。"""
    try:
        import pyrealsense2  # noqa: F401
    except ImportError:
        return False, "未安装 pyrealsense2（macOS arm64 无官方 wheel，需本地编译 librealsense）"
    if sys.platform == "darwin" and os.geteuid() != 0:
        return False, "macOS 12+ 要求以 root 通过 libusb 访问设备，请用 sudo 运行"
    return True, ""


class FrameSource(ABC):
    display_name = "Camera"

    # start() 之后填充；UVC 模式没有出厂标定，只能按 FOV 近似。
    camera_matrix: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None
    intrinsics_exact: bool = False

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self) -> np.ndarray | None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError


class RealSenseSource(FrameSource):
    display_name = "RealSense SDK (pyrealsense2)"

    def __init__(self, width: int, height: int, fps: int, serial: str = "", with_depth: bool = True) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.with_depth = with_depth
        self._pipeline = None
        self._colorizer = None

    def start(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("未安装 pyrealsense2，无法使用 SDK 模式") from exc

        pipeline = rs.pipeline()
        stream_config = rs.config()
        if self.serial:
            stream_config.enable_device(self.serial)
        stream_config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if self.with_depth:
            stream_config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        try:
            profile = pipeline.start(stream_config)
        except RuntimeError as exc:
            if "power state" in str(exc).lower():
                raise RuntimeError(
                    f"{exc}；macOS 12+ 需以 root 访问 USB，请用 sudo 运行本程序"
                ) from exc
            raise

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()
        self.camera_matrix = np.array(
            [[intrinsics.fx, 0.0, intrinsics.ppx], [0.0, intrinsics.fy, intrinsics.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.dist_coeffs = np.asarray(intrinsics.coeffs, dtype=np.float64)
        self.intrinsics_exact = True
        self._pipeline = pipeline
        self._colorizer = rs.colorizer() if self.with_depth else None

    def read(self) -> np.ndarray | None:
        import cv2

        if self._pipeline is None:
            raise RuntimeError("RealSense 尚未启动")
        frames = self._pipeline.wait_for_frames(2000)
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        image_bgr = np.asanyarray(color_frame.get_data())
        if self._colorizer is not None:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_bgr = np.asanyarray(self._colorizer.colorize(depth_frame).get_data())
                image_bgr = np.hstack((image_bgr, depth_bgr))
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        self._colorizer = None


class UvcSource(FrameSource):
    display_name = "UVC 彩色流 (OpenCV)"

    def __init__(self, index: int, width: int, height: int, fps: int, hfov_deg: float = D435I_RGB_HFOV_DEG) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.hfov_deg = hfov_deg
        self._capture = None

    def preflight(self) -> None:
        """在 GUI 主线程试开一次：macOS 的摄像头授权请求只能从主 run loop 发起。"""
        import cv2

        capture = cv2.VideoCapture(self.index, uvc_backend())
        opened = capture.isOpened()
        capture.release()
        if not opened:
            if sys.platform == "darwin":
                hint = (
                    "通常是 macOS 摄像头权限未授予当前程序："
                    "系统设置 → 隐私与安全性 → 摄像头 中勾选运行本脚本的终端/IDE，然后完全退出并重开它。"
                )
            else:
                hint = "确认摄像头未被其他程序独占，或用 --index 换一个索引重试。"
            raise RuntimeError(f"无法打开摄像头索引 {self.index}。{hint}")

    def start(self) -> None:
        import cv2

        capture = cv2.VideoCapture(self.index, uvc_backend())
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"无法打开摄像头索引 {self.index}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        self._capture = capture

        actual_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH) or self.width
        actual_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.height
        self.camera_matrix = approximate_intrinsics(int(actual_width), int(actual_height), self.hfov_deg)
        self.dist_coeffs = np.zeros(5, dtype=np.float64)
        self.intrinsics_exact = False

    def read(self) -> np.ndarray | None:
        import cv2

        if self._capture is None:
            raise RuntimeError("摄像头尚未启动")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class PreviewWorker(QThread):
    frame_ready = pyqtSignal(object)
    connected = pyqtSignal(str)
    failed = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self, source: FrameSource, poll_hz: int, parent=None) -> None:
        super().__init__(parent)
        self.source = source
        self.poll_hz = poll_hz

    def run(self) -> None:
        started_source = False
        try:
            self.source.start()
            started_source = True
            self.connected.emit(self.source.display_name)
            period = 1.0 / self.poll_hz
            while not self.isInterruptionRequested():
                started = time.monotonic()
                frame = self.source.read()
                if frame is not None:
                    self.frame_ready.emit(frame)
                remaining = period - (time.monotonic() - started)
                if remaining > 0:
                    self.msleep(max(1, int(remaining * 1000)))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if started_source:
                try:
                    self.source.stop()
                except Exception as exc:
                    self.failed.emit(f"关闭设备失败: {exc}")
            self.stopped.emit()


class PreviewWindow(QMainWindow):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.worker: PreviewWorker | None = None
        self.latest_frame: np.ndarray | None = None
        self.frame_count = 0
        self.fps_started = time.monotonic()

        self.setWindowTitle("RealSense 影像预览")
        self.resize(1180, 760)

        self.source_box = QComboBox()
        self.source_box.addItem("自动选择", "auto")
        self.source_box.addItem("RealSense SDK", "sdk")
        self.source_box.addItem("UVC 彩色流", "uvc")

        self.camera_box = QComboBox()
        target = realsense_capture_device()
        for device in capture_devices():
            suffix = "  (RealSense)" if target is not None and device.index == target.index else ""
            self.camera_box.addItem(f"{device.index}: {device.name}{suffix}", device.index)
        if args.index is not None:
            position = self.camera_box.findData(args.index)
            if position >= 0:
                self.camera_box.setCurrentIndex(position)
        elif target is not None:
            position = self.camera_box.findData(target.index)
            if position >= 0:
                self.camera_box.setCurrentIndex(position)

        self.depth_check = QCheckBox("并排显示深度图 (仅 SDK)")
        self.depth_check.setChecked(True)

        self.start_button = QPushButton("开始")
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.snapshot_button = QPushButton("保存快照")
        self.snapshot_button.setEnabled(False)

        self.start_button.clicked.connect(self.start_preview)
        self.stop_button.clicked.connect(self.stop_preview)
        self.snapshot_button.clicked.connect(self.save_snapshot)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("数据源:"))
        controls.addWidget(self.source_box)
        controls.addWidget(QLabel("摄像头:"))
        controls.addWidget(self.camera_box)
        controls.addWidget(self.depth_check)
        controls.addStretch(1)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.snapshot_button)

        self.video_label = QLabel("点击 开始 拉取影像")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(960, 560)
        self.video_label.setStyleSheet("background:#101418; color:#8fa3b0; border-radius:6px;")

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(self.video_label, 1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.statusBar().showMessage(self._device_summary())

    def _device_summary(self) -> str:
        devices = realsense_usb_devices()
        if not devices:
            return "USB 上未发现 RealSense 相机"
        device = devices[0]
        return f"{device.name} | SN {device.serial} | {device.speed_label}"

    def _build_source(self) -> FrameSource:
        choice = self.source_box.currentData()
        if choice in {"auto", "sdk"}:
            usable, reason = sdk_status()
            if usable:
                return RealSenseSource(
                    self.args.width,
                    self.args.height,
                    self.args.fps,
                    self.args.serial,
                    self.depth_check.isChecked(),
                )
            if choice == "sdk":
                raise RuntimeError(reason)
            self.statusBar().showMessage(f"SDK 不可用({reason})，回退到 UVC 彩色流")

        index = self.camera_box.currentData()
        if index is None:
            raise RuntimeError("系统未列出任何摄像头，请用 --index 手动指定索引")
        return UvcSource(int(index), self.args.width, self.args.height, self.args.fps)

    def start_preview(self) -> None:
        try:
            source = self._build_source()
            if isinstance(source, UvcSource):
                source.preflight()
        except RuntimeError as exc:
            self.statusBar().showMessage(f"启动失败: {exc}")
            return

        self.frame_count = 0
        self.fps_started = time.monotonic()
        self.worker = PreviewWorker(source, self.args.poll_hz, self)
        self.worker.frame_ready.connect(self.show_frame)
        self.worker.connected.connect(lambda name: self.statusBar().showMessage(f"已连接: {name}"))
        self.worker.failed.connect(lambda message: self.statusBar().showMessage(f"错误: {message}"))
        self.worker.stopped.connect(self.on_stopped)
        self.worker.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.source_box.setEnabled(False)
        self.camera_box.setEnabled(False)
        self.depth_check.setEnabled(False)

    def stop_preview(self) -> None:
        if self.worker is not None:
            self.worker.requestInterruption()
            self.worker.wait(3000)

    def on_stopped(self) -> None:
        self.worker = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.source_box.setEnabled(True)
        self.camera_box.setEnabled(True)
        self.depth_check.setEnabled(True)

    def show_frame(self, frame: np.ndarray) -> None:
        self.latest_frame = frame
        self.snapshot_button.setEnabled(True)
        height, width, channels = frame.shape
        image = QImage(frame.data, width, height, channels * width, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

        self.frame_count += 1
        elapsed = time.monotonic() - self.fps_started
        if elapsed >= 1.0:
            self.statusBar().showMessage(f"{width}x{height} | {self.frame_count / elapsed:.1f} fps | {self._device_summary()}")
            self.frame_count = 0
            self.fps_started = time.monotonic()

    def save_snapshot(self) -> None:
        import cv2

        if self.latest_frame is None:
            return
        directory = Path(self.args.snapshot_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"realsense_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
        cv2.imwrite(str(path), cv2.cvtColor(self.latest_frame, cv2.COLOR_RGB2BGR))
        self.statusBar().showMessage(f"已保存 {path}")

    def closeEvent(self, event) -> None:
        self.stop_preview()
        super().closeEvent(event)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 PyQt5 预览 RealSense 影像")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--poll-hz", type=int, default=30)
    parser.add_argument("--serial", default="", help="指定 RealSense 序列号 (SDK 模式)")
    parser.add_argument("--index", type=int, default=None, help="手动指定 UVC 摄像头索引")
    parser.add_argument("--snapshot-dir", default="snapshots")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = QApplication(sys.argv[:1])
    window = PreviewWindow(args)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
