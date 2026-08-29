from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motion_capture.config import load_config
from tools.check_realsense import realsense_capture_device
from tools.realsense_preview import FrameSource, RealSenseSource, UvcSource, sdk_status

ARUCO_DICTIONARIES = {
    "tag36h11": "DICT_APRILTAG_36h11",
    "tag25h9": "DICT_APRILTAG_25h9",
    "tag16h5": "DICT_APRILTAG_16h5",
}


@dataclass(frozen=True)
class TagObservation:
    tag_id: int
    center_px: tuple[float, float]
    position_mm: tuple[float, float, float]
    distance_mm: float
    reprojection_error_px: float
    corners_px: tuple[tuple[float, float], ...] = ()

    def bounding_box(self) -> tuple[int, int, int, int]:
        """返回 (x_min, y_min, x_max, y_max)，标签据此画在 tag 外侧避免遮挡图案。"""
        if not self.corners_px:
            x, y = self.center_px
            return int(x), int(y), int(x), int(y)
        xs = [point[0] for point in self.corners_px]
        ys = [point[1] for point in self.corners_px]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    def format_line(self) -> str:
        x, y, z = self.position_mm
        return (
            f"    id={self.tag_id:>3}  px=({self.center_px[0]:6.0f},{self.center_px[1]:6.0f})"
            f"  xyz=({x:8.1f},{y:8.1f},{z:8.1f}) mm  距离={self.distance_mm:7.1f} mm"
            f"  重投影={self.reprojection_error_px:4.2f}px"
        )


def draw_label(canvas: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    """画带深色底衬的文字。tag 图案是高对比黑白块，纯文字压上去无法辨认。"""
    import cv2

    scale, thickness = 0.55, 2
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(2, min(int(origin[0]), canvas.shape[1] - width - 6))
    y = max(height + 6, min(int(origin[1]), canvas.shape[0] - baseline - 2))
    cv2.rectangle(canvas, (x - 4, y - height - 6), (x + width + 4, y + baseline + 2), (16, 20, 26), -1)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


@dataclass(frozen=True)
class TagStatistics:
    tag_id: int
    samples: int
    frames: int
    variance_mm2: tuple[float, float, float]
    sigma_mm: float
    mean_mm: tuple[float, float, float]

    @property
    def detection_rate(self) -> float:
        return self.samples / self.frames if self.frames else 0.0

    def format_lines(self) -> list[str]:
        vx, vy, vz = self.variance_mm2
        return [
            f"        {self.window_label} 方差 x={vx:7.3f} y={vy:7.3f} z={vz:7.3f} mm²"
            f"  σ={self.sigma_mm:6.3f} mm"
            f"  检出 {self.detection_rate * 100:5.1f}% ({self.samples}/{self.frames})"
        ]

    window_label = "5s"


class SlidingWindowStats:
    """维护每个 tag 在固定时间窗内的位置样本，用于算方差与检出率。"""

    def __init__(self, window_s: float = 5.0) -> None:
        self.window_s = float(window_s)
        self._frames: deque[float] = deque()
        self._history: dict[int, deque[tuple[float, np.ndarray]]] = {}

    def update(self, now: float, observations: list[TagObservation]) -> None:
        self._frames.append(now)
        for observation in observations:
            history = self._history.setdefault(observation.tag_id, deque())
            history.append((now, np.asarray(observation.position_mm, dtype=np.float64)))
        self._expire(now)

    def _expire(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._frames and self._frames[0] < cutoff:
            self._frames.popleft()
        for tag_id in list(self._history):
            history = self._history[tag_id]
            while history and history[0][0] < cutoff:
                history.popleft()
            if not history:
                del self._history[tag_id]

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def span_s(self) -> float:
        return self._frames[-1] - self._frames[0] if len(self._frames) > 1 else 0.0

    def summary(self, tag_id: int) -> TagStatistics | None:
        history = self._history.get(tag_id)
        if not history:
            return None
        positions = np.stack([position for _, position in history])
        # 少于两个样本时方差无定义，先按 0 上报，由样本数列体现可信度。
        if len(positions) < 2:
            variance = np.zeros(3)
        else:
            variance = positions.var(axis=0, ddof=1)
        return TagStatistics(
            tag_id=tag_id,
            samples=len(positions),
            frames=self.frame_count,
            variance_mm2=tuple(float(value) for value in variance),
            sigma_mm=float(np.sqrt(variance.sum())),
            mean_mm=tuple(float(value) for value in positions.mean(axis=0)),
        )

    def tracked_ids(self) -> list[int]:
        return sorted(self._history)


class AprilTagDetector:
    def __init__(self, family: str, tag_size_m: float, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
        import cv2

        key = family.strip().lower().replace("_", "")
        if key not in ARUCO_DICTIONARIES:
            raise ValueError(f"不支持的 AprilTag family: {family}")
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICTIONARIES[key]))
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        self.tag_size_m = float(tag_size_m)
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        # aruco 角点顺序为左上→右上→右下→左下，标记坐标系 X 右 / Y 上 / Z 朝外。
        half = self.tag_size_m / 2.0
        self._object_points = np.array(
            [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
            dtype=np.float32,
        )

    def detect(self, image_rgb: np.ndarray, annotate: bool = True) -> tuple[list[TagObservation], np.ndarray]:
        import cv2

        canvas = image_rgb.copy() if annotate else image_rgb
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        observations: list[TagObservation] = []
        if ids is None:
            return observations, canvas

        for marker_corners, marker_id in zip(corners, ids.flatten()):
            image_points = marker_corners.reshape(4, 2).astype(np.float32)
            # IPPE_SQUARE 是平面正方形标记的专用解法，比迭代法更稳。
            success, rvec, tvec = cv2.solvePnP(
                self._object_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not success:
                continue
            projected, _ = cv2.projectPoints(
                self._object_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
            )
            error = float(np.mean(np.linalg.norm(projected.reshape(4, 2) - image_points, axis=1)))
            position_mm = tuple(float(value) * 1000.0 for value in tvec.reshape(3))
            center = image_points.mean(axis=0)
            observations.append(
                TagObservation(
                    tag_id=int(marker_id),
                    center_px=(float(center[0]), float(center[1])),
                    position_mm=position_mm,
                    distance_mm=float(np.linalg.norm(position_mm)),
                    reprojection_error_px=error,
                    corners_px=tuple((float(px), float(py)) for px, py in image_points),
                )
            )

            if annotate:
                cv2.polylines(canvas, [image_points.astype(np.int32)], True, (80, 220, 160), 2)
                cv2.drawFrameAxes(
                    canvas, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.tag_size_m * 0.5
                )
                xs = image_points[:, 0]
                ys = image_points[:, 1]
                draw_label(
                    canvas,
                    f"id={int(marker_id)}  z={position_mm[2]:.0f}mm",
                    (int(xs.min()), int(ys.min()) - 8),
                    (120, 230, 170),
                )

        observations.sort(key=lambda item: item.tag_id)
        return observations, canvas


def annotate_statistics(canvas: np.ndarray, observations: list[TagObservation], stats: SlidingWindowStats) -> None:
    """把滑动窗口统计画到影像上。cv2.putText 不支持中文，这里一律用英文简写。"""
    import cv2

    lines = [f"{stats.window_s:.0f}s window | {stats.frame_count} frames | {stats.span_s:.1f}s"]
    for tag_id in stats.tracked_ids():
        summary = stats.summary(tag_id)
        if summary is None:
            continue
        vx, vy, vz = summary.variance_mm2
        lines.append(
            f"id {tag_id}: sd {summary.sigma_mm:6.3f}mm  var x{vx:6.2f} y{vy:6.2f} z{vz:6.2f}"
            f"  det {summary.detection_rate * 100:3.0f}%"
        )

    padding = 10
    line_height = 24
    box_height = padding * 2 + line_height * len(lines)
    box_width = 640
    overlay_layer = canvas.copy()
    cv2.rectangle(overlay_layer, (12, 12), (12 + box_width, 12 + box_height), (18, 22, 28), -1)
    cv2.addWeighted(overlay_layer, 0.65, canvas, 0.35, 0, canvas)

    for index, line in enumerate(lines):
        color = (150, 200, 230) if index == 0 else (120, 230, 170)
        cv2.putText(
            canvas,
            line,
            (12 + padding, 12 + padding + line_height * (index + 1) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )

    for observation in observations:
        summary = stats.summary(observation.tag_id)
        if summary is None:
            continue
        x_min, _, _, y_max = observation.bounding_box()
        draw_label(canvas, f"sd {summary.sigma_mm:.2f}mm", (x_min, y_max + 26), (150, 200, 230))


def build_source(args: argparse.Namespace) -> tuple[FrameSource, str]:
    usable, reason = sdk_status()
    if args.source == "sdk" and not usable:
        raise RuntimeError(reason)
    if args.source in {"auto", "sdk"} and usable:
        return RealSenseSource(args.width, args.height, args.fps, args.serial, with_depth=False), ""

    if args.index is not None:
        index = args.index
    else:
        device = realsense_capture_device()
        if device is None:
            raise RuntimeError("未找到 RealSense 的 UVC 彩色流，请用 --index 指定摄像头索引")
        index = device.index
    note = "" if args.source == "uvc" else f"SDK 不可用({reason})，改用 UVC 彩色流"
    return UvcSource(index, args.width, args.height, args.fps), note


def print_observations(
    frame_index: int,
    elapsed: float,
    observations: list[TagObservation],
    known_ids: set[int],
    stats: SlidingWindowStats | None = None,
) -> None:
    window_note = ""
    if stats is not None:
        window_note = f" | {stats.window_s:.0f}s 窗口: {stats.frame_count} 帧 / {stats.span_s:.1f}s"
    if not observations:
        print(f"[{elapsed:7.1f}s] frame {frame_index:>6} | 未检测到 AprilTag{window_note}", flush=True)
        return
    print(f"[{elapsed:7.1f}s] frame {frame_index:>6} | {len(observations)} 个 tag{window_note}", flush=True)
    for observation in observations:
        suffix = "" if observation.tag_id in known_ids else "   <- 不在 APRILTAG_IDS 配置中"
        print(observation.format_line() + suffix, flush=True)
        if stats is not None:
            summary = stats.summary(observation.tag_id)
            if summary is not None:
                for line in summary.format_lines():
                    print(line, flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config = load_config(Path(__file__).resolve().parent.parent)
    parser = argparse.ArgumentParser(description="实时检测 AprilTag 并打印 id 与位置")
    parser.add_argument("--source", choices=["auto", "sdk", "uvc"], default="auto")
    parser.add_argument("--index", type=int, default=None, help="UVC 摄像头索引")
    parser.add_argument("--serial", default=config.realsense.serial)
    parser.add_argument("--width", type=int, default=config.realsense.width)
    parser.add_argument("--height", type=int, default=config.realsense.height)
    parser.add_argument("--fps", type=int, default=config.realsense.fps)
    parser.add_argument("--family", default=config.realsense.tag_family)
    parser.add_argument("--tag-size-m", type=float, default=config.realsense.tag_size_m)
    parser.add_argument("--print-hz", type=float, default=5.0, help="终端打印频率，避免刷屏")
    parser.add_argument("--window-s", type=float, default=5.0, help="方差统计的滑动窗口长度（秒）")
    parser.add_argument("--window", action="store_true", help="同时开 PyQt5 窗口显示叠加画面")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    args.known_ids = set(config.realsense.tag_ids)
    return args


def run_headless(args: argparse.Namespace) -> int:
    source, note = build_source(args)
    if note:
        print(f"提示: {note}", flush=True)
    source.start()
    try:
        detector = AprilTagDetector(args.family, args.tag_size_m, source.camera_matrix, source.dist_coeffs)
        print(f"数据源: {source.display_name} | family={args.family} | tag={args.tag_size_m * 1000:.0f}mm")
        print(f"配置中的 tag id: {sorted(args.known_ids)}")
        if not source.intrinsics_exact:
            print("警告: 使用按 FOV 近似的内参，深度方向误差可达 10% 以上；要精确值请用 sudo 走 SDK 模式。")
        print("按 Ctrl+C 停止\n")

        stats = SlidingWindowStats(args.window_s)
        started = time.monotonic()
        interval = 1.0 / args.print_hz if args.print_hz > 0 else 0.0
        last_print = 0.0
        frame_index = 0
        while True:
            frame = source.read()
            if frame is None:
                continue
            frame_index += 1
            now = time.monotonic()
            # 每帧都要检测，否则滑动窗口样本不足，方差失真；限流只作用于打印。
            observations, _ = detector.detect(frame, annotate=False)
            stats.update(now, observations)
            if now - last_print < interval:
                continue
            last_print = now
            print_observations(frame_index, now - started, observations, args.known_ids, stats)
    except KeyboardInterrupt:
        print("\n已停止", flush=True)
        return 0
    finally:
        source.stop()


def run_window(args: argparse.Namespace) -> int:
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QWidget

    @dataclass(frozen=True)
    class FrameResult:
        observations: list[TagObservation]
        canvas: np.ndarray
        summaries: list[TagStatistics]
        frame_index: int
        elapsed: float
        frame_count: int
        span_s: float

    class DetectWorker(QThread):
        result_ready = pyqtSignal(object)
        failed = pyqtSignal(str)

        def __init__(self, source: FrameSource, family: str, tag_size_m: float, window_s: float, parent=None) -> None:
            super().__init__(parent)
            self.source = source
            self.family = family
            self.tag_size_m = tag_size_m
            self.window_s = window_s

        def run(self) -> None:
            try:
                detector = AprilTagDetector(
                    self.family, self.tag_size_m, self.source.camera_matrix, self.source.dist_coeffs
                )
                # 统计全程在本线程内完成，避免与 GUI 线程竞争。
                stats = SlidingWindowStats(self.window_s)
                started = time.monotonic()
                frame_index = 0
                while not self.isInterruptionRequested():
                    frame = self.source.read()
                    if frame is None:
                        continue
                    frame_index += 1
                    now = time.monotonic()
                    observations, canvas = detector.detect(frame)
                    stats.update(now, observations)
                    annotate_statistics(canvas, observations, stats)
                    summaries = [
                        summary
                        for summary in (stats.summary(tag_id) for tag_id in stats.tracked_ids())
                        if summary is not None
                    ]
                    self.result_ready.emit(
                        FrameResult(
                            observations=observations,
                            canvas=canvas,
                            summaries=summaries,
                            frame_index=frame_index,
                            elapsed=now - started,
                            frame_count=stats.frame_count,
                            span_s=stats.span_s,
                        )
                    )
            except Exception as exc:
                self.failed.emit(f"{type(exc).__name__}: {exc}")

    source, note = build_source(args)
    if note:
        print(f"提示: {note}", flush=True)
    source.start()
    if not source.intrinsics_exact:
        print("警告: 使用按 FOV 近似的内参，深度方向误差可达 10% 以上；要精确值请用 sudo 走 SDK 模式。")

    app = QApplication(sys.argv[:1])
    window = QMainWindow()
    window.setWindowTitle("AprilTag 实时检测与稳定性统计")
    window.resize(1420, 780)

    video_label = QLabel("等待画面")
    video_label.setAlignment(Qt.AlignCenter)
    video_label.setMinimumWidth(820)
    video_label.setStyleSheet("background:#101418; color:#8fa3b0; border-radius:6px;")

    stats_label = QLabel("等待统计")
    stats_label.setAlignment(Qt.AlignTop)
    stats_label.setFixedWidth(360)
    stats_label.setTextFormat(Qt.RichText)
    stats_label.setStyleSheet("background:#141a20; color:#c8d6e0; border-radius:6px; padding:12px;")

    layout = QHBoxLayout()
    layout.addWidget(video_label, 1)
    layout.addWidget(stats_label)
    container = QWidget()
    container.setLayout(layout)
    window.setCentralWidget(container)

    interval = 1.0 / args.print_hz if args.print_hz > 0 else 0.0
    state = {"last_print": 0.0}

    def render_stats(result) -> str:
        rows = [
            f"<div style='font-size:13px;color:#8fa3b0'>{args.window_s:.0f} 秒滑动窗口<br>"
            f"{result.frame_count} 帧 / {result.span_s:.1f} 秒</div><br>"
        ]
        if not result.summaries:
            rows.append("<div style='color:#d08f6a'>窗口内没有检测到任何 tag</div>")
        for summary in result.summaries:
            vx, vy, vz = summary.variance_mm2
            mx, my, mz = summary.mean_mm
            known = "" if summary.tag_id in args.known_ids else " <span style='color:#d08f6a'>(未配置)</span>"
            rows.append(
                f"<div style='margin-bottom:14px'>"
                f"<b style='color:#78e6aa;font-size:15px'>id {summary.tag_id}</b>{known}<br>"
                f"<span style='font-family:monospace;font-size:12px'>"
                f"方差 x {vx:8.3f}<br>"
                f"&nbsp;&nbsp;&nbsp;&nbsp; y {vy:8.3f}<br>"
                f"&nbsp;&nbsp;&nbsp;&nbsp; z {vz:8.3f} mm²<br>"
                f"标准差 σ = {summary.sigma_mm:.3f} mm<br>"
                f"均值 ({mx:.1f}, {my:.1f}, {mz:.1f}) mm<br>"
                f"检出 {summary.detection_rate * 100:.1f}% ({summary.samples}/{summary.frames})"
                f"</span></div>"
            )
        return "".join(rows)

    def on_result(result) -> None:
        canvas = result.canvas
        height, width, channels = canvas.shape
        image = QImage(canvas.data, width, height, channels * width, QImage.Format_RGB888).copy()
        video_label.setPixmap(
            QPixmap.fromImage(image).scaled(video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        stats_label.setText(render_stats(result))
        window.statusBar().showMessage(
            f"{len(result.observations)} 个 tag | frame {result.frame_index} | {width}x{height}"
        )
        now = time.monotonic()
        if now - state["last_print"] >= interval:
            state["last_print"] = now
            print_observations(result.frame_index, result.elapsed, result.observations, args.known_ids)
            for summary in result.summaries:
                for line in summary.format_lines():
                    print(line, flush=True)

    worker = DetectWorker(source, args.family, args.tag_size_m, args.window_s)
    worker.result_ready.connect(on_result)
    worker.failed.connect(lambda message: window.statusBar().showMessage(f"错误: {message}"))
    worker.start()

    window.show()
    code = app.exec_()
    worker.requestInterruption()
    worker.wait(3000)
    source.stop()
    return code


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_window(args) if args.window else run_headless(args)
    except RuntimeError as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
