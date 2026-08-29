from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import numpy as np

from motion_capture.config import FFmpegConfig


@dataclass(frozen=True)
class VideoRecordingResult:
    path: Path
    metadata_path: Path
    log_path: Path
    width: int
    height: int
    fps: float
    frames_written: int
    frames_dropped: int

    @property
    def duration_seconds(self) -> float:
        return self.frames_written / self.fps


class FFmpegVideoRecorder:
    """Encode RGB frames to a local MP4 file through an FFmpeg subprocess."""

    _STOP = object()

    def __init__(
        self,
        directory: Path,
        config: FFmpegConfig,
        *,
        queue_size: int = 120,
        stop_timeout: float = 15.0,
    ) -> None:
        self.directory = Path(directory)
        self.config = config
        self.queue_size = max(1, int(queue_size))
        self.stop_timeout = max(1.0, float(stop_timeout))
        self.path: Path | None = None
        self.partial_path: Path | None = None
        self.log_path: Path | None = None
        self.metadata_path: Path | None = None
        self.last_result: VideoRecordingResult | None = None
        self._width = 0
        self._height = 0
        self._fps = 0.0
        self._started_at: datetime | None = None
        self._frames_written = 0
        self._frames_dropped = 0
        self._process: subprocess.Popen | None = None
        self._stdin: BinaryIO | None = None
        self._log_file: BinaryIO | None = None
        self._queue: queue.Queue[bytes | object] | None = None
        self._thread: threading.Thread | None = None
        self._writer_error: BaseException | None = None

    @property
    def active(self) -> bool:
        return self._process is not None

    @property
    def frames_dropped(self) -> int:
        return self._frames_dropped

    def start(self, width: int, height: int, fps: float, stem: str) -> Path:
        if self.active:
            raise RuntimeError("FFmpeg 视频录制已经开始")
        if int(width) <= 0 or int(height) <= 0:
            raise ValueError("视频分辨率必须大于 0")
        if float(fps) <= 0:
            raise ValueError("视频帧率必须大于 0")
        executable = self._resolve_executable()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"{stem}.mp4"
        self.partial_path = self.directory / f".{stem}.partial.mp4"
        self.log_path = self.directory / f"{stem}.ffmpeg.log"
        self.metadata_path = self.directory / f"{stem}.json"
        if self.path.exists() or self.partial_path.exists():
            raise FileExistsError(f"录制文件已存在：{self.path}")

        self._width = int(width)
        self._height = int(height)
        self._fps = float(fps)
        self._started_at = datetime.now(timezone.utc)
        self._frames_written = 0
        self._frames_dropped = 0
        self._writer_error = None
        self.last_result = None
        self._queue = queue.Queue(maxsize=self.queue_size)
        self._log_file = self.log_path.open("wb")
        command = self._command(executable)
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._log_file,
                shell=False,
            )
        except Exception:
            self._log_file.close()
            self._log_file = None
            self._reset_runtime()
            raise
        if self._process.stdin is None:
            self._terminate_process()
            raise RuntimeError("FFmpeg 未提供标准输入管道")
        self._stdin = self._process.stdin
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="ffmpeg-video-writer",
            daemon=True,
        )
        self._thread.start()
        return self.path

    def write_frame(self, image_rgb: np.ndarray) -> bool:
        if not self.active or self._queue is None:
            return False
        if self._writer_error is not None:
            raise RuntimeError(f"FFmpeg 写入失败：{self._writer_error}") from self._writer_error
        image = np.asarray(image_rgb)
        expected = (self._height, self._width, 3)
        if image.shape != expected:
            raise ValueError(f"视频帧尺寸必须为 {expected}，当前为 {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"视频帧类型必须为 uint8，当前为 {image.dtype}")
        payload = np.ascontiguousarray(image).tobytes()
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            self._frames_dropped += 1
            return False
        return True

    def stop(self) -> VideoRecordingResult | None:
        if not self.active:
            return self.last_result
        process = self._process
        writer = self._thread
        frame_queue = self._queue
        if frame_queue is not None and writer is not None and writer.is_alive():
            try:
                frame_queue.put(self._STOP, timeout=self.stop_timeout)
            except queue.Full as exc:
                self._terminate_process()
                raise RuntimeError("FFmpeg 视频队列无法在停止前排空") from exc
        if writer is not None:
            writer.join(self.stop_timeout)
        if writer is not None and writer.is_alive():
            self._terminate_process()
            raise RuntimeError("FFmpeg 视频写入线程停止超时")

        try:
            return_code = process.wait(timeout=self.stop_timeout)
        except subprocess.TimeoutExpired as exc:
            self._terminate_process()
            raise RuntimeError("FFmpeg 进程停止超时") from exc
        self._close_log()
        writer_error = self._writer_error
        if writer_error is not None or return_code != 0:
            detail = self._log_tail()
            self._reset_runtime()
            message = f"FFmpeg 录制失败，退出码 {return_code}"
            if writer_error is not None:
                message += f"：{writer_error}"
            if detail:
                message += f"\n{detail}"
            raise RuntimeError(message)

        if self.partial_path is None or self.path is None or self.metadata_path is None or self.log_path is None:
            self._reset_runtime()
            raise RuntimeError("FFmpeg 录制路径状态不完整")
        if not self.partial_path.exists() or self.partial_path.stat().st_size == 0:
            self._reset_runtime()
            raise RuntimeError("FFmpeg 未生成有效 MP4 文件")
        self.partial_path.replace(self.path)
        result = VideoRecordingResult(
            path=self.path,
            metadata_path=self.metadata_path,
            log_path=self.log_path,
            width=self._width,
            height=self._height,
            fps=self._fps,
            frames_written=self._frames_written,
            frames_dropped=self._frames_dropped,
        )
        self.last_result = result
        self._reset_runtime()
        self._write_metadata(result)
        return result

    def _resolve_executable(self) -> str:
        configured = self.config.executable.strip()
        if not configured:
            raise RuntimeError("FFMPEG_PATH 不能为空")
        candidate = shutil.which(configured)
        if candidate is None and Path(configured).expanduser().is_file():
            candidate = str(Path(configured).expanduser().resolve())
        if candidate is None:
            raise RuntimeError(f"未找到本地 FFmpeg：{configured}")
        return candidate

    def _command(self, executable: str) -> list[str]:
        if self.partial_path is None:
            raise RuntimeError("FFmpeg 临时输出路径尚未初始化")
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{self._width}x{self._height}",
            "-framerate",
            f"{self._fps:g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            self.config.codec,
        ]
        if self.config.codec == "libx264":
            command.extend(["-preset", self.config.preset, "-crf", str(self.config.crf)])
        command.extend(
            [
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.partial_path),
            ]
        )
        return command

    def _writer_loop(self) -> None:
        try:
            while self._queue is not None:
                payload = self._queue.get()
                if payload is self._STOP:
                    break
                if self._stdin is None:
                    raise RuntimeError("FFmpeg 标准输入已关闭")
                self._stdin.write(payload)
                self._frames_written += 1
        except (BrokenPipeError, OSError, RuntimeError) as exc:
            self._writer_error = exc
        finally:
            if self._stdin is not None:
                try:
                    self._stdin.close()
                except OSError:
                    pass
                self._stdin = None

    def _terminate_process(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        self._close_log()
        self._reset_runtime()

    def _close_log(self) -> None:
        if self._log_file is not None:
            self._log_file.flush()
            self._log_file.close()
            self._log_file = None

    def _log_tail(self, max_bytes: int = 4096) -> str:
        if self.log_path is None or not self.log_path.exists():
            return ""
        with self.log_path.open("rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - max_bytes))
            return handle.read().decode("utf-8", errors="replace").strip()

    def _write_metadata(self, result: VideoRecordingResult) -> None:
        ended_at = datetime.now(timezone.utc)
        payload = {
            "schema_version": 1,
            "video_path": result.path.name,
            "started_at_utc": None if self._started_at is None else self._started_at.isoformat(),
            "ended_at_utc": ended_at.isoformat(),
            "width": result.width,
            "height": result.height,
            "fps": result.fps,
            "codec": self.config.codec,
            "frames_written": result.frames_written,
            "frames_dropped": result.frames_dropped,
            "duration_seconds": result.duration_seconds,
        }
        result.metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _reset_runtime(self) -> None:
        self._process = None
        self._stdin = None
        self._queue = None
        self._thread = None
        self._writer_error = None
