import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from motion_capture.config import FFmpegConfig
from motion_capture.video_recorder import FFmpegVideoRecorder


def _fake_ffmpeg(tmp_path: Path) -> Path:
    payload_code = (
        "import pathlib, sys\n"
        "payload = sys.stdin.buffer.read()\n"
        "pathlib.Path(sys.argv[-1]).write_bytes(payload)\n"
    )
    if os.name == "nt":
        # Windows 无法直接执行 shebang 脚本，用 cmd 批处理转发给当前解释器；
        # Popen 可直接执行 .bat，stdin 管道会透传给 python 子进程。
        script = tmp_path / "fake_ffmpeg.py"
        script.write_text(payload_code, encoding="utf-8")
        wrapper = tmp_path / "fake_ffmpeg.bat"
        wrapper.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
        return wrapper
    executable = tmp_path / "fake_ffmpeg.py"
    executable.write_text("#!/usr/bin/env python3\n" + payload_code, encoding="utf-8")
    executable.chmod(0o755)
    return executable


def _config(executable: Path | str) -> FFmpegConfig:
    return FFmpegConfig(True, str(executable), "libx264", "veryfast", 18)


def test_ffmpeg_recorder_streams_rgb_frames_and_writes_metadata(tmp_path: Path) -> None:
    recorder = FFmpegVideoRecorder(tmp_path / "recordings", _config(_fake_ffmpeg(tmp_path)))
    frame = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)

    path = recorder.start(6, 4, 20, "capture_test")
    assert recorder.write_frame(frame) is True
    assert recorder.write_frame(frame) is True
    result = recorder.stop()

    assert result is not None
    assert path == tmp_path / "recordings/capture_test.mp4"
    assert result.path.read_bytes() == frame.tobytes() * 2
    assert result.frames_written == 2
    assert result.frames_dropped == 0
    assert result.duration_seconds == pytest.approx(0.1)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["video_path"] == "capture_test.mp4"
    assert metadata["width"] == 6
    assert metadata["height"] == 4
    assert metadata["fps"] == 20
    assert metadata["frames_written"] == 2
    assert metadata["frames_dropped"] == 0
    assert recorder.active is False


def test_ffmpeg_recorder_rejects_wrong_frame_shape(tmp_path: Path) -> None:
    recorder = FFmpegVideoRecorder(tmp_path, _config(_fake_ffmpeg(tmp_path)))
    recorder.start(6, 4, 30, "capture_shape")
    try:
        with pytest.raises(ValueError, match="视频帧尺寸"):
            recorder.write_frame(np.zeros((5, 6, 3), dtype=np.uint8))
    finally:
        recorder.write_frame(np.zeros((4, 6, 3), dtype=np.uint8))
        recorder.stop()


def test_ffmpeg_recorder_requires_local_executable(tmp_path: Path) -> None:
    recorder = FFmpegVideoRecorder(tmp_path, _config("ffmpeg-that-does-not-exist"))
    with pytest.raises(RuntimeError, match="未找到本地 FFmpeg"):
        recorder.start(6, 4, 30, "capture_missing")

