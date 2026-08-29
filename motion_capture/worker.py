from __future__ import annotations

import time
from collections.abc import Mapping

from PyQt5.QtCore import QThread, pyqtSignal

from motion_capture.backends import create_backend
from motion_capture.config import AppConfig


class TrackingWorker(QThread):
    packet_ready = pyqtSignal(object)
    connected = pyqtSignal(str)
    failed = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(
        self,
        source: str,
        config: AppConfig,
        parent=None,
        tag_sizes_mm: Mapping[int, float] | None = None,
    ) -> None:
        super().__init__(parent)
        self.source = source
        self.config = config
        self.tag_sizes_mm = dict(tag_sizes_mm or {})

    def set_tag_size(self, tag_id: int, size_mm: float) -> None:
        self.tag_sizes_mm[int(tag_id)] = float(size_mm)

    def run(self) -> None:
        backend = None
        try:
            backend = create_backend(self.source, self.config, self.tag_sizes_mm)
            backend.start()
            self.connected.emit(backend.display_name)
            period = 1.0 / self.config.poll_hz
            while not self.isInterruptionRequested():
                started = time.monotonic()
                packet = backend.read()
                self.packet_ready.emit(packet)
                remaining = period - (time.monotonic() - started)
                if remaining > 0:
                    self.msleep(max(1, int(remaining * 1000)))
        except Exception as exc:  # Surface hardware/SDK errors in the UI.
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if backend is not None:
                try:
                    backend.stop()
                except Exception as exc:
                    self.failed.emit(f"停止设备失败: {exc}")
            self.stopped.emit()
