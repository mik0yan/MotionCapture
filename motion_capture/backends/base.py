from __future__ import annotations

from abc import ABC, abstractmethod

from motion_capture.models import TrackingPacket


class TrackerBackend(ABC):
    display_name = "Tracker"

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self) -> TrackingPacket:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError
