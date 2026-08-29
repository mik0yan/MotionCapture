from __future__ import annotations

import time

import numpy as np

from motion_capture.backends.base import TrackerBackend
from motion_capture.config import RealSenseConfig
from motion_capture.models import PoseSample, TrackingPacket


class RealSenseAprilTagBackend(TrackerBackend):
    display_name = "RealSense / AprilTag"

    def __init__(self, config: RealSenseConfig) -> None:
        self.config = config
        self._pipeline = None
        self._detector = None
        self._camera_matrix = None
        self._distortion = None
        self._frame = 0

    def start(self) -> None:
        try:
            import cv2
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("RealSense 模式需要安装 pyrealsense2 和 opencv-contrib-python") from exc

        dictionary_name = self.config.tag_family.lower().replace("_", "")
        dictionaries = {
            "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
            "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
            "tagcircle21h7": cv2.aruco.DICT_APRILTAG_21h7,
        }
        if dictionary_name not in dictionaries:
            raise ValueError(f"不支持的 AprilTag family: {self.config.tag_family}")
        dictionary = cv2.aruco.getPredefinedDictionary(dictionaries[dictionary_name])
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        pipeline = rs.pipeline()
        stream_config = rs.config()
        if self.config.serial:
            stream_config.enable_device(self.config.serial)
        stream_config.enable_stream(rs.stream.color, self.config.width, self.config.height, rs.format.bgr8, self.config.fps)
        profile = pipeline.start(stream_config)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()
        self._camera_matrix = np.array([[intrinsics.fx, 0.0, intrinsics.ppx], [0.0, intrinsics.fy, intrinsics.ppy], [0.0, 0.0, 1.0]], dtype=np.float64)
        self._distortion = np.asarray(intrinsics.coeffs, dtype=np.float64)
        self._pipeline = pipeline

    def _board_points(self, tag_id: int) -> np.ndarray:
        index = self.config.tag_ids.index(tag_id)
        row, col = divmod(index, self.config.board_cols)
        step = self.config.tag_size_m + self.config.tag_spacing_m
        center_x = (col - (self.config.board_cols - 1) / 2.0) * step
        center_y = (row - (self.config.board_rows - 1) / 2.0) * step
        half = self.config.tag_size_m / 2.0
        return np.array([
            [center_x - half, center_y - half, 0.0],
            [center_x + half, center_y - half, 0.0],
            [center_x + half, center_y + half, 0.0],
            [center_x - half, center_y + half, 0.0],
        ], dtype=np.float32)

    def read(self) -> TrackingPacket:
        import cv2

        if self._pipeline is None:
            raise RuntimeError("RealSense 尚未启动")
        frames = self._pipeline.wait_for_frames(1000)
        color_frame = frames.get_color_frame()
        if not color_frame:
            return TrackingPacket(())
        image_bgr = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        self._frame = int(color_frame.get_frame_number())
        samples: tuple[PoseSample, ...] = ()

        if ids is not None:
            board_object_points: list[np.ndarray] = []
            board_image_points: list[np.ndarray] = []
            visible_ids: list[int] = []
            for marker_corners, marker_id_array in zip(corners, ids):
                marker_id = int(marker_id_array[0])
                if marker_id not in self.config.tag_ids:
                    continue
                visible_ids.append(marker_id)
                board_object_points.append(self._board_points(marker_id))
                board_image_points.append(marker_corners.reshape(4, 2).astype(np.float32))
                cv2.polylines(image_bgr, [marker_corners.astype(np.int32)], True, (80, 220, 160), 2)
                center = marker_corners.reshape(4, 2).mean(axis=0).astype(int)
                cv2.putText(image_bgr, str(marker_id), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 160), 2)

            visible_count = len(visible_ids)
            if visible_count >= self.config.min_visible_tags:
                object_points = np.concatenate(board_object_points)
                image_points = np.concatenate(board_image_points)
                success, rvec, tvec = cv2.solvePnP(object_points, image_points, self._camera_matrix, self._distortion, flags=cv2.SOLVEPNP_ITERATIVE)
                if success:
                    rotation, _ = cv2.Rodrigues(rvec)
                    position = tuple(float(value) for value in tvec.reshape(3) * 1000.0)
                    quality = visible_count / len(self.config.tag_ids)
                    samples = (PoseSample("realsense", "apriltag_board", self._frame, position, rotation, quality),)
                    cv2.drawFrameAxes(image_bgr, self._camera_matrix, self._distortion, rvec, tvec, self.config.tag_size_m)
            status_color = (80, 220, 160) if samples else (80, 180, 255)
            cv2.putText(
                image_bgr,
                f"Board: {visible_count}/{len(self.config.tag_ids)} tags (need {self.config.min_visible_tags})",
                (22, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
            )
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return TrackingPacket(samples, image_rgb)

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
