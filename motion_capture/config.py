from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from dotenv import dotenv_values, set_key


def _integer(values: Mapping[str, str], key: str, default: int) -> int:
    raw = values.get(key, str(default))
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} 必须是整数，当前值为 {raw!r}") from exc


def _floating(values: Mapping[str, str], key: str, default: float) -> float:
    raw = values.get(key, str(default))
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} 必须是数字，当前值为 {raw!r}") from exc


def _csv(values: Mapping[str, str], key: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in values.get(key, "").split(",") if part.strip())


def _boolean(values: Mapping[str, str], key: str, default: bool) -> bool:
    raw = values.get(key, "true" if default else "false")
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} 必须是 true/false，当前值为 {raw!r}")


@dataclass(frozen=True)
class RealSenseConfig:
    serial: str
    width: int
    height: int
    fps: int
    tag_family: str
    tag_ids: tuple[int, ...]
    board_rows: int
    board_cols: int
    tag_size_m: float
    tag_spacing_m: float
    min_visible_tags: int
    record_format: str
    reference_transform: tuple[float, ...]
    calibration_samples: int
    calibration_max_std_mm: float
    calibration_max_angle_deg: float


@dataclass(frozen=True)
class NDIConfig:
    tracker_type: str
    ip_address: str
    port: int
    serial_port: str
    rom_files: tuple[Path, ...]
    record_orientation: str
    record_format: str


@dataclass(frozen=True)
class Open3DConfig:
    enabled: bool
    render_hz: int
    width: int
    height: int
    trail_points: int
    axis_size_mm: float
    camera_fov_deg: float


@dataclass(frozen=True)
class FFmpegConfig:
    enabled: bool
    executable: str
    codec: str
    preset: str
    crf: int


@dataclass(frozen=True)
class AppConfig:
    root: Path
    source: str
    poll_hz: int
    record_dir: Path
    realsense: RealSenseConfig
    ndi: NDIConfig
    open3d: Open3DConfig
    ffmpeg: FFmpegConfig
    database_path: Path


def save_source_mode(root: Path, source: str) -> Path:
    """Persist the selected application work mode to root .env."""
    normalized = source.strip().lower()
    if normalized not in {"simulator", "realsense", "ndi"}:
        raise ValueError("工作模式仅支持 simulator、realsense 或 ndi")
    env_path = root.resolve() / ".env"
    env_path.touch(exist_ok=True)
    set_key(str(env_path), "MOCAP_SOURCE", normalized, quote_mode="auto")
    return env_path


def build_realsense_config(
    serial: str,
    width: int,
    height: int,
    fps: int,
    tag_family: str,
    tag_ids: Iterable[int | str],
    board_rows: int,
    board_cols: int,
    tag_size_m: float,
    tag_spacing_m: float,
    min_visible_tags: int = 1,
    record_format: str = "xlsx",
    reference_transform: Iterable[float] | None = None,
    calibration_samples: int = 30,
    calibration_max_std_mm: float = 2.0,
    calibration_max_angle_deg: float = 1.0,
) -> RealSenseConfig:
    """Validate RealSense stream, AprilTag board, and zero-reference settings."""
    normalized_family = tag_family.strip().lower().replace("_", "")
    supported_families = {"tag36h11", "tag25h9", "tag16h5"}
    if normalized_family not in supported_families:
        raise ValueError("AprilTag family 仅支持 tag36h11、tag25h9 或 tag16h5")
    if not 320 <= int(width) <= 4096 or not 240 <= int(height) <= 2160:
        raise ValueError("RealSense 分辨率必须在 320×240 到 4096×2160 范围内")
    if not 1 <= int(fps) <= 120:
        raise ValueError("RealSense FPS 必须在 1 到 120 之间")

    try:
        normalized_ids = tuple(int(tag_id) for tag_id in tag_ids)
    except (TypeError, ValueError) as exc:
        raise ValueError("APRILTAG_IDS 必须是逗号分隔的整数") from exc
    if not normalized_ids or any(tag_id < 0 for tag_id in normalized_ids):
        raise ValueError("APRILTAG_IDS 至少包含一个非负整数")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("APRILTAG_IDS 不能包含重复编号")
    if int(board_rows) < 1 or int(board_cols) < 1:
        raise ValueError("AprilTag 标定板行列数必须大于 0")
    if int(board_rows) * int(board_cols) != len(normalized_ids):
        raise ValueError("APRILTAG_IDS 数量必须等于 APRILTAG_BOARD_ROWS × APRILTAG_BOARD_COLS")
    if float(tag_size_m) <= 0 or float(tag_spacing_m) < 0:
        raise ValueError("APRILTAG_SIZE_M 必须大于 0，APRILTAG_SPACING_M 不能小于 0")
    if not 1 <= int(min_visible_tags) <= len(normalized_ids):
        raise ValueError("APRILTAG_MIN_VISIBLE_TAGS 必须在 1 到标定板 Tag 总数之间")
    normalized_record_format = record_format.strip().lower()
    if normalized_record_format not in {"csv", "xlsx"}:
        raise ValueError("RealSense 位置文件格式仅支持 csv 或 xlsx")

    raw_reference = np.eye(4).reshape(-1) if reference_transform is None else reference_transform
    reference = tuple(float(value) for value in raw_reference)
    if len(reference) != 16 or not np.all(np.isfinite(reference)):
        raise ValueError("REALSENSE_REFERENCE_TRANSFORM 必须包含 16 个有限数字")
    matrix = np.asarray(reference, dtype=np.float64).reshape(4, 4)
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError("REALSENSE_REFERENCE_TRANSFORM 最后一行必须是 0,0,0,1")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-4
    ):
        raise ValueError("REALSENSE_REFERENCE_TRANSFORM 旋转部分必须是有效旋转矩阵")
    if not 5 <= int(calibration_samples) <= 300:
        raise ValueError("REALSENSE_CALIBRATION_SAMPLES 必须在 5 到 300 之间")
    if float(calibration_max_std_mm) <= 0 or float(calibration_max_angle_deg) <= 0:
        raise ValueError("RealSense 标定稳定性阈值必须大于 0")

    return RealSenseConfig(
        serial=serial.strip(),
        width=int(width),
        height=int(height),
        fps=int(fps),
        tag_family=normalized_family,
        tag_ids=normalized_ids,
        board_rows=int(board_rows),
        board_cols=int(board_cols),
        tag_size_m=float(tag_size_m),
        tag_spacing_m=float(tag_spacing_m),
        min_visible_tags=int(min_visible_tags),
        record_format=normalized_record_format,
        reference_transform=reference,
        calibration_samples=int(calibration_samples),
        calibration_max_std_mm=float(calibration_max_std_mm),
        calibration_max_angle_deg=float(calibration_max_angle_deg),
    )


def save_realsense_config(root: Path, config: RealSenseConfig) -> Path:
    """Persist RealSense and AprilTag calibration settings to root .env."""
    env_path = root.resolve() / ".env"
    env_path.touch(exist_ok=True)
    values = {
        "REALSENSE_SERIAL": config.serial,
        "REALSENSE_WIDTH": str(config.width),
        "REALSENSE_HEIGHT": str(config.height),
        "REALSENSE_FPS": str(config.fps),
        "APRILTAG_FAMILY": config.tag_family,
        "APRILTAG_IDS": ",".join(str(tag_id) for tag_id in config.tag_ids),
        "APRILTAG_BOARD_ROWS": str(config.board_rows),
        "APRILTAG_BOARD_COLS": str(config.board_cols),
        "APRILTAG_SIZE_M": f"{config.tag_size_m:.12g}",
        "APRILTAG_SPACING_M": f"{config.tag_spacing_m:.12g}",
        "APRILTAG_MIN_VISIBLE_TAGS": str(config.min_visible_tags),
        "REALSENSE_RECORD_FORMAT": config.record_format,
        "REALSENSE_REFERENCE_TRANSFORM": ",".join(
            f"{value:.12g}" for value in config.reference_transform
        ),
        "REALSENSE_CALIBRATION_SAMPLES": str(config.calibration_samples),
        "REALSENSE_CALIBRATION_MAX_STD_MM": f"{config.calibration_max_std_mm:.12g}",
        "REALSENSE_CALIBRATION_MAX_ANGLE_DEG": f"{config.calibration_max_angle_deg:.12g}",
    }
    for key, value in values.items():
        set_key(str(env_path), key, value, quote_mode="auto")
    return env_path


def build_ndi_config(
    root: Path,
    tracker_type: str,
    ip_address: str,
    port: int,
    serial_port: str,
    rom_files: Iterable[str | Path],
    record_orientation: str = "quaternion",
    record_format: str = "csv",
) -> NDIConfig:
    """Validate user-entered NDI settings and resolve ROM paths."""
    root = root.resolve()
    normalized_type = tracker_type.strip().lower()
    if normalized_type not in {"vega", "polaris"}:
        raise ValueError("NDI 设备类型仅支持 Vega 或 Polaris")
    if normalized_type == "vega" and not ip_address.strip():
        raise ValueError("Vega 模式必须填写 IP 地址")
    if not 1 <= int(port) <= 65535:
        raise ValueError("NDI 端口必须在 1 到 65535 之间")

    resolved_roms: list[Path] = []
    for raw_path in rom_files:
        value = str(raw_path).strip()
        if not value:
            continue
        path = Path(value).expanduser()
        resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
        if resolved.suffix.lower() != ".rom":
            raise ValueError(f"NDI 工具配置必须是 .rom 文件：{value}")
        resolved_roms.append(resolved)
    if not resolved_roms:
        raise ValueError("至少需要配置一个 NDI ROM 文件")
    normalized_orientation = record_orientation.strip().lower()
    if normalized_orientation not in {"quaternion", "matrix"}:
        raise ValueError("NDI 轨迹姿态格式仅支持 quaternion 或 matrix")
    normalized_format = record_format.strip().lower()
    if normalized_format not in {"csv", "xlsx"}:
        raise ValueError("NDI 轨迹文件格式仅支持 csv 或 xlsx")

    return NDIConfig(
        tracker_type=normalized_type,
        ip_address=ip_address.strip(),
        port=int(port),
        serial_port=serial_port.strip(),
        rom_files=tuple(resolved_roms),
        record_orientation=normalized_orientation,
        record_format=normalized_format,
    )


def save_ndi_config(root: Path, config: NDIConfig) -> Path:
    """Persist NDI connection settings to the project-root .env file."""
    root = root.resolve()
    env_path = root / ".env"
    env_path.touch(exist_ok=True)

    serialized_roms: list[str] = []
    for path in config.rom_files:
        try:
            serialized_roms.append(str(path.relative_to(root)))
        except ValueError:
            serialized_roms.append(str(path))
    values = {
        "NDI_TRACKER_TYPE": config.tracker_type,
        "NDI_IP_ADDRESS": config.ip_address,
        "NDI_PORT": str(config.port),
        "NDI_SERIAL_PORT": config.serial_port,
        "NDI_ROM_FILES": ",".join(serialized_roms),
        "NDI_RECORD_ORIENTATION": config.record_orientation,
        "NDI_RECORD_FORMAT": config.record_format,
    }
    for key, value in values.items():
        set_key(str(env_path), key, value, quote_mode="auto")
    return env_path


def load_config(root: Path | None = None, environ: Mapping[str, str] | None = None) -> AppConfig:
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    env_path = root / ".env"
    file_values = {key: value for key, value in dotenv_values(env_path).items() if value is not None}
    values = dict(file_values)
    values.update(dict(os.environ if environ is None else environ))

    source = values.get("MOCAP_SOURCE", "simulator").strip().lower()
    if source not in {"simulator", "realsense", "ndi"}:
        raise ValueError("MOCAP_SOURCE 仅支持 simulator、realsense 或 ndi")

    raw_tag_ids = _csv(values, "APRILTAG_IDS") or ("0",)
    raw_reference = _csv(values, "REALSENSE_REFERENCE_TRANSFORM")
    try:
        reference_transform = tuple(float(value) for value in raw_reference) if raw_reference else None
    except ValueError as exc:
        raise ValueError("REALSENSE_REFERENCE_TRANSFORM 必须是逗号分隔的数字") from exc
    realsense = build_realsense_config(
        serial=values.get("REALSENSE_SERIAL", ""),
        width=_integer(values, "REALSENSE_WIDTH", 1280),
        height=_integer(values, "REALSENSE_HEIGHT", 720),
        fps=_integer(values, "REALSENSE_FPS", 30),
        tag_family=values.get("APRILTAG_FAMILY", "tag36h11"),
        tag_ids=raw_tag_ids,
        board_rows=_integer(values, "APRILTAG_BOARD_ROWS", 1),
        board_cols=_integer(values, "APRILTAG_BOARD_COLS", len(raw_tag_ids)),
        tag_size_m=_floating(values, "APRILTAG_SIZE_M", 0.08),
        tag_spacing_m=_floating(values, "APRILTAG_SPACING_M", 0.02),
        min_visible_tags=_integer(values, "APRILTAG_MIN_VISIBLE_TAGS", 1),
        record_format=values.get("REALSENSE_RECORD_FORMAT", "xlsx"),
        reference_transform=reference_transform,
        calibration_samples=_integer(values, "REALSENSE_CALIBRATION_SAMPLES", 30),
        calibration_max_std_mm=_floating(values, "REALSENSE_CALIBRATION_MAX_STD_MM", 2.0),
        calibration_max_angle_deg=_floating(values, "REALSENSE_CALIBRATION_MAX_ANGLE_DEG", 1.0),
    )

    rom_files = tuple(
        path if path.is_absolute() else (root / path).resolve()
        for path in (Path(item).expanduser() for item in _csv(values, "NDI_ROM_FILES"))
    )
    ndi = NDIConfig(
        tracker_type=values.get("NDI_TRACKER_TYPE", "vega").strip().lower(),
        ip_address=values.get("NDI_IP_ADDRESS", "192.168.2.17").strip(),
        port=_integer(values, "NDI_PORT", 8765),
        serial_port=values.get("NDI_SERIAL_PORT", "").strip(),
        rom_files=rom_files,
        record_orientation=values.get("NDI_RECORD_ORIENTATION", "quaternion").strip().lower(),
        record_format=values.get("NDI_RECORD_FORMAT", "csv").strip().lower(),
    )
    if ndi.tracker_type not in {"vega", "polaris"}:
        raise ValueError("NDI_TRACKER_TYPE 仅支持 vega 或 polaris")
    if ndi.record_orientation not in {"quaternion", "matrix"}:
        raise ValueError("NDI_RECORD_ORIENTATION 仅支持 quaternion 或 matrix")
    if ndi.record_format not in {"csv", "xlsx"}:
        raise ValueError("NDI_RECORD_FORMAT 仅支持 csv 或 xlsx")

    open3d = Open3DConfig(
        enabled=_boolean(values, "OPEN3D_ENABLED", True),
        render_hz=_integer(values, "OPEN3D_RENDER_HZ", 15),
        width=_integer(values, "OPEN3D_WIDTH", 720),
        height=_integer(values, "OPEN3D_HEIGHT", 360),
        trail_points=_integer(values, "OPEN3D_TRAIL_POINTS", 240),
        axis_size_mm=_floating(values, "OPEN3D_AXIS_SIZE_MM", 60.0),
        camera_fov_deg=_floating(values, "OPEN3D_CAMERA_FOV_DEG", 60.0),
    )
    if not 1 <= open3d.render_hz <= 60:
        raise ValueError("OPEN3D_RENDER_HZ 必须在 1 到 60 之间")
    if not 320 <= open3d.width <= 1920 or not 240 <= open3d.height <= 1080:
        raise ValueError("OPEN3D_WIDTH/HEIGHT 必须在 320×240 到 1920×1080 范围内")
    if not 2 <= open3d.trail_points <= 5000:
        raise ValueError("OPEN3D_TRAIL_POINTS 必须在 2 到 5000 之间")
    if open3d.axis_size_mm <= 0:
        raise ValueError("OPEN3D_AXIS_SIZE_MM 必须大于 0")
    if not 20 <= open3d.camera_fov_deg <= 90:
        raise ValueError("OPEN3D_CAMERA_FOV_DEG 必须在 20 到 90 之间")

    ffmpeg = FFmpegConfig(
        enabled=_boolean(values, "FFMPEG_VIDEO_ENABLED", True),
        executable=values.get("FFMPEG_PATH", "ffmpeg").strip(),
        codec=values.get("FFMPEG_VIDEO_CODEC", "libx264").strip(),
        preset=values.get("FFMPEG_VIDEO_PRESET", "veryfast").strip(),
        crf=_integer(values, "FFMPEG_VIDEO_CRF", 18),
    )
    if not ffmpeg.executable:
        raise ValueError("FFMPEG_PATH 不能为空")
    if not ffmpeg.codec:
        raise ValueError("FFMPEG_VIDEO_CODEC 不能为空")
    if ffmpeg.codec == "libx264" and not ffmpeg.preset:
        raise ValueError("FFMPEG_VIDEO_PRESET 不能为空")
    if not 0 <= ffmpeg.crf <= 51:
        raise ValueError("FFMPEG_VIDEO_CRF 必须在 0 到 51 之间")

    record_value = Path(values.get("MOCAP_RECORD_DIR", "recordings")).expanduser()
    record_dir = record_value if record_value.is_absolute() else root / record_value
    poll_hz = _integer(values, "MOCAP_POLL_HZ", 30)
    if not 1 <= poll_hz <= 240:
        raise ValueError("MOCAP_POLL_HZ 必须在 1 到 240 之间")
    database_value = Path(values.get("MOCAP_DATABASE_PATH", "data/motion_capture.sqlite3")).expanduser()
    database_path = database_value if database_value.is_absolute() else root / database_value
    return AppConfig(
        root,
        source,
        poll_hz,
        record_dir.resolve(),
        realsense,
        ndi,
        open3d,
        ffmpeg,
        database_path.resolve(),
    )
