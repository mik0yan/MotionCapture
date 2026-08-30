from __future__ import annotations

import argparse
import json
import os
import plistlib
import re
import subprocess
import sys
import time
from dataclasses import dataclass

INTEL_VENDOR_ID = 0x8086

USB_SPEEDS = {
    0: "Low Speed (1.5 Mb/s)",
    1: "Full Speed (12 Mb/s, USB 1.1)",
    2: "High Speed (480 Mb/s, USB 2.0)",
    3: "SuperSpeed (5 Gb/s, USB 3.0)",
    4: "SuperSpeed+ (10 Gb/s, USB 3.1)",
    5: "SuperSpeed+ (20 Gb/s, USB 3.2)",
}


@dataclass(frozen=True)
class UsbDevice:
    name: str
    serial: str
    vendor_id: int
    product_id: int
    speed: int | None
    location_id: int
    hub_chain: tuple[str, ...]

    @property
    def speed_label(self) -> str:
        if self.speed is None:
            return "未知"
        return USB_SPEEDS.get(self.speed, f"未知速率码 {self.speed}")

    @property
    def is_usb3(self) -> bool:
        return self.speed is not None and self.speed >= 3

    @property
    def avfoundation_uid(self) -> str:
        """AVFoundation 给 UVC 设备的 uniqueID 是 locationID + VID + PID 的拼接。"""
        return f"{self.location_id:x}{self.vendor_id:04x}{self.product_id:04x}"


@dataclass(frozen=True)
class CaptureDevice:
    index: int
    name: str
    unique_id: str


def _run(command: list[str]) -> bytes:
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} 执行失败: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def usb_devices() -> tuple[UsbDevice, ...]:
    if sys.platform == "darwin":
        return _darwin_usb_devices()
    if sys.platform == "win32":
        return _windows_usb_devices()
    if sys.platform.startswith("linux"):
        return _linux_usb_devices()
    return ()


def _darwin_usb_devices() -> tuple[UsbDevice, ...]:
    tree = plistlib.loads(_run(["ioreg", "-a", "-r", "-c", "IOUSBHostDevice", "-l", "-w0"]))
    found: dict[int, UsbDevice] = {}

    def walk(nodes: list[dict], hubs: tuple[str, ...]) -> None:
        for node in nodes:
            name = str(node.get("USB Product Name", "")).strip()
            vendor_id = node.get("idVendor")
            location_id = node.get("locationID")
            is_device = isinstance(vendor_id, int) and isinstance(location_id, int)
            if is_device and location_id not in found and node.get("Device Speed") is not None:
                found[location_id] = UsbDevice(
                    name=name or "(未命名设备)",
                    serial=str(node.get("USB Serial Number", "")).strip(),
                    vendor_id=vendor_id,
                    product_id=int(node.get("idProduct", 0)),
                    speed=node.get("Device Speed"),
                    location_id=location_id,
                    hub_chain=hubs,
                )
            children = node.get("IORegistryEntryChildren", [])
            child_hubs = hubs + (name,) if is_device and "hub" in name.lower() else hubs
            walk(children, child_hubs)

    walk(tree, ())
    return tuple(found.values())


def _windows_usb_devices() -> tuple[UsbDevice, ...]:
    script = (
        "Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -like 'USB\\VID_*' } | "
        "ForEach-Object { \"$($_.InstanceId)|$($_.FriendlyName)\" }"
    )
    output = _run(["powershell", "-NoProfile", "-Command", script]).decode("utf-8", errors="replace")
    devices: list[UsbDevice] = []
    for line in output.splitlines():
        entry = line.strip()
        if not entry:
            continue
        instance_id, _, name = entry.partition("|")
        match = re.search(r"VID_([0-9A-Fa-f]{4})&PID_([0-9A-Fa-f]{4})", instance_id)
        if match is None:
            continue
        # USB\VID_xxxx&PID_yyyy\<serial> 的末段通常是序列号；Windows 生成的
        # 临时实例号（含 & ）没有稳定意义，但保留作诊断线索。
        serial = instance_id.rsplit("\\", 1)[-1]
        devices.append(
            UsbDevice(
                name=name.strip() or "(未命名设备)",
                serial=serial,
                vendor_id=int(match.group(1), 16),
                product_id=int(match.group(2), 16),
                speed=None,
                location_id=0,
                hub_chain=(),
            )
        )
    return tuple(devices)


def _linux_usb_devices() -> tuple[UsbDevice, ...]:
    try:
        output = _run(["lsusb"]).decode("utf-8", errors="replace")
    except (OSError, RuntimeError):
        return ()
    devices: list[UsbDevice] = []
    for line in output.splitlines():
        match = re.search(r"ID ([0-9a-fA-F]{4}):([0-9a-fA-F]{4}) (.+)$", line.strip())
        if match is None:
            continue
        devices.append(
            UsbDevice(
                name=match.group(3).strip() or "(未命名设备)",
                serial="",
                vendor_id=int(match.group(1), 16),
                product_id=int(match.group(2), 16),
                speed=None,
                location_id=0,
                hub_chain=(),
            )
        )
    return tuple(devices)


def realsense_usb_devices() -> tuple[UsbDevice, ...]:
    return tuple(
        device
        for device in usb_devices()
        if device.vendor_id == INTEL_VENDOR_ID and "realsense" in device.name.lower()
    )


def _system_profiler_cameras() -> tuple[str, ...]:
    payload = json.loads(_run(["system_profiler", "-json", "SPCameraDataType"]))
    return tuple(str(item.get("_name", "")).strip() for item in payload.get("SPCameraDataType", []))


def _windows_pnp_cameras() -> tuple[str, ...]:
    script = (
        "Get-PnpDevice -Class Camera -PresentOnly | "
        "Sort-Object FriendlyName | "
        "ForEach-Object { $_.FriendlyName }"
    )
    output = _run(["powershell", "-NoProfile", "-Command", script]).decode("utf-8", errors="replace")
    return tuple(line.strip() for line in output.splitlines() if line.strip())


def _linux_video_devices() -> tuple[str, ...]:
    if not os.path.isdir("/dev"):
        return ()
    return tuple(
        f"/dev/{name}" for name in sorted(os.listdir("/dev")) if re.fullmatch(r"video\d+", name)
    )


def capture_devices() -> tuple[CaptureDevice, ...]:
    """枚举 OpenCV VideoCapture 可用的摄像头。

    macOS 上按 AVFoundation 后端顺序（Video + Muxed）返回，下标即 VideoCapture
    的索引，可用 pyobjc 交叉验证 uniqueID。Windows/Linux 枚举接口不保证与
    OpenCV 后端顺序一致，索引仅作参考，必要时用 --index 手动指定。
    """
    if sys.platform == "darwin":
        try:
            import AVFoundation as AVF
        except ImportError:
            return tuple(CaptureDevice(index, name, "") for index, name in enumerate(_system_profiler_cameras()))

        video = AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo) or []
        muxed = AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeMuxed) or []
        return tuple(
            CaptureDevice(index, str(device.localizedName()), str(device.uniqueID()))
            for index, device in enumerate(list(video) + list(muxed))
        )
    if sys.platform == "win32":
        return tuple(CaptureDevice(index, name, "") for index, name in enumerate(_windows_pnp_cameras()))
    if sys.platform.startswith("linux"):
        return tuple(CaptureDevice(index, name, "") for index, name in enumerate(_linux_video_devices()))
    return ()


def realsense_capture_device() -> CaptureDevice | None:
    """定位 RealSense 的彩色流索引，优先用 USB locationID/VID/PID 交叉验证。"""
    devices = capture_devices()
    usb = realsense_usb_devices()
    for device in devices:
        uid = device.unique_id.lower().removeprefix("0x")
        if any(uid == camera.avfoundation_uid for camera in usb):
            return device

    def is_color(node: CaptureDevice) -> bool:
        lowered = node.name.lower()
        return "realsense" in lowered and ("rgb" in lowered or "color" in lowered)

    for device in devices:
        if is_color(device):
            return device
    for device in devices:
        if "realsense" in device.name.lower():
            return device
    return None


def pyrealsense_report() -> list[str]:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        return [f"pyrealsense2 未安装 ({exc})"]

    lines = [f"pyrealsense2 {getattr(rs, '__version__', '?')} 已安装 ({rs.__file__})"]
    try:
        devices = rs.context().query_devices()
    except Exception as exc:
        lines.append(f"查询设备失败: {type(exc).__name__}: {exc}")
        return lines

    count = len(devices)
    lines.append(f"SDK 枚举到 {count} 个设备")
    if count == 0:
        return lines

    # macOS 12+ 下不带 sudo 时，取设备句柄就会抛 "failed to set power state"。
    try:
        for device in devices:
            def info(key) -> str:
                try:
                    return device.get_info(key)
                except Exception:
                    return "?"

            lines.append(
                f"{info(rs.camera_info.name)} | SN {info(rs.camera_info.serial_number)} "
                f"| 固件 {info(rs.camera_info.firmware_version)} | USB {info(rs.camera_info.usb_type_descriptor)}"
            )
    except Exception as exc:
        lines.append(f"读取设备信息失败: {type(exc).__name__}: {exc}")
        # os.geteuid 仅存在于 Unix；Windows 上平台判断必须短路在前。
        if "power state" in str(exc).lower() or (hasattr(os, "geteuid") and os.geteuid() != 0):
            lines.append("原因: macOS 12+ 的 USB 安全策略要求 librealsense 以 root 访问设备。")
            lines.append("用 sudo 重跑本脚本即可读到设备信息，例如:")
            lines.append("  sudo .venv/bin/python tools/check_realsense.py")
    return lines


def uvc_backend() -> int:
    """按平台选择 OpenCV VideoCapture 后端常量。"""
    import cv2

    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def probe_opencv(indices: list[int]) -> bool:
    """逐个试开摄像头并即时打印。单次打开可能耗时数秒，因此不缓冲输出。"""
    try:
        import cv2
    except ImportError as exc:
        print(f"  opencv 未安装 ({exc})", flush=True)
        return False

    names = {device.index: device.name for device in capture_devices()}
    any_frame = False
    for index in indices:
        print(f"  索引 {index} ({names.get(index, '?')}) 试打开...", end=" ", flush=True)
        started = time.monotonic()
        capture = cv2.VideoCapture(index, uvc_backend())
        if not capture.isOpened():
            capture.release()
            print(f"失败 ({time.monotonic() - started:.1f}s)", flush=True)
            continue
        ok, frame = capture.read()
        elapsed = time.monotonic() - started
        if ok and frame is not None:
            any_frame = True
            print(f"成功, 首帧 {frame.shape[1]}x{frame.shape[0]} ({elapsed:.1f}s)", flush=True)
        else:
            print(f"已打开但读不到帧 ({elapsed:.1f}s)", flush=True)
        capture.release()
    return any_frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查 RealSense 的 USB 连接与可读性")
    parser.add_argument("--probe", action="store_true", help="用 OpenCV 实际打开摄像头读一帧（会触发系统摄像头权限）")
    parser.add_argument("--index", type=int, default=None, help="只探测指定索引，默认自动定位 RealSense")
    parser.add_argument("--all", action="store_true", help="配合 --probe 探测全部索引")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    print("=== USB 连接 ===")
    cameras = realsense_usb_devices()
    if not cameras:
        print("未在 USB 总线上找到 RealSense 相机。检查线缆与供电，确认使用数据线而非充电线。")
        if sys.platform != "darwin":
            print("（Windows/Linux 按设备名称匹配，若 SDK 能枚举到设备则以 SDK 结果为准）")
    for device in cameras:
        print(f"设备: {device.name}")
        print(f"  序列号: {device.serial or '(无)'}")
        print(f"  VID:PID: 0x{device.vendor_id:04X}:0x{device.product_id:04X}")
        print(f"  链路速率: {device.speed_label}")
        if device.hub_chain:
            print(f"  接入路径: {' -> '.join(device.hub_chain)}")
        elif sys.platform == "darwin":
            print("  接入路径: 直连主机端口")
        else:
            print("  接入路径: 未知（当前平台不提供 Hub 拓扑）")
        if device.speed is not None and not device.is_usb3:
            print("  警告: 以 USB 2.0 及以下速率握手，深度+彩色高分辨率同步会受限。改用 USB 3 端口与线缆。")

    print()
    print("=== pyrealsense2 (深度/红外/对齐所需) ===")
    for line in pyrealsense_report():
        print(f"  {line}")

    print()
    print("=== 摄像头索引 (OpenCV VideoCapture 顺序) ===")
    target = realsense_capture_device()
    for device in capture_devices():
        marker = "  <- RealSense 彩色流" if target is not None and device.index == target.index else ""
        print(f"  索引 {device.index}: {device.name}{marker}")
    if target is None:
        print("  未能定位 RealSense 的彩色流")

    if args.probe:
        print()
        print("=== OpenCV 读取测试 ===")
        if args.index is not None:
            indices = [args.index]
        elif args.all:
            indices = [device.index for device in capture_devices()]
        elif target is not None:
            indices = [target.index]
        else:
            indices = [device.index for device in capture_devices()]
        if not probe_opencv(indices):
            if sys.platform == "darwin":
                print("  没读到画面。若上方出现 not authorized，需在 系统设置 → 隐私与安全性 → 摄像头 授权当前终端，并 ⌘Q 完全退出后重开。")
            else:
                print("  没读到画面。确认摄像头未被其他程序独占，或用 --index 换一个索引重试。")

    return 0 if cameras else 1


if __name__ == "__main__":
    raise SystemExit(main())
