#!/usr/bin/env bash
# 把本地编译的 pyrealsense2 装进项目 venv，避免 sudo make install 污染 /usr/local。
# 产物依赖 @rpath 定位核心库，这里改成 @loader_path 让包目录自包含。
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LIBRS_DIR="${LIBRS_DIR:-/Users/kuanmi/Public/librealsense}"
RELEASE_DIR="${RELEASE_DIR:-$LIBRS_DIR/build/Release}"
VENV="${VENV:-$PROJECT_DIR/.venv}"

PYTHON="$VENV/bin/python"
[ -x "$PYTHON" ] || { echo "错误: 找不到 venv 解释器 $PYTHON" >&2; exit 1; }

EXT_SUFFIX="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
SITE_PACKAGES="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
TARGET="$SITE_PACKAGES/pyrealsense2"

MODULE_LINK="$RELEASE_DIR/pyrealsense2$EXT_SUFFIX"
[ -e "$MODULE_LINK" ] || { echo "错误: 未找到 $MODULE_LINK，请先完成编译" >&2; exit 1; }

# 符号链接指向带版本号的真实文件，复制真实文件而非链接。
MODULE_REAL="$(cd "$RELEASE_DIR" && python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "pyrealsense2$EXT_SUFFIX")"

# 依赖名形如 librealsense2.2.57.dylib，必须原名保留，否则 dyld 找不到。
CORE_NAME="$(otool -L "$MODULE_REAL" | awk '/librealsense2\..*dylib/ {print $1}' | grep -v '^/Users' | head -1 | xargs basename)"
[ -n "$CORE_NAME" ] || { echo "错误: 无法从模块中解析核心库依赖名" >&2; exit 1; }
CORE_REAL="$(cd "$RELEASE_DIR" && python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$CORE_NAME")"
[ -f "$CORE_REAL" ] || { echo "错误: 未找到核心库 $CORE_NAME" >&2; exit 1; }

echo "模块:   $MODULE_REAL"
echo "核心库: $CORE_REAL  (依赖名 $CORE_NAME)"
echo "安装到: $TARGET"

rm -rf "$TARGET"
mkdir -p "$TARGET"
cp "$MODULE_REAL" "$TARGET/pyrealsense2$EXT_SUFFIX"
cp "$CORE_REAL" "$TARGET/$CORE_NAME"
printf 'from .pyrealsense2 import *\n' > "$TARGET/__init__.py"
chmod u+w "$TARGET/pyrealsense2$EXT_SUFFIX" "$TARGET/$CORE_NAME"

# @rpath 原本指向构建目录，加 @loader_path 让它在包内解析。
install_name_tool -add_rpath "@loader_path" "$TARGET/pyrealsense2$EXT_SUFFIX" 2>/dev/null || true
install_name_tool -id "@loader_path/$CORE_NAME" "$TARGET/$CORE_NAME"

# Apple Silicon 上修改过 Mach-O 头后必须重新签名，否则加载被拒。
codesign --force --sign - "$TARGET/$CORE_NAME"
codesign --force --sign - "$TARGET/pyrealsense2$EXT_SUFFIX"

echo
echo "--- 验证 ---"
"$PYTHON" -c "
import pyrealsense2 as rs
print('pyrealsense2 导入成功:', rs.__file__)
ctx = rs.context()
devices = ctx.query_devices()
print('枚举到设备数:', len(devices))
for d in devices:
    def info(k):
        try: return d.get_info(k)
        except Exception: return '?'
    print(' ', info(rs.camera_info.name), '| SN', info(rs.camera_info.serial_number),
          '| 固件', info(rs.camera_info.firmware_version), '| USB', info(rs.camera_info.usb_type_descriptor))
"
