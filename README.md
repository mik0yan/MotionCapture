# MotionCapture

基于 PyQt5 的统一动作捕获界面，支持：

- Intel RealSense 彩色流 + AprilTag 网格标定板位姿；
- NDI Polaris Vega / Polaris Vicra、Spectra + ROM 红外反光球工具；
- 无硬件模拟器、AprilTag 独立监控、Open3D 三维空间轨迹和 CSV/XLSX 录制；
- 本地 SQLite 持久化设备设置、Tag 监控参数、采集会话和逐帧位姿样本。

> 当前版本是科研/工程采集工具，不是已验证的医疗器械软件。进入测量或临床验证前，需要完成坐标系、精度、丢帧、时间同步和设备异常的独立确认。

## 快速启动

推荐 Python 3.10–3.12。先在项目根目录创建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python main.py
```

默认 `.env` 使用 `MOCAP_SOURCE=simulator`，可在没有硬件时直接验收界面、轨迹和录制功能。

视频录制依赖本机 FFmpeg。macOS 可使用 `brew install ffmpeg`，随后用 `ffmpeg -version` 确认命令可执行；也可通过 `FFMPEG_PATH` 配置绝对路径。

### 平台兼容性

`requirements.txt` 使用精确版本和 PEP 508 平台条件，目标安装矩阵如下：

| 平台 | Python | `pip install -r requirements.txt` | RealSense | FFmpeg |
| --- | --- | --- | --- | --- |
| macOS 11+ Intel / Apple Silicon | 3.10–3.12 | 支持 | 运行 `tools/install_pyrealsense2.sh` 安装本地编译绑定 | Homebrew 或配置 `FFMPEG_PATH` |
| Windows 10/11 x64 | 3.10–3.12 | 支持 | requirements 自动安装锁定 wheel | 单独安装并加入 `PATH` |
| Linux x86_64，glibc 2.31+ | 3.10–3.12 | 支持 | requirements 自动安装锁定 wheel | 使用系统包管理器安装 |

Linux ARM64 和 Windows ARM64 缺少当前锁定版本完整的 PyQt5、Open3D、RealSense 官方 wheel 组合，不属于本锁定文件的直接安装矩阵，需要自行编译对应依赖。FFmpeg 是系统程序而不是 Python 库，因此不使用 PyPI 上名称相似的包替代。

主界面按 Figma `01 · RealSense 实时监控 · 监控功能` 重构：左侧为 1280×720 监控画面、RGB/D/Tag 图层开关、录制计时和实时指标，右侧为独立 AprilTag 监控卡及设置页。设置页可切换 `RealSense（USB）`、`NDI（IP/串口）` 和 `模拟器`，连接期间会锁定配置。

每张 AprilTag 卡片直接显示 XYZ、相机距离偏移量和最近 5 秒位置方差；长按卡片至少 500 ms 可单独修改该 ID 的实际尺寸与质量阈值。运行中修改尺寸会立即用于该 Tag 的后续 PnP 解算，并持久化到本地 SQLite。监控画面内的方向箭头同样基于该阈值：Tag 相对初始基准位置的偏移超过卡片阈值后，画面上才显示从初始位置指向当前偏移方向的箭头，回到阈值内箭头消失；可用画面左上角图层面板的「方向」开关整体关闭。

Open3D 0.19 由基础依赖安装；在“设置”页点击“打开 Open3D 三维轨迹”进入独立窗口。三维面板通过隐藏的 GLFW 渲染上下文生成画面并嵌入 PyQt5。

## 配置

实际配置保存在根目录 `.env`，该文件已加入 `.gitignore`；可提交的模板是 `.env.example`。手工修改 `.env` 后重启应用。

### 本地 SQLite 数据库

首次启动会自动创建 `data/motion_capture.sqlite3`。应用会把 `.env` 作为首次启动/部署默认值导入 SQLite，之后界面保存的运行配置以 SQLite 为准，同时回写 `.env` 以兼容现有硬件脚本。

```dotenv
MOCAP_DATABASE_PATH=data/motion_capture.sqlite3
```

数据库包含：

- `app_settings`：当前运行配置；
- `tag_monitors`：Tag 尺寸、质量阈值、警戒距离和启停状态；
- `capture_sessions`：每次设备连接形成的采集会话；
- `pose_samples`：会话内 XYZ、姿态、质量与硬件帧号。

### Open3D 三维实时显示

```dotenv
OPEN3D_ENABLED=true
OPEN3D_RENDER_HZ=15
OPEN3D_WIDTH=720
OPEN3D_HEIGHT=360
OPEN3D_TRAIL_POINTS=240
OPEN3D_AXIS_SIZE_MM=60
OPEN3D_CAMERA_FOV_DEG=60
```

三维视图直接使用 `PoseSample` 的 XYZ 毫米坐标和 3×3 旋转矩阵：

- 红、绿、蓝轴分别表示工具局部 X、Y、Z；
- 彩色折线是每个工具最近 `OPEN3D_TRAIL_POINTS` 个有效位置；
- 数据采集频率与渲染频率解耦，`OPEN3D_RENDER_HZ` 只控制界面刷新，不会降低 CSV 记录频率；
- RealSense 未设置初始零点时输出标定板相对彩色相机光学坐标系，设置后输出相对该零点的位姿；NDI 仍按 SDK 坐标系显示。不同来源的数据不能在未完成外参标定时直接叠加解释。

若 Open3D 初始化或图形驱动失败，错误会显示在三维面板内，设备采集、位姿表和 CSV 录制仍可继续使用。可设置 `OPEN3D_ENABLED=false` 暂时关闭三维渲染。

### RealSense + AprilTag 标定板

在“设置”页选择“Intel RealSense（USB 设备）”后，可设置设备序列号、彩色流分辨率/FPS、Tag family、ID 顺序、标定板行列、Tag 实际边长、间距、最少可见 Tag 数和位置导出格式。点击“保存设置”或“连接设备”都会把当前配置写入本地 SQLite，并同步根目录 `.env`。

```dotenv
MOCAP_SOURCE=realsense
REALSENSE_SERIAL=
REALSENSE_WIDTH=1280
REALSENSE_HEIGHT=720
REALSENSE_FPS=30
APRILTAG_FAMILY=tag36h11
APRILTAG_IDS=0,1,2,3
APRILTAG_BOARD_ROWS=2
APRILTAG_BOARD_COLS=2
APRILTAG_SIZE_M=0.024
APRILTAG_SPACING_M=0.020
APRILTAG_MIN_VISIBLE_TAGS=1
REALSENSE_RECORD_FORMAT=xlsx
REALSENSE_REFERENCE_TRANSFORM=1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1
REALSENSE_CALIBRATION_SAMPLES=30
REALSENSE_CALIBRATION_MAX_STD_MM=2.0
REALSENSE_CALIBRATION_MAX_ANGLE_DEG=1.0
```

`APRILTAG_SIZE_M` 是单个黑色 Tag 外边界的实际边长；`APRILTAG_SPACING_M` 是相邻 Tag 黑色边界间距。ID 按从左到右、从上到下排列。可见 Tag 数达到 `APRILTAG_MIN_VISIBLE_TAGS` 后，程序才使用 RealSense RGB 内参和所有可见板角点联合执行 PnP，得到“标定板坐标系相对彩色相机光学坐标系”的原始位姿，平移单位为 mm。

连接 RealSense 后，将相机和标定板固定并确保所有 Tag 完整可见，再点击“开始初始标定”。程序默认连续采集 30 帧，计算平均参考位姿，并检查位置 RMS 与最大角度波动；稳定性通过后把 4×4 参考矩阵写入 `REALSENSE_REFERENCE_TRANSFORM`。之后显示和位置记录的数据为 `T_reference⁻¹ × T_current`，标定时刻对应 XYZ≈0、姿态≈0。点击“清除零点”可恢复相机光学坐标系原始输出。

RealSense 连接成功后点击“开始位置记录”，按 `REALSENSE_RECORD_FORMAT` 输出 `xlsx` 或 `csv`。默认 XLSX 工作表名为 `trajectory`，包含 UTC 时间、来源、工具、硬件帧号、有效性、质量及 `x_mm,y_mm,z_mm` 数值列，不混入姿态列。

此工具用于建立运行时初始零点，不替代 RealSense 相机内参、畸变或多相机外参标定。

安装 RealSense 硬件依赖：

```bash
python -m pip install pyrealsense2
```

macOS 上若当前 Python 没有可用的 `pyrealsense2` wheel，应安装与 librealsense 匹配的 Python/SDK；这与 PyQt 界面本身无关。

### NDI + ROM 工具

在主界面选择“NDI / ROM 反光球工具”后，会显示 NDI 连接配置表单。Vega 使用 IP 地址和端口，Polaris 使用串口；可通过文件选择器一次添加多个 ROM。点击“保存配置”会写入根目录 `.env`，点击“连接设备”也会先校验并保存表单，因此不需要重启应用。

```dotenv
MOCAP_SOURCE=ndi
NDI_TRACKER_TYPE=vega
NDI_IP_ADDRESS=192.168.2.17
NDI_PORT=8765
NDI_ROM_FILES=roms/tool_1.rom,roms/tool_2.rom
NDI_RECORD_ORIENTATION=quaternion
NDI_RECORD_FORMAT=csv
```

NDI 面板中的“导入 ROM”会把外部 `.rom` 文件复制到项目根目录 `roms/` 并自动加入配置：相同内容直接复用，同名但内容不同的文件会自动保存为 `_2`、`_3` 等名称，不会覆盖已有工具。ROM 路径仍可手工填写绝对路径或项目根目录相对路径。Polaris Vicra/Spectra 可改为：

```dotenv
NDI_TRACKER_TYPE=polaris
NDI_SERIAL_PORT=/dev/tty.usbserial-XXXX
NDI_ROM_FILES=roms/tool_1.rom
```

NDI 面板还可选择轨迹姿态和文件格式：

- `quaternion`：保存 `qw,qx,qy,qz`；
- `matrix`：按行保存方向余弦矩阵 `r11` 到 `r33`；
- `csv`：UTF-8 CSV，采集时逐帧刷新；
- `xlsx`：数值单元格工作簿，工作表名为 `trajectory`，录制期间周期性自动保存。

安装 NDI Python 接口：

```bash
python -m pip install scikit-surgerynditracker
```

NDI 位姿矩阵中的平移按 SDK 约定作为 mm 原样记录。采集与对齐应优先使用 NDI `frame_number`，不要把主机时钟当成硬件采样时钟。

## 本地 FFmpeg 视频与轨迹输出

点击画面内录制按钮后，RealSense/模拟器的 RGB 帧通过后台队列写入本机 FFmpeg，默认使用 `libx264` 编码为兼容性较好的 `yuv420p` MP4；停止后执行 `+faststart` 封装并生成同名 JSON 元数据。视频先写入隐藏的 `.partial.mp4`，只有 FFmpeg 正常结束后才改名为正式文件。NDI 当前后端只提供位姿数据、不提供视频帧，因此保持轨迹记录。

视频与轨迹文件使用同一录制编号，例如：

```text
recordings/capture_20260829_143218_123456.mp4
recordings/capture_20260829_143218_123456.json
recordings/capture_20260829_143218_123456.xlsx
```

轨迹公共字段包括 UTC 时间、来源、工具、硬件帧号、有效性、质量和 XYZ。NDI 模式根据表单追加四元数或 3×3 方向余弦矩阵，并输出 CSV 或 XLSX；RealSense 默认输出仅含位置的 XLSX，也可在表单切换为 CSV；模拟器默认输出 CSV。`.env` 中可通过 `MOCAP_RECORD_DIR` 修改目录，通过 `FFMPEG_VIDEO_ENABLED`、`FFMPEG_PATH`、`FFMPEG_VIDEO_CODEC`、`FFMPEG_VIDEO_PRESET` 和 `FFMPEG_VIDEO_CRF` 调整本地视频编码。

## 测试

```bash
python -m pytest
```

硬件联调建议至少逐项核对：ROM 与实体工具对应关系、AprilTag 打印尺寸、相机内参、坐标轴方向、静态抖动、工作距离全范围误差、遮挡恢复、丢帧标记和 CSV 帧号连续性。
