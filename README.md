# MotionCapture

基于 PyQt5 的统一动作捕获界面，支持：

- Intel RealSense 彩色流 + AprilTag 网格标定板位姿；
- NDI Polaris Vega / Polaris Vicra、Spectra + ROM 红外反光球工具；
- 无硬件模拟器、实时位姿表、Open3D 三维空间轨迹和 CSV 录制。

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

主界面顶部提供三段式“工作模式”状态切换器：`NDI 模式`、`RealSense`、`模拟器模式`。选中按钮使用青绿色状态，切换时自动显示对应设备配置表单并更新根目录 `.env` 的 `MOCAP_SOURCE`；设备连接期间切换器会锁定，断开后恢复。

Open3D 0.19 由基础依赖安装。三维面板通过隐藏的 GLFW 渲染上下文生成画面并嵌入 PyQt5，不会显示额外的 Open3D 窗口。

## 配置

实际配置保存在根目录 `.env`，该文件已加入 `.gitignore`；可提交的模板是 `.env.example`。手工修改 `.env` 后重启应用。

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

在主界面选择“Intel RealSense / AprilTag”后，会显示 RealSense 配置与初始标定表单，可设置设备序列号、彩色流分辨率/FPS、Tag family、ID 顺序、标定板行列、Tag 实际边长、间距、最少可见 Tag 数和位置导出格式。点击“保存配置”或“连接设备”都会把当前表单写入根目录 `.env`。

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
APRILTAG_SIZE_M=0.080
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

## 轨迹输出

录制文件默认写到 `recordings/`，公共字段包括 UTC 时间、来源、工具、硬件帧号、有效性、质量和 XYZ。NDI 模式根据表单追加四元数或 3×3 方向余弦矩阵，并输出 CSV 或 XLSX；RealSense 默认输出仅含位置的 XLSX，也可在表单切换为 CSV；模拟器默认输出 CSV + 四元数。`.env` 中可通过 `MOCAP_RECORD_DIR` 修改目录。

## 测试

```bash
python -m pip install pytest
python -m pytest
```

硬件联调建议至少逐项核对：ROM 与实体工具对应关系、AprilTag 打印尺寸、相机内参、坐标轴方向、静态抖动、工作距离全范围误差、遮挡恢复、丢帧标记和 CSV 帧号连续性。
