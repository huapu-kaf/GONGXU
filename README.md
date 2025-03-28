# 视觉识别系统

<div align="center">
<img src="https://img.shields.io/badge/Python-3.6+-blue.svg" alt="Python 3.6+"/>
<img src="https://img.shields.io/badge/OpenCV-4.5+-green.svg" alt="OpenCV 4.5+"/>
<img src="https://img.shields.io/badge/Platform-JetsonNano-orange.svg" alt="Jetson Nano"/>
</div>

## 系统简介

这是一个基于OpenCV和PyTorch的视觉识别系统，主要功能包括：

- **圆环检测**：检测不同尺寸和颜色的圆环（红、绿、蓝）
- **物料识别**：识别不同颜色的物料
- **二维码识别**：检测和解码二维码
- **串口通信**：将识别结果通过串口发送至下位机

系统支持多种运行模式，包括带YOLO模型的高精度模式和基于传统计算机视觉的轻量级模式，适合在资源受限的平台如Jetson Nano上运行。

## 目录

- [快速开始](#快速开始)
- [系统架构](#系统架构)
- [部署指南](#部署指南)
  - [在Jetson Nano上部署](#在jetson-nano上部署)
  - [在普通PC上部署](#在普通pc上部署)
- [使用指南](#使用指南)
  - [运行程序](#运行程序)
  - [参数配置](#参数配置)
- [配置文件详解](#配置文件详解)
- [串口通信协议](#串口通信协议)
- [常见问题](#常见问题)
- [故障排除](#故障排除)

## 快速开始

### 在Jetson Nano上一键部署

```bash
# 1. 克隆仓库
git clone https://github.com/huapu/vision-system.git
cd vision-system

# 2. 运行安装脚本
chmod +x setup_jetson.sh
./setup_jetson.sh

# 3. 运行系统
./run_jetson.sh
```

## 系统架构

系统由以下几个主要模块组成：

- **CircleDetector**：圆环检测模块，支持霍夫圆变换和YOLOv5
- **ColorClassifier**：颜色分类器，基于HSV颜色空间
- **QRCodeDetector**：二维码检测和解码
- **SerialManager**：串口通信管理
- **EnhancedVisionPipeline**：核心流水线，协调各模块工作

## 部署指南

### 在Jetson Nano上部署

Jetson Nano是一个资源受限的平台，我们提供了优化的部署方案：

1. **系统要求**
   - Jetson Nano (2GB或4GB版本)
   - JetPack 4.5+
   - 16GB+ microSD卡

2. **安装步骤**
   ```bash
   # 运行安装脚本
   ./setup_jetson.sh
   ```

   安装脚本会自动完成以下工作：
   - 安装系统依赖
   - 安装适用于Jetson Nano的PyTorch和OpenCV
   - 下载YOLOv5模型
   - 创建优化的配置文件

3. **注意事项**
   - 建议使用外部摄像头，获得更好的图像质量
   - 默认使用无YOLO模式，以提高帧率

### 在普通PC上部署

如果您使用普通PC部署，请按以下步骤操作：

```bash
# 安装依赖
pip install -r requirements.txt

# 下载YOLOv5模型（可选）
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v6.1
pip install -r requirements.txt
cd ..
```

## 使用指南

### 运行程序

在Jetson Nano上：
```bash
# 基本运行
./run_jetson.sh

# 带参数运行
./run_jetson.sh --camera 0 --port /dev/ttyUSB0 --debug
```

在PC上：
```bash
# 无YOLO模式（推荐）
python main_enhanced.py --no-yolo

# 带YOLO模式
python main_enhanced.py

# 带串口通信
python main_enhanced.py --port COM3  # Windows
python main_enhanced.py --port /dev/ttyUSB0  # Linux
```

### 参数配置

所有配置参数都保存在`vision_params.yaml`（PC版）或`vision_params_jetson.yaml`（Jetson版）中。您可以通过编辑此文件来调整系统参数。

## 配置文件详解

配置文件采用YAML格式，分为以下几个主要部分：

### 1. 置信度阈值

```yaml
confidence_thresholds:
  circle_confidence: 40  # 圆环检测置信度阈值，范围0-100
  color_confidence: 50   # 颜色分类置信度阈值，范围0-100
  qr_confidence: 50      # 二维码检测置信度阈值，范围0-100
```

### 2. 霍夫圆检测参数

```yaml
hough_params:
  blur_ksize: 5       # 中值滤波核大小，必须是奇数，值越大模糊效果越强
  dp: 12              # 累加器分辨率，实际值为这个数/10，通常1.0-2.0
  maxRadius: 100      # 最大圆半径（像素）
  minDist: 30         # 两个圆心之间的最小距离
  minRadius: 20       # 最小圆半径（像素）
  param1: 100         # Canny边缘检测的高阈值
  param2: 25          # 累加器阈值，值越小检测到的圆越多，但误检率也越高
  roi_padding: 20     # ROI区域的额外边界大小
  edge_threshold: 80  # 边缘检测阈值，值越高边缘越少
  max_circles: 8      # 最大检测圆数量
```

### 3. HSV颜色阈值

```yaml
hsv_thresholds:
  # 红色有两个范围（横跨H通道的两端）
  red_lower_h: 0      # 红色范围1的H下限（0-180）
  red_lower_s: 100    # 红色范围1的S下限（0-255）
  red_lower_v: 100    # 红色范围1的V下限（0-255）
  red_upper_h: 10     # 红色范围1的H上限
  red_upper_s: 255    # 红色范围1的S上限
  red_upper_v: 255    # 红色范围1的V上限
  
  red_lower2_h: 160   # 红色范围2的H下限
  red_lower2_s: 100   # 红色范围2的S下限
  red_lower2_v: 100   # 红色范围2的V下限
  red_upper2_h: 180   # 红色范围2的H上限
  red_upper2_s: 255   # 红色范围2的S上限
  red_upper2_v: 255   # 红色范围2的V上限
  
  # 绿色范围
  green_lower_h: 35   # 绿色H下限
  green_lower_s: 80   # 绿色S下限
  green_lower_v: 80   # 绿色V下限
  green_upper_h: 85   # 绿色H上限
  green_upper_s: 255  # 绿色S上限
  green_upper_v: 255  # 绿色V上限
  
  # 蓝色范围
  blue_lower_h: 90    # 蓝色H下限
  blue_lower_s: 80    # 蓝色S下限
  blue_lower_v: 80    # 蓝色V下限
  blue_upper_h: 130   # 蓝色H上限
  blue_upper_s: 255   # 蓝色S上限
  blue_upper_v: 255   # 蓝色V上限
```

### 4. 静止物体识别设置
```yaml
static_object_detection:
  enable: true          # 是否启用静止物体检测
  min_area: 100         # 最小物体面积（像素）
  min_circularity: 0.7  # 最小圆度，范围0-1，越接近1要求越圆
```

### 5. YOLOv5模型配置
```yaml
yolov5_model_path: "yolov5"  # YOLOv5模型目录或权重文件(.pt)路径
use_yolo: false              # 是否使用YOLO模型，在资源受限的环境建议设为false
```

### 6. 串口配置
```yaml
serial_config:
  baudrate: 115200           # 波特率
  timeout: 1.0               # 超时时间（秒）
  reconnect_interval: 5.0    # 断线重连间隔（秒）
```

### 7. Jetson优化参数
```yaml
jetson_optimize:
  enable_gpu: true              # 是否启用GPU加速
  camera_resolution: [640, 480] # 摄像头分辨率
  frame_rate: 15                # 目标帧率
  enable_tensorrt: false        # 是否启用TensorRT加速
```

## 串口通信协议

系统使用以下协议与下位机通信：

**数据帧格式**：
```
0xAA [长度] [数据] [CRC8] 0x55
```

- **0xAA**: 帧起始标志
- **[长度]**: 数据长度（1字节）
- **[数据]**: 实际数据，格式如下：
  - 二维码数据: 原始内容
  - 圆环数据: `C:x,y,color`（例如：`C:320.5,240.2,red`）
  - 物块数据: `M:x,y,color`（例如：`M:150.0,200.0,blue`）
- **[CRC8]**: CRC8校验值（使用多项式0x31）
- **[0x55]**: 帧结束标志

## 常见问题

1. **如何调整颜色阈值？**
   - 颜色检测是基于HSV颜色空间的，您需要根据实际光照条件调整HSV阈值
   - 使用`python test_circle_color.py`查看检测效果，调整配置文件中的值

2. **如何提高圆环检测精度？**
   - 调整`hough_params`中的参数，特别是`param2`和`edge_threshold`
   - 确保摄像头图像清晰、光照均匀

3. **系统运行缓慢怎么办？**
   - 在Jetson Nano上，关闭YOLO模式（`use_yolo: false`）
   - 降低分辨率和帧率
   - 关闭调试模式运行

## 故障排除

1. **串口连接问题**
   - 检查串口名称是否正确（Windows系统通常为COMx，Linux系统为/dev/ttyUSBx）
   - 确认硬件连接无误，串口参数配置正确

2. **摄像头无法打开**
   - 确认摄像头ID正确（通常内置摄像头为0，外置摄像头为1）
   - 使用`v4l2-ctl --list-devices`检查可用摄像头

3. **模型加载失败**
   - 确认YOLOv5路径配置正确
   - 检查是否已下载权重文件

4. **识别精度不理想**
   - 调整光照条件
   - 微调HSV颜色阈值和霍夫圆参数
   - 考虑创建一个自定义的YOLOv5模型 