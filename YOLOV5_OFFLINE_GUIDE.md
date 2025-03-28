# YOLOv5离线使用教程

## 概述

本教程将指导您如何在离线环境中使用YOLOv5模型进行圆环检测。由于网络连接问题，我们需要手动下载YOLOv5模型并在本地使用。

## 步骤一：下载YOLOv5源码

1. 在有互联网连接的计算机上，访问YOLOv5官方仓库：https://github.com/ultralytics/yolov5
2. 点击绿色的"Code"按钮，然后点击"Download ZIP"下载源码压缩包
3. 将下载的ZIP文件解压到一个文件夹，例如命名为"yolov5"

## 步骤二：下载预训练模型权重

1. 在有互联网连接的计算机上，访问：https://github.com/ultralytics/yolov5/releases
2. 下载YOLOv5s模型权重文件（yolov5s.pt）：
   - 直接下载链接：https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
3. 将下载的权重文件放到刚才解压的"yolov5"文件夹中的"weights"子文件夹中
   - 如果"weights"文件夹不存在，请创建它

## 步骤三：安装依赖

在项目根目录下运行以下命令安装必要的依赖项：

```bash
pip install -r requirements.txt
```

## 步骤四：配置使用离线模型

1. 将准备好的"yolov5"文件夹复制到项目根目录下
2. 确保"vision_params.yaml"文件中已正确配置YOLOv5模型路径：

```yaml
# YOLOv5模型配置
yolov5_model_path: "yolov5"  # 指向yolov5文件夹
```

如果您只有模型权重文件（.pt文件），也可以直接指定权重文件路径：

```yaml
# YOLOv5模型配置
yolov5_model_path: "path/to/yolov5s.pt"  # 指向权重文件
```

## 步骤五：运行程序

现在您可以正常运行程序，系统将使用本地模型而不是尝试从网络下载：

```bash
python main_enhanced.py --no-yolo
```

## 常见问题解答

### Q: 如何确认系统使用了离线模型？

A: 启动系统时，日志会显示"离线模式：从xxx加载模型..."的信息。

### Q: 如果仍然无法加载模型，该怎么办？

A: 系统会自动退回到仅使用霍夫圆检测模式，虽然准确度可能会降低，但仍然可以运行。

### Q: 可以使用其他版本的YOLOv5模型吗？

A: 是的，您可以下载并使用其他版本的YOLOv5模型（如yolov5m.pt、yolov5l.pt等），只需更新"vision_params.yaml"文件中的路径即可。

### Q: 如何训练自己的模型用于圆环检测？

A: 您可以按照YOLOv5官方教程准备数据集并训练自己的模型，然后在"vision_params.yaml"中指向您的自定义模型权重文件。 

## 系统优化总结

我已经为您的视觉识别系统进行了以下优化：

1. **增强串口通信功能**：
   - 支持发送所有检测结果（二维码、圆环、物块）
   - 改进数据帧格式：
     - 二维码数据: 原始内容
     - 圆环数据: `C:x,y,radius,confidence`
     - 物块数据: `M:x,y,color,confidence`
   - 添加高效的去重机制，减少不必要的通信

2. **无YOLO模式支持**：
   - 增强CircleDetector类，增加`use_yolo`参数
   - 添加`--no-yolo`命令行参数
   - 在配置文件中添加`use_yolo`开关选项
   - 大幅提升霍夫圆检测算法的准确度：
     - 自适应对比度增强
     - 高精度边缘检测
     - 基于边缘重合度的置信度计算
     - 更灵活的霍夫圆参数配置

3. **系统配置优化**：
   - 添加霍夫圆检测的高级参数：
     - `edge_threshold`: 边缘检测阈值
     - `max_circles`: 最大检测圆数量
   - 增加串口配置项，方便调整通信参数

## 使用方法

1. **启动无YOLO模式**：
   ```
   python main_enhanced.py --no-yolo
   ```

2. **调整配置提高检测质量**：
   在`vision_params.yaml`中优化霍夫参数：
   ```yaml
   hough_params:
     param2: 25       # 减小可检测更多潜在圆
     edge_threshold: 120  # 调整以匹配环境光照
     max_circles: 5   # 增加检测圆数量
   ```

3. **使用串口发送所有检测结果**：
   ```
   python main_enhanced.py --port COM3
   ```

系统现在可以在不使用YOLO的情况下工作，同时通过串口发送所有检测到的信息（二维码内容、圆环位置、物块位置）。

还可以根据实际环境和设备条件调整参数，以获得最佳的检测效果。更多详细说明请参考更新后的README_SERIAL_QR.md文档。 