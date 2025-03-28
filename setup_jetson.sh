#!/bin/bash

# 视觉识别系统 - Jetson Nano 安装脚本
echo "===== 视觉识别系统部署脚本 - Jetson Nano版 ====="
echo "此脚本将安装所有必要的依赖和库"

# 创建日志文件
LOG_FILE="setup_log.txt"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "安装日志将保存到: $LOG_FILE"

# 创建必要的目录
echo "创建必要的文件夹..."
mkdir -p cache logs configs vision_system models/weights

# 更新系统
echo "=== 步骤1: 更新系统包 ==="
sudo apt-get update
sudo apt-get upgrade -y

# 安装基础依赖
echo "=== 步骤2: 安装基础依赖 ==="
sudo apt-get install -y build-essential cmake pkg-config git python3-dev python3-pip libgtk-3-dev 
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev gfortran
sudo apt-get install -y python3-numpy python3-matplotlib

# 为pip安装设置国内源（可选，如需使用请取消注释）
# echo "=== 设置pip国内源 ==="
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 升级pip
echo "=== 步骤3: 升级pip ==="
python3 -m pip install --upgrade pip

# 安装PyTorch (Jetson Nano专用版本)
echo "=== 步骤4: 安装PyTorch (Jetson Nano版) ==="
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
pip3 install Cython
pip3 install numpy torch-1.8.0-cp36-cp36m-linux_aarch64.whl

# 安装torchvision
echo "=== 步骤5: 安装torchvision ==="
sudo apt-get install -y libjpeg-dev zlib1g-dev
git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision
cd torchvision
python3 setup.py install
cd ..
rm -rf torchvision

# 安装OpenCV（使用pip，避免耗时的编译过程）
echo "=== 步骤6: 安装OpenCV ==="
pip3 install opencv-python==4.5.3.56

# 下载YOLOv5（如果使用离线模式）
echo "=== 步骤7: 下载YOLOv5 ==="
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v6.1  # 使用稳定版本
pip3 install -r requirements.txt
# 下载权重文件
mkdir -p weights
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -P weights/
cd ..

# 安装其他依赖
echo "=== 步骤8: 安装项目依赖 ==="
pip3 install pillow==9.0.1 
pip3 install pyzbar==0.1.9
pip3 install pandas==1.3.5
pip3 install pyyaml==6.0
pip3 install psutil==5.9.0
pip3 install matplotlib==3.5.1
pip3 install seaborn==0.11.2
pip3 install pyserial==3.5

# 创建Jetson Nano优化的配置文件
echo "=== 步骤9: 创建优化配置 ==="
cat > vision_params_jetson.yaml << EOL
confidence_thresholds:
  circle_confidence: 40
  color_confidence: 50
  qr_confidence: 50
hough_params:
  blur_ksize: 5
  dp: 12
  maxRadius: 100
  minDist: 30
  minRadius: 20
  param1: 100
  param2: 25
  roi_padding: 20
  edge_threshold: 80
  max_circles: 8
hsv_thresholds:
  red_lower_h: 0
  red_lower_s: 100
  red_lower_v: 100
  red_upper_h: 10
  red_upper_s: 255
  red_upper_v: 255
  red_lower2_h: 160
  red_lower2_s: 100
  red_lower2_v: 100
  red_upper2_h: 180
  red_upper2_s: 255
  red_upper2_v: 255
  green_lower_h: 35
  green_lower_s: 80
  green_lower_v: 80
  green_upper_h: 85
  green_upper_s: 255
  green_upper_v: 255
  blue_lower_h: 90
  blue_lower_s: 80
  blue_lower_v: 80
  blue_upper_h: 130
  blue_upper_s: 255
  blue_upper_v: 255

# 静止物体识别设置
static_object_detection:
  enable: true
  min_area: 100        # 最小面积（像素）
  min_circularity: 0.7 # 最小圆度
  
# YOLOv5模型配置
yolov5_model_path: "yolov5"  # 这里填写本地YOLOv5模型目录或权重文件路径
use_yolo: false  # 在Jetson Nano上，建议关闭YOLO以提高性能

# 串口配置
serial_config:
  baudrate: 115200
  timeout: 1.0
  reconnect_interval: 5.0

# Jetson Nano优化参数
jetson_optimize:
  enable_gpu: true
  camera_resolution: [640, 480]
  frame_rate: 15
  enable_tensorrt: false
EOL

# 创建启动脚本
echo "=== 步骤10: 创建便捷启动脚本 ==="
cat > run_jetson.sh << EOL
#!/bin/bash
echo "启动视觉识别系统 - Jetson Nano版"

# 参数设置
CAMERA=0
WIDTH=640
HEIGHT=480
CONFIG="vision_params_jetson.yaml"
PORT=""

# 解析命令行参数
while [[ \$# -gt 0 ]]; do
  case \$1 in
    --camera)
      CAMERA="\$2"
      shift 2
      ;;
    --port)
      PORT="\$2"
      shift 2
      ;;
    --width)
      WIDTH="\$2"
      shift 2
      ;;
    --height)
      HEIGHT="\$2"
      shift 2
      ;;
    --debug)
      DEBUG="--debug"
      shift
      ;;
    *)
      echo "未知参数: \$1"
      shift
      ;;
  esac
done

# 构建命令
CMD="python3 main_enhanced.py --camera \$CAMERA --width \$WIDTH --height \$HEIGHT --config \$CONFIG"

# 添加可选参数
if [ ! -z "\$PORT" ]; then
  CMD="\$CMD --port \$PORT"
fi

if [ ! -z "\$DEBUG" ]; then
  CMD="\$CMD \$DEBUG"
fi

# 输出并执行命令
echo "执行: \$CMD"
\$CMD
EOL

# 增加脚本执行权限
chmod +x run_jetson.sh

echo "===== 安装完成! ====="
echo "您可以通过以下命令启动系统："
echo "./run_jetson.sh"
echo "或者指定参数："
echo "./run_jetson.sh --camera 0 --port /dev/ttyUSB0 --debug"
echo "详情请查看README.md文件" 