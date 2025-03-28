@echo off
echo ===== 安装智能物流机器人视觉识别系统依赖 =====
echo.

REM 检查Python是否已安装
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未检测到Python，请先安装Python 3.8或更高版本
    pause
    exit /b 1
)

echo 安装基本依赖...
pip install opencv-python==4.8.0.76 numpy==1.24.3 pillow==10.0.0 pyzbar==0.1.9

echo.
echo 安装PyTorch (CPU版本)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo.
echo 安装其他依赖...
pip install pandas==2.0.3 pyyaml==6.0 psutil==5.9.5

echo.
echo 依赖安装完成！
echo 现在可以运行 run_test.bat 测试系统了。
pause 