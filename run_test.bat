@echo off
echo ===== 智能物流机器人视觉识别系统测试 =====
echo.

REM 检查Python是否已安装
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未检测到Python，请先安装Python 3.8或更高版本
    pause
    exit /b 1
)

echo 1. 检查摄像头和基本视觉功能
python test_camera.py
if %errorlevel% neq 0 (
    echo 摄像头测试失败，尝试不同的摄像头ID...
    echo.
    python test_camera.py --camera=1
    if %errorlevel% neq 0 (
        echo 摄像头测试仍然失败，请检查您的摄像头连接
        pause
        exit /b 1
    )
)

echo.
echo 2. 运行简化版视觉系统（不使用YOLOv5模型）
echo 按q键退出程序
python main_simple.py
if %errorlevel% neq 0 (
    echo 简化版系统运行失败，请检查错误信息
    pause
    exit /b 1
)

echo.
echo 测试完成！
pause 