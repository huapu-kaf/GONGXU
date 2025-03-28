@echo off
echo YOLOv5离线模式示例启动脚本
echo =================================

REM 检查Python是否已安装
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo 错误: 未找到Python安装，请安装Python 3.8或更高版本。
    goto :end
)

REM 确认YOLOv5文件夹存在
if not exist yolov5 (
    echo 警告: 未找到yolov5文件夹。
    echo 请确保您已按照YOLOV5_OFFLINE_GUIDE.md中的说明下载并准备好YOLOv5。
    
    set /p answer=是否仍要继续? (Y/N): 
    if /i "%answer%" neq "Y" goto :end
)

REM 检查测试图像是否存在，如果不存在则提示用户提供
if not exist test.jpg (
    echo 警告: 未找到测试图像文件(test.jpg)。
    
    set /p image_path=请输入测试图像的路径: 
    if "%image_path%"=="" (
        echo 错误: 未提供图像路径，程序将退出。
        goto :end
    )
) else (
    set image_path=test.jpg
)

REM 检查是否提供了YOLOv5模型路径
if "%1"=="" (
    if exist yolov5 (
        set model_path=yolov5
    ) else (
        set /p model_path=请输入YOLOv5模型路径或权重文件(.pt)路径: 
        if "%model_path%"=="" (
            echo 错误: 未提供模型路径，程序将退出。
            goto :end
        )
    )
) else (
    set model_path=%1
)

REM 运行YOLOv5示例脚本
echo 正在使用以下参数运行:
echo - 图像: %image_path%
echo - 模型: %model_path%
echo.
echo 正在启动...
python yolov5_example.py --image "%image_path%" --model "%model_path%" --output "result.jpg"

if %ERRORLEVEL% neq 0 (
    echo.
    echo 程序运行出错。请检查错误信息并修复问题。
) else (
    echo.
    echo 处理完成！结果已保存为result.jpg
    
    REM 询问是否打开结果图像
    set /p open_result=是否打开结果图像? (Y/N): 
    if /i "%open_result%"=="Y" (
        start result.jpg
    )
)

:end
echo.
echo 按任意键退出...
pause > nul 