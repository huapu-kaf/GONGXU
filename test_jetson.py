#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Jetson Nano 安装验证脚本
用于测试视觉识别系统的各个组件是否正常工作
"""

import os
import sys
import cv2
import numpy as np
import torch
import time
import argparse

def test_camera(camera_id=0, resolution=(640, 480)):
    """测试摄像头是否正常工作"""
    print("测试摄像头...")
    
    try:
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        if not cap.isOpened():
            print("❌ 无法打开摄像头")
            return False
            
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取图像")
            cap.release()
            return False
            
        # 显示图像
        cv2.imshow("Camera Test", frame)
        cv2.waitKey(1000)
        
        # 保存测试图像
        cv2.imwrite("camera_test.jpg", frame)
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        
        print("✅ 摄像头测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 摄像头测试失败: {e}")
        return False

def test_opencv():
    """测试OpenCV是否正常工作"""
    print("测试OpenCV...")
    
    try:
        # 创建测试图像
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(img, (100, 100), 50, (0, 0, 255), 2)
        cv2.putText(img, "OpenCV", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示图像
        cv2.imshow("OpenCV Test", img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        
        # 检查版本
        print(f"✅ OpenCV版本: {cv2.__version__}")
        return True
        
    except Exception as e:
        print(f"❌ OpenCV测试失败: {e}")
        return False

def test_pytorch():
    """测试PyTorch是否正常工作"""
    print("测试PyTorch...")
    
    try:
        # 检查PyTorch是否可用
        x = torch.rand(5, 3)
        
        # 检查CUDA是否可用
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA可用，设备数量: {device_count}, 设备名称: {device_name}")
        else:
            print("⚠️ CUDA不可用，将使用CPU模式")
        
        print(f"✅ PyTorch版本: {torch.__version__}")
        return True
        
    except Exception as e:
        print(f"❌ PyTorch测试失败: {e}")
        return False

def test_yolov5():
    """测试YOLOv5是否正常工作"""
    print("测试YOLOv5...")
    
    try:
        # 检查YOLOv5目录是否存在
        if not os.path.exists("yolov5"):
            print("❌ YOLOv5目录不存在")
            return False
            
        # 检查权重文件是否存在
        if not os.path.exists("yolov5/weights/yolov5s.pt"):
            print("❌ YOLOv5权重文件不存在")
            return False
            
        print("✅ YOLOv5文件检查通过")
        return True
        
    except Exception as e:
        print(f"❌ YOLOv5测试失败: {e}")
        return False

def test_config():
    """测试配置文件是否正常"""
    print("测试配置文件...")
    
    try:
        # 检查配置文件是否存在
        if not os.path.exists("vision_params_jetson.yaml"):
            print("❌ 配置文件不存在")
            return False
            
        # 检查配置文件大小
        file_size = os.path.getsize("vision_params_jetson.yaml")
        if file_size < 100:
            print("❌ 配置文件可能不完整")
            return False
            
        print("✅ 配置文件检查通过")
        return True
        
    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")
        return False

def test_performance():
    """测试系统性能"""
    print("测试系统性能...")
    
    try:
        # 创建测试图像
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 测试霍夫圆检测性能
        start_time = time.time()
        for _ in range(10):
            # 添加一些随机圆
            for i in range(5):
                center_x = np.random.randint(100, 540)
                center_y = np.random.randint(100, 380)
                radius = np.random.randint(20, 50)
                cv2.circle(img, (center_x, center_y), radius, (255, 255, 255), 2)
            
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 应用高斯模糊
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 霍夫圆检测
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=30,
                param1=100,
                param2=30,
                minRadius=20,
                maxRadius=100
            )
        
        elapsed_time = time.time() - start_time
        fps = 10 / elapsed_time
        
        print(f"✅ 霍夫圆检测性能: {fps:.2f} FPS")
        return True
        
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Jetson Nano 安装验证脚本")
    parser.add_argument("--camera", type=int, default=0, help="摄像头ID")
    parser.add_argument("--width", type=int, default=640, help="图像宽度")
    parser.add_argument("--height", type=int, default=480, help="图像高度")
    args = parser.parse_args()
    
    print("===== Jetson Nano 视觉识别系统安装验证 =====")
    
    # 运行所有测试
    tests = [
        test_opencv,
        test_pytorch,
        test_yolov5,
        test_config,
        lambda: test_camera(args.camera, (args.width, args.height)),
        test_performance
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print("-" * 50)
    
    # 打印总结
    print("\n===== 测试结果汇总 =====")
    all_passed = all(results)
    
    if all_passed:
        print("✅ 所有测试通过！系统已准备就绪。")
    else:
        print("⚠️ 部分测试未通过，请检查上述错误信息。")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 