#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time

def test_camera():
    """测试摄像头是否正常工作"""
    print("正在测试摄像头...")
    
    # 尝试打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return False
    
    # 读取一帧
    ret, frame = cap.read()
    
    if not ret:
        print("错误：无法读取摄像头图像")
        cap.release()
        return False
    
    # 显示图像尺寸
    h, w = frame.shape[:2]
    print(f"摄像头工作正常，图像尺寸: {w}x{h}")
    
    # 显示图像
    cv2.imshow("Camera Test", frame)
    cv2.waitKey(2000)  # 显示2秒
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    return True

def test_color_detection():
    """测试简单的颜色检测功能"""
    print("正在测试颜色检测...")
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return False
    
    # 读取10帧进行测试
    for i in range(10):
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # 转换到HSV空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 定义红色的HSV范围
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # 创建掩码
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # 应用掩码
        red_detection = cv2.bitwise_and(frame, frame, mask=mask)
        
        # 在原图上标记红色区域
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # 过滤小区域
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        
        # 显示原图和检测结果
        cv2.imshow("Original", frame)
        cv2.imshow("Red Detection", red_detection)
        
        key = cv2.waitKey(100)
        if key & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("颜色检测测试完成")
    return True

def main():
    """主函数"""
    print("=== 视觉系统基础测试 ===")
    
    # 测试摄像头
    camera_ok = test_camera()
    
    if camera_ok:
        # 测试颜色检测
        test_color_detection()
    
    print("测试完成")

if __name__ == "__main__":
    main() 