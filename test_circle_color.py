#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
静止物块和彩色圆环检测测试脚本
"""

import cv2
import numpy as np
import argparse
from circle_detector import CircleDetector
from color_classifier import ColorClassifier

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="圆环颜色检测测试")
    parser.add_argument("--camera", type=int, default=0, help="摄像头ID")
    parser.add_argument("--width", type=int, default=640, help="图像宽度")
    parser.add_argument("--height", type=int, default=480, help="图像高度")
    parser.add_argument("--image", type=str, help="测试图像路径（如果不使用摄像头）")
    args = parser.parse_args()
    
    # 初始化圆环检测器（不使用YOLO）
    circle_detector = CircleDetector(
        confidence_threshold=0.4, 
        use_yolo=False
    )
    
    # 改进霍夫圆检测参数
    circle_detector.hough_params.update({
        "minDist": 30,
        "param2": 25,
        "minRadius": 20,
        "maxRadius": 100,
        "edge_threshold": 80
    })
    
    # 初始化颜色分类器
    color_classifier = ColorClassifier()
    
    # HSV颜色范围（用于圆环/物块颜色检测）
    color_ranges = {
        "red": [
            ((0, 100, 100), (10, 255, 255)),
            ((160, 100, 100), (180, 255, 255))
        ],
        "green": [((35, 80, 80), (85, 255, 255))],
        "blue": [((90, 80, 80), (130, 255, 255))]
    }
    
    # 更新颜色检测器的颜色范围
    circle_detector.color_ranges = color_ranges
    color_classifier.color_ranges = {
        "red": [color_ranges["red"][0], color_ranges["red"][1]],
        "green": color_ranges["green"][0],
        "blue": color_ranges["blue"][0]
    }
    
    # 打开视频源
    if args.image:
        # 如果提供了图像路径，使用静态图像
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"无法读取图像: {args.image}")
            return
            
        # 调整图像大小
        frame = cv2.resize(frame, (args.width, args.height))
        
        # 处理单张图像
        process_frame(frame, circle_detector, color_classifier)
        
        # 等待按键
        cv2.waitKey(0)
    else:
        # 否则使用摄像头
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        
        if not cap.isOpened():
            print(f"无法打开摄像头 {args.camera}")
            return
            
        # 创建窗口
        cv2.namedWindow("Circle Color Detection", cv2.WINDOW_NORMAL)
        
        print("按ESC键退出")
        
        while True:
            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                print("无法获取图像")
                break
                
            # 处理当前帧
            process_frame(frame, circle_detector, color_classifier)
            
            # 按ESC键退出
            if cv2.waitKey(1) == 27:
                break
                
        # 释放资源
        cap.release()
        
    cv2.destroyAllWindows()

def process_frame(frame, circle_detector, color_classifier):
    """处理一帧图像"""
    # 复制原始图像用于绘制结果
    result_image = frame.copy()
    
    # 检测圆环
    circle_result = circle_detector.detect_with_hough(frame)
    
    # 检测静止物块颜色
    blobs = detect_color_blobs(frame, color_classifier)
    
    # 绘制检测结果
    # 1. 绘制圆环
    if circle_result["detected"]:
        center = tuple(map(int, circle_result["position"]))
        radius = int(circle_result["radius"])
        color_name = circle_result["color"]
        
        # 根据颜色选择BGR值
        circle_color = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "unknown": (128, 128, 128)
        }.get(color_name, (255, 255, 255))
        
        # 绘制圆环和中心点
        cv2.circle(result_image, center, radius, circle_color, 2)
        cv2.circle(result_image, center, 4, (0, 0, 255), -1)
        
        # 显示圆环信息
        text = f"{color_name} r:{radius}"
        cv2.putText(result_image, text, 
                   (center[0] - 40, center[1] - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, circle_color, 2)
    
    # 2. 绘制颜色物块
    for blob in blobs:
        center = tuple(map(int, blob["center"]))
        color_name = blob["color"]
        area = blob["area"]
        
        # 根据颜色选择BGR值
        blob_color = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0)
        }.get(color_name, (255, 255, 255))
        
        # 绘制物块中心和信息
        cv2.circle(result_image, center, 5, blob_color, -1)
        cv2.putText(result_image, f"{color_name}", 
                   (center[0] - 20, center[1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, blob_color, 2)
    
    # 显示结果
    cv2.imshow("Circle Color Detection", result_image)
    
    # 显示调试信息（可选）
    if "edges" in circle_detector.debug_info and circle_detector.debug_info["edges"] is not None:
        cv2.imshow("Edges", circle_detector.debug_info["edges"])
    
    for color, mask in circle_detector.debug_info.get("color_masks", {}).items():
        if mask is not None:
            cv2.imshow(f"Mask: {color}", mask)

def detect_color_blobs(image, color_classifier):
    """
    检测图像中的彩色物块
    
    Args:
        image: BGR格式图像
        color_classifier: 颜色分类器
        
    Returns:
        检测到的物块列表，每个物块包含中心位置、颜色和面积
    """
    # 转换为HSV色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 存储检测到的物块
    blobs = []
    
    # 对每种颜色进行检测
    for color, ranges in color_classifier.color_ranges.items():
        # 创建颜色掩码
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if isinstance(ranges, list) and len(ranges) == 2 and isinstance(ranges[0], tuple):
            # 处理红色（有两个范围）
            for lower, upper in ranges:
                range_mask = cv2.inRange(hsv, lower, upper)
                mask = cv2.bitwise_or(mask, range_mask)
        else:
            # 处理其他颜色
            lower, upper = ranges
            mask = cv2.inRange(hsv, lower, upper)
        
        # 形态学操作去噪
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 处理每个轮廓
        for contour in contours:
            # 计算面积，过滤小区域
            area = cv2.contourArea(contour)
            if area < 200:  # 最小面积阈值
                continue
                
            # 计算中心点
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # 添加到物块列表
                blobs.append({
                    "center": (cx, cy),
                    "color": color,
                    "area": area
                })
    
    return blobs

if __name__ == "__main__":
    main() 