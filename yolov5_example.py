#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLOv5离线模式示例脚本
此脚本展示如何使用离线模式的YOLOv5模型检测图像中的圆环
"""

import cv2
import os
import sys
import argparse
from circle_detector import CircleDetector

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv5离线模式示例')
    parser.add_argument('--image', type=str, default='test.jpg', help='要检测的图像路径')
    parser.add_argument('--model', type=str, default='yolov5', help='YOLOv5模型路径或权重文件路径')
    parser.add_argument('--conf', type=float, default=0.4, help='置信度阈值')
    parser.add_argument('--output', type=str, default='output.jpg', help='输出图像路径')
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 检查输入图像是否存在
    if not os.path.isfile(args.image):
        print(f"错误：找不到输入图像文件 {args.image}")
        return
    
    # 检查模型路径是否存在
    if not (os.path.isdir(args.model) or (os.path.isfile(args.model) and args.model.endswith('.pt'))):
        print(f"错误：无效的模型路径 {args.model}")
        return
    
    print(f"正在使用离线模式加载YOLOv5模型：{args.model}")
    
    # 创建圆环检测器实例（使用离线模式）
    detector = CircleDetector(
        confidence_threshold=args.conf,
        offline_mode=True,
        model_path=args.model
    )
    
    # 读取输入图像
    print(f"正在读取图像：{args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"错误：无法读取图像 {args.image}")
        return
    
    # 进行圆环检测
    print("正在检测圆环...")
    result = detector.detect(image)
    
    # 打印检测结果
    if result["detected"]:
        print(f"检测到圆环:")
        print(f"- 位置: {result['position']}")
        print(f"- 半径: {result['radius']}")
        print(f"- 置信度: {result['confidence']}")
        print(f"- 使用方法: {result['method']}")
        
        # 在图像上标记检测结果
        cx, cy = int(result["position"][0]), int(result["position"][1])
        radius = int(result["radius"])
        
        # 绘制圆和圆心
        cv2.circle(image, (cx, cy), radius, (0, 255, 0), 2)
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
        
        # 添加文本标签
        label = f"{result['method']}: {result['confidence']:.2f}"
        cv2.putText(image, label, (cx - 20, cy - radius - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        print("未检测到圆环")
    
    # 保存输出图像
    print(f"正在保存结果图像到：{args.output}")
    cv2.imwrite(args.output, image)
    print("处理完成！")

if __name__ == "__main__":
    main() 