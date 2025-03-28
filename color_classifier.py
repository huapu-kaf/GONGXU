#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

class ColorClassifier:
    """物料颜色分类器"""
    
    def __init__(self):
        """初始化颜色分类器，定义HSV阈值范围"""
        # 定义三种颜色的HSV阈值范围（在H通道上使用OpenCV的0-180范围）
        # 红色的HSV范围比较特殊，横跨H通道的两端，需要两个范围
        self.color_ranges = {
            "red": [
                {"lower": np.array([0, 100, 100]), "upper": np.array([10, 255, 255])},
                {"lower": np.array([160, 100, 100]), "upper": np.array([180, 255, 255])}
            ],
            "green": [
                {"lower": np.array([35, 100, 100]), "upper": np.array([85, 255, 255])}
            ],
            "blue": [
                {"lower": np.array([100, 100, 100]), "upper": np.array([130, 255, 255])}
            ]
        }
        
        # 形态学操作的核
        self.kernel = np.ones((5, 5), np.uint8)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像，减少噪声并增强特征
        
        Args:
            image: BGR格式输入图像
            
        Returns:
            预处理后的图像
        """
        # 应用高斯模糊降噪
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        return hsv
    
    def create_mask_for_color(self, hsv_image: np.ndarray, color: str) -> np.ndarray:
        """
        为指定颜色创建二值掩码
        
        Args:
            hsv_image: HSV格式图像
            color: 颜色名称 ("red", "green" 或 "blue")
            
        Returns:
            二值掩码，颜色区域为白色(255)，其他为黑色(0)
        """
        if color not in self.color_ranges:
            raise ValueError(f"不支持的颜色: {color}")
        
        # 初始化空掩码
        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        
        # 对每个颜色范围创建掩码并合并
        for range_dict in self.color_ranges[color]:
            lower = range_dict["lower"]
            upper = range_dict["upper"]
            
            # 创建当前范围的掩码
            current_mask = cv2.inRange(hsv_image, lower, upper)
            
            # 合并掩码
            mask = cv2.bitwise_or(mask, current_mask)
        
        # 应用形态学闭运算（先膨胀后腐蚀）消除小噪点
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        
        return mask
    
    def find_largest_contour(self, mask: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        在掩码中找到最大的轮廓
        
        Args:
            mask: 二值掩码图像
            
        Returns:
            (最大轮廓, 轮廓面积占比) 元组，如果没有找到轮廓，返回 (None, 0.0)
        """
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, 0.0
        
        # 找到最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 计算最大轮廓面积
        area = cv2.contourArea(largest_contour)
        
        # 计算面积占整个图像的比例作为置信度
        total_area = mask.shape[0] * mask.shape[1]
        confidence = min(area / total_area * 5, 1.0)  # 缩放因子5使得占比较小的物体也能获得较高置信度
        
        return largest_contour, confidence
    
    def classify(self, image: np.ndarray) -> Dict[str, Any]:
        """
        对图像中的物料进行颜色分类
        
        Args:
            image: BGR格式输入图像
            
        Returns:
            分类结果字典：
            {
                "detected": 是否检测到物料,
                "color": 物料颜色名称,
                "center": 物料中心坐标 [x, y],
                "confidence": 置信度 (0.0-1.0)
            }
        """
        # 初始化结果
        result = {
            "detected": False,
            "color": "",
            "center": [0, 0],
            "confidence": 0.0
        }
        
        # 预处理图像
        hsv_image = self.preprocess_image(image)
        
        # 对每种颜色进行检测
        best_color = None
        best_contour = None
        best_confidence = 0.0
        
        for color in self.color_ranges.keys():
            # 创建当前颜色的掩码
            mask = self.create_mask_for_color(hsv_image, color)
            
            # 找到最大轮廓
            contour, confidence = self.find_largest_contour(mask)
            
            # 更新最佳结果
            if contour is not None and confidence > best_confidence:
                best_color = color
                best_contour = contour
                best_confidence = confidence
        
        # 检查是否找到了有效物料
        if best_contour is not None and best_confidence > 0.1:  # 最小置信度阈值
            # 计算物料中心
            M = cv2.moments(best_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # 更新结果
                result["detected"] = True
                result["color"] = best_color
                result["center"] = [cx, cy]
                result["confidence"] = best_confidence
        
        return result


if __name__ == "__main__":
    # 简单测试代码
    classifier = ColorClassifier()
    
    # 从摄像头读取图像
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 分类物料颜色
        result = classifier.classify(frame)
        
        # 可视化结果
        if result["detected"]:
            center = tuple(map(int, result["center"]))
            color_name = result["color"]
            confidence = result["confidence"]
            
            # 绘制物料中心和颜色标签
            color_bgr = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}.get(color_name, (255, 255, 255))
            cv2.circle(frame, center, 20, color_bgr, 2)
            cv2.putText(frame, f"{color_name}: {confidence:.2f}", 
                      (center[0] - 40, center[1] - 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
            
            print(f"检测到{color_name}物料，置信度: {confidence:.2f}")
        
        # 显示图像
        cv2.imshow("Color Classifier", frame)
        
        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows() 