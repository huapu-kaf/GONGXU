#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar
from PIL import Image
import time
from typing import Dict, List, Tuple, Any, Optional

class QRCodeDetector:
    """二维码识别模块"""
    
    def __init__(self, retry_count: int = 3, contrast_factor: float = 1.5):
        """
        初始化二维码检测器
        
        Args:
            retry_count: 检测失败时的最大重试次数
            contrast_factor: 图像对比度增强因子
        """
        self.retry_count = retry_count
        self.contrast_factor = contrast_factor
        
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        增强图像对比度以提高二维码识别率
        
        Args:
            image: 输入图像
            
        Returns:
            增强后的图像
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 增强对比度
        alpha = self.contrast_factor  # 对比度控制
        beta = 10  # 亮度控制
        enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
        
        # 应用高斯模糊降噪
        enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # 自适应阈值分割
        enhanced = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return enhanced
    
    def find_qr_contours(self, image: np.ndarray) -> List[np.ndarray]:
        """
        查找图像中可能的二维码轮廓
        
        Args:
            image: 输入图像（灰度或二值图）
            
        Returns:
            可能的二维码轮廓列表
        """
        # 查找轮廓
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 过滤掉太小的轮廓
        min_area = 1000
        possible_qr_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # 计算轮廓的近似多边形
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # 二维码通常具有4个角点或更多
                if len(approx) >= 4:
                    possible_qr_contours.append(approx)
                    
        return possible_qr_contours
    
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        检测图像中的二维码
        
        Args:
            image: 输入BGR格式图像
            
        Returns:
            包含二维码信息的字典：
            {
                "detected": 是否检测到二维码,
                "content": 二维码内容,
                "position": 二维码四角坐标,
                "confidence": 置信度
            }
        """
        result = {
            "detected": False,
            "content": "",
            "position": [],
            "confidence": 0.0
        }
        
        # 进行多次尝试
        for attempt in range(self.retry_count):
            # 尝试直接解码
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            codes = pyzbar.decode(gray)
            
            if codes:
                # 成功解码
                qr_code = codes[0]  # 取第一个检测到的码
                try:
                    # 解析内容（pyzbar已经处理了UTF-8解码）
                    content = qr_code.data.decode('utf-8')
                    
                    # 获取位置信息
                    points = qr_code.polygon
                    if points is None or len(points) < 4:
                        # 如果polygon为空，尝试使用rect
                        rect = qr_code.rect
                        # 从矩形创建四个角点
                        points = [
                            (rect.left, rect.top),
                            (rect.left + rect.width, rect.top),
                            (rect.left + rect.width, rect.top + rect.height),
                            (rect.left, rect.top + rect.height)
                        ]
                        
                    # 转换为数组
                    position = [[p.x, p.y] for p in points] if hasattr(points[0], 'x') else points
                    
                    result["detected"] = True
                    result["content"] = content
                    result["position"] = position
                    result["confidence"] = 1.0
                    
                    # 检测成功，直接返回
                    return result
                        
                except UnicodeDecodeError:
                    # 解码失败，使用原始二进制数据
                    result["content"] = str(qr_code.data)
            
            # 首次尝试失败，增强图像再试
            if attempt < self.retry_count - 1:
                # 增强图像
                enhanced_image = self.enhance_image(image)
                
                # 用增强的图像进行二维码扫描
                enhanced_codes = pyzbar.decode(enhanced_image)
                
                if enhanced_codes:
                    try:
                        # 解析内容
                        qr_code = enhanced_codes[0]
                        content = qr_code.data.decode('utf-8')
                        
                        # 获取位置信息
                        points = qr_code.polygon
                        if points is None or len(points) < 4:
                            # 如果polygon为空，尝试使用rect
                            rect = qr_code.rect
                            # 从矩形创建四个角点
                            points = [
                                (rect.left, rect.top),
                                (rect.left + rect.width, rect.top),
                                (rect.left + rect.width, rect.top + rect.height),
                                (rect.left, rect.top + rect.height)
                            ]
                            
                        # 转换为数组
                        position = [[p.x, p.y] for p in points] if hasattr(points[0], 'x') else points
                        
                        result["detected"] = True
                        result["content"] = content
                        result["position"] = position
                        result["confidence"] = 0.8  # 通过增强检测到的置信度略低
                        
                        # 检测成功，直接返回
                        return result
                            
                    except UnicodeDecodeError:
                        result["content"] = str(enhanced_codes[0].data)
                        
                # 调整对比度因子，为下次尝试做准备
                self.contrast_factor += 0.5
        
        # 重置对比度因子
        self.contrast_factor = 1.5
                
        return result
    
    
if __name__ == "__main__":
    # 简单测试代码
    detector = QRCodeDetector()
    
    # 从摄像头读取图像
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 检测二维码
        result = detector.detect(frame)
        
        # 可视化结果
        if result["detected"]:
            # 绘制二维码边界框
            points = np.array(result["position"], dtype=np.int32)
            cv2.polylines(frame, [points], True, (0, 255, 0), 2)
            
            # 显示二维码内容
            cv2.putText(frame, result["content"][:20], 
                     (points[0][0], points[0][1] - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            print(f"检测到二维码: {result['content']}")
        
        # 显示图像
        cv2.imshow("QR Code Detector", frame)
        
        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows() 