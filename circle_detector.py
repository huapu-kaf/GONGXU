#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional

class CircleDetector:
    """圆环定位系统"""
    
    def __init__(self, confidence_threshold: float = 0.4, nms_threshold: float = 0.45, 
                offline_mode: bool = False, model_path: Optional[str] = None,
                use_yolo: bool = True):
        """
        初始化圆环检测器
        
        Args:
            confidence_threshold: 目标检测置信度阈值
            nms_threshold: 非极大值抑制阈值
            offline_mode: 是否使用离线模式
            model_path: 预训练模型路径（离线模式下使用）
            use_yolo: 是否使用YOLO模型（False则仅使用霍夫圆变换）
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.use_yolo = use_yolo
        
        # Hough圆检测参数
        self.hough_params = {
            "dp": 1.2,               # 累加器分辨率与图像分辨率的比率
            "minDist": 30,           # 检测到的圆的最小距离
            "param1": 100,           # Canny边缘检测的高阈值
            "param2": 25,            # 累加器阈值（降低以检测更多圆）
            "minRadius": 20,         # 最小圆半径 (对应5cm) 
            "maxRadius": 100,        # 最大圆半径 (对应10cm)
            "blur_ksize": 5,         # 中值滤波核大小
            "roi_padding": 20,       # ROI裁剪的额外边界
            "edge_threshold": 80,    # Canny边缘检测阈值
            "max_circles": 8,        # 最大检测圆数量（增加以便检测多个圆环）
            "size_ranges": [         # 圆环大小范围（像素单位）
                (20, 30),            # 最小尺寸圆环
                (30, 40),
                (40, 50),
                (50, 60),
                (60, 70),
                (70, 100)            # 最大尺寸圆环
            ]
        }
        
        # 颜色阈值（HSV空间）
        self.color_ranges = {
            "red": [
                ((0, 100, 100), (10, 255, 255)),     # 红色范围1
                ((160, 100, 100), (180, 255, 255))   # 红色范围2
            ],
            "green": [((35, 80, 80), (85, 255, 255))],   # 绿色范围
            "blue": [((90, 80, 80), (130, 255, 255))]    # 蓝色范围
        }
        
        # 加载YOLOv5模型（如果使用）
        self.model = None
        if use_yolo:
            print("正在加载YOLOv5模型...")
            try:
                if offline_mode and model_path:
                    # 离线模式：从本地文件加载模型
                    print(f"离线模式：从{model_path}加载模型...")
                    import sys
                    import os
                    
                    # 如果提供的是目录路径，添加到系统路径
                    if os.path.isdir(model_path):
                        sys.path.insert(0, model_path)
                        from models.common import AutoShape
                        from models.experimental import attempt_load
                        
                        # 在models目录中查找预训练权重文件
                        weights_file = None
                        for file in os.listdir(os.path.join(model_path, 'weights')):
                            if file.endswith('.pt'):
                                weights_file = os.path.join(model_path, 'weights', file)
                                break
                        
                        if weights_file:
                            # 加载模型权重
                            self.model = attempt_load(weights_file)
                            self.model = AutoShape(self.model)
                        else:
                            raise FileNotFoundError("在指定目录中未找到.pt权重文件")
                    # 如果提供的是权重文件路径
                    elif os.path.isfile(model_path) and model_path.endswith('.pt'):
                        # 添加模型所在目录到系统路径
                        parent_dir = os.path.dirname(model_path)
                        if parent_dir not in sys.path:
                            sys.path.insert(0, parent_dir)
                        
                        from models.common import AutoShape
                        from models.experimental import attempt_load
                        
                        # 加载模型权重
                        self.model = attempt_load(model_path)
                        self.model = AutoShape(self.model)
                    else:
                        raise ValueError("无效的模型路径，请提供YOLOv5源码目录或权重文件(.pt)路径")
                else:
                    # 在线模式：从torch hub加载
                    self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            except Exception as e:
                print(f"加载YOLOv5模型失败: {e}")
                print("将使用仅霍夫圆检测模式...")
                self.model = None
                self.use_yolo = False
        
        if self.model and self.use_yolo:
            # 配置模型参数
            if hasattr(self.model, 'conf'):
                self.model.conf = confidence_threshold  # 置信度阈值
            if hasattr(self.model, 'iou'):
                self.model.iou = nms_threshold  # NMS IoU阈值
            if hasattr(self.model, 'classes'):
                self.model.classes = [0]  # 只检测"person"类别，可以根据实际情况修改为圆环对应的类别
            if hasattr(self.model, 'max_det'):
                self.model.max_det = 10  # 最大检测数量
            if hasattr(self.model, 'agnostic'):
                self.model.agnostic = False  # 是否使用类别无关的NMS
            
            # 设置推理设备
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
            print(f"YOLOv5模型加载完成，使用设备: {self.device}")
            
            # 预热模型
            self._warmup()
        else:
            print("使用纯霍夫圆检测模式，无需YOLO模型")
        
        # 调试变量
        self.debug_info = {
            "yolo_box": None,        # YOLO检测框
            "roi_image": None,       # ROI裁剪图像
            "processed_roi": None,   # 处理后的ROI图像
            "hough_circles": None,   # 霍夫圆检测结果
            "gray": None,            # 灰度图
            "blurred": None,         # 模糊处理图
            "enhanced": None,        # 增强图
            "edges": None,           # 边缘图
            "detected_circle": None, # 检测到的圆
            "color_masks": {}        # 颜色掩码
        }
    
    def _warmup(self):
        """预热模型，减少第一次推理延迟"""
        if self.model:
            dummy_img = torch.zeros((1, 3, 640, 640), device=self.device)
            for _ in range(2):
                _ = self.model(dummy_img)
            print("模型预热完成")
    
    def set_hough_params(self, params_dict: Dict[str, Any]):
        """
        设置霍夫圆检测参数
        
        Args:
            params_dict: 包含霍夫参数的字典
        """
        for key, value in params_dict.items():
            if key in self.hough_params:
                self.hough_params[key] = value
    
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        检测图像中的圆环，先使用YOLO进行粗定位，再用霍夫圆变换进行精定位
        
        Args:
            image: BGR格式输入图像
            
        Returns:
            检测结果字典：
            {
                "detected": 是否检测到圆环,
                "position": 圆环中心坐标 [x, y],
                "radius": 圆环半径,
                "confidence": 检测置信度,
                "method": 使用的检测方法 ("yolo", "hough", "combined"),
                "color": 圆环颜色
            }
        """
        # 初始化结果
        result = {
            "detected": False,
            "position": [0, 0],
            "radius": 0,
            "confidence": 0.0,
            "method": "none",
            "color": "unknown"
        }
        
        # 清空调试信息
        for key in self.debug_info:
            if key != "color_masks":
                self.debug_info[key] = None
        self.debug_info["color_masks"] = {}
        
        # 确保图像尺寸合适
        img_height, img_width = image.shape[:2]
        
        # 如果不使用YOLO或模型未加载，直接使用霍夫圆检测
        if not self.use_yolo or self.model is None:
            hough_result = self.detect_with_hough(image)
            if hough_result["detected"]:
                return hough_result
            else:
                return result
                
        # 1. YOLO粗定位
        yolo_result = self._detect_with_yolo(image)
        
        # 如果YOLO检测到了目标
        if yolo_result["detected"]:
            # 获取YOLO检测到的边界框
            x1, y1, x2, y2 = self._get_bbox_from_yolo(yolo_result)
            
            # 存储YOLO检测框用于调试
            self.debug_info["yolo_box"] = (x1, y1, x2, y2)
            
            # 2. 对ROI区域进行霍夫圆精定位
            # 扩大ROI区域以确保圆完全包含在内
            padding = self.hough_params["roi_padding"]
            roi_x1 = max(0, int(x1) - padding)
            roi_y1 = max(0, int(y1) - padding)
            roi_x2 = min(img_width, int(x2) + padding)
            roi_y2 = min(img_height, int(y2) + padding)
            
            # 裁剪ROI区域
            roi_image = image[roi_y1:roi_y2, roi_x1:roi_x2].copy()
            self.debug_info["roi_image"] = roi_image
            
            # 在ROI区域内进行霍夫圆检测
            hough_result = self._detect_hough_circles_in_roi(roi_image)
            
            # 如果霍夫圆检测成功
            if hough_result["detected"]:
                # 将ROI坐标系中的圆心坐标转换回原图坐标系
                cx_global = roi_x1 + hough_result["position"][0]
                cy_global = roi_y1 + hough_result["position"][1]
                
                # 检测圆环颜色
                color = self._detect_circle_color(image, cx_global, cy_global, hough_result["radius"])
                
                # 3. 融合粗定位和精定位结果
                combined_result = {
                    "detected": True,
                    "position": [cx_global, cy_global],
                    "radius": hough_result["radius"],
                    "confidence": (yolo_result["confidence"] + hough_result["confidence"]) / 2,  # 平均置信度
                    "method": "combined",
                    "color": color
                }
                
                # 更新结果
                result = combined_result
            else:
                # 如果霍夫检测失败，返回YOLO结果并尝试检测颜色
                color = self._detect_circle_color(
                    image, 
                    int(yolo_result["position"][0]), 
                    int(yolo_result["position"][1]), 
                    int(yolo_result["radius"])
                )
                yolo_result["color"] = color
                yolo_result["method"] = "yolo"
                result = yolo_result
        else:
            # 如果YOLO检测失败，尝试直接在整个图像上进行霍夫圆检测
            hough_result = self.detect_with_hough(image)
            
            if hough_result["detected"]:
                result = hough_result
        
        return result
    
    def _detect_with_yolo(self, image: np.ndarray) -> Dict[str, Any]:
        """
        使用YOLO检测圆环
        
        Args:
            image: BGR格式输入图像
            
        Returns:
            检测结果字典
        """
        # 初始化结果
        result = {
            "detected": False,
            "position": [0, 0],
            "radius": 0,
            "confidence": 0.0
        }
        
        # 如果模型未加载成功，直接返回
        if self.model is None:
            return result
        
        # 执行推理
        # YOLOv5模型输入RGB图像
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 模型推理
        try:
            detections = self.model(rgb_image, size=640)
            
            # 获取检测结果
            results = detections.pandas().xyxy[0]  # 使用 xyxy 格式 (x1, y1, x2, y2)
            
            # 检查是否有检测结果
            if not results.empty:
                # 取置信度最高的检测结果
                best_detection = results.sort_values('confidence', ascending=False).iloc[0]
                
                # 提取边界框坐标
                x1, y1, x2, y2 = best_detection[['xmin', 'ymin', 'xmax', 'ymax']]
                confidence = float(best_detection['confidence'])
                
                # 计算中心点和半径
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # 估计半径为边界框的短边一半
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                radius = min(width, height) / 2
                
                # 更新结果
                result["detected"] = True
                result["position"] = [center_x, center_y]
                result["radius"] = radius
                result["confidence"] = confidence
                
        except Exception as e:
            print(f"YOLO检测过程中出错: {e}")
        
        return result
    
    def _get_bbox_from_yolo(self, yolo_result: Dict[str, Any]) -> Tuple[float, float, float, float]:
        """
        从YOLO结果中提取边界框坐标
        
        Args:
            yolo_result: YOLO检测结果
            
        Returns:
            边界框坐标 (x1, y1, x2, y2)
        """
        cx, cy = yolo_result["position"]
        r = yolo_result["radius"] * 2  # 使用直径作为边界框大小
        
        x1 = cx - r
        y1 = cy - r
        x2 = cx + r
        y2 = cy + r
        
        return x1, y1, x2, y2
    
    def _detect_hough_circles_in_roi(self, roi_image: np.ndarray) -> Dict[str, Any]:
        """
        在ROI区域内使用霍夫变换检测圆环
        
        Args:
            roi_image: ROI区域图像
            
        Returns:
            检测结果字典
        """
        # 初始化结果
        result = {
            "detected": False,
            "position": [0, 0],
            "radius": 0,
            "confidence": 0.0
        }
        
        # 检查ROI是否为空
        if roi_image is None or roi_image.size == 0:
            return result
        
        # 转换为灰度图
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        
        # 中值滤波去噪
        blur_ksize = self.hough_params["blur_ksize"]
        if blur_ksize % 2 == 0:  # 确保核大小为奇数
            blur_ksize += 1
        blurred = cv2.medianBlur(gray, blur_ksize)
        
        # 保存处理后的ROI用于调试
        self.debug_info["processed_roi"] = blurred
        
        # 使用霍夫梯度法检测圆
        try:
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=self.hough_params["dp"],
                minDist=self.hough_params["minDist"],
                param1=self.hough_params["param1"],
                param2=self.hough_params["param2"],
                minRadius=self.hough_params["minRadius"],
                maxRadius=self.hough_params["maxRadius"]
            )
            
            # 保存霍夫圆检测结果用于调试
            self.debug_info["hough_circles"] = circles
            
            if circles is not None:
                # 将检测到的圆转换为整数坐标
                circles = np.uint16(np.around(circles))
                
                # 取第一个检测到的圆
                circle = circles[0, 0]
                center_x, center_y, radius = circle
                
                # 计算置信度（这里使用一个简单的启发式方法）
                # 基于圆的清晰度和完整性评估
                edge_image = cv2.Canny(blurred, 50, 150)
                circle_mask = np.zeros_like(edge_image)
                cv2.circle(circle_mask, (center_x, center_y), radius, 255, 2)
                
                # 计算圆边界上的点与边缘点的重合率
                circle_points = np.count_nonzero(circle_mask)
                if circle_points > 0:
                    overlap = np.count_nonzero(cv2.bitwise_and(edge_image, circle_mask)) / circle_points
                    confidence = min(overlap * 2, 1.0)  # 缩放因子2可以调整
                else:
                    confidence = 0.5  # 默认置信度
                
                # 更新结果
                result["detected"] = True
                result["position"] = [float(center_x), float(center_y)]
                result["radius"] = float(radius)
                result["confidence"] = confidence
                
        except Exception as e:
            print(f"霍夫圆检测出错: {e}")
        
        return result
    
    def detect_with_hough(self, image: np.ndarray) -> Dict[str, Any]:
        """
        使用Hough变换检测圆环（可用于整个图像）
        
        Args:
            image: BGR格式输入图像
            
        Returns:
            检测结果字典
        """
        # 初始化结果
        result = {
            "detected": False,
            "position": [0, 0],
            "radius": 0,
            "confidence": 0.0,
            "method": "hough",
            "color": "unknown"
        }
        
        try:
            # 预处理图像
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 中值滤波去噪
            blur_ksize = self.hough_params["blur_ksize"]
            if blur_ksize % 2 == 0:  # 确保核大小为奇数
                blur_ksize += 1
            blurred = cv2.medianBlur(gray, blur_ksize)
            
            # 增强对比度（可选）
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(blurred)
            
            # 边缘检测
            edge_threshold = self.hough_params.get("edge_threshold", 100)
            edges = cv2.Canny(enhanced, edge_threshold // 2, edge_threshold)
            
            # 保存中间结果用于调试
            self.debug_info["gray"] = gray
            self.debug_info["blurred"] = blurred
            self.debug_info["enhanced"] = enhanced
            self.debug_info["edges"] = edges
            
            # 使用Hough圆检测
            circles = cv2.HoughCircles(
                enhanced,
                cv2.HOUGH_GRADIENT,
                dp=self.hough_params["dp"],
                minDist=self.hough_params["minDist"],
                param1=self.hough_params["param1"],
                param2=self.hough_params["param2"],
                minRadius=self.hough_params["minRadius"],
                maxRadius=self.hough_params["maxRadius"]
            )
            
            # 如果检测到圆
            if circles is not None:
                # 转换为整数坐标
                circles = np.uint16(np.around(circles))
                
                # 存储所有检测到的圆，待后续处理
                detected_circles = []
                
                # 限制最大检测圆数量
                max_circles = min(self.hough_params.get("max_circles", 8), len(circles[0]))
                
                for i, circle in enumerate(circles[0,:max_circles]):
                    center_x, center_y, radius = circle
                    
                    # 创建圆掩码
                    circle_mask = np.zeros_like(edges)
                    cv2.circle(circle_mask, (center_x, center_y), radius, 255, 2)
                    
                    # 计算圆边界上的点与边缘点的重合率
                    circle_points = np.count_nonzero(circle_mask)
                    if circle_points > 0:
                        overlap = np.count_nonzero(cv2.bitwise_and(edges, circle_mask)) / circle_points
                        confidence = min(overlap * 2, 1.0)  # 缩放因子2可以调整
                    else:
                        confidence = 0.5  # 默认置信度
                    
                    # 检测圆环的颜色
                    color = self._detect_circle_color(image, center_x, center_y, radius)
                    
                    # 将检测到的圆添加到列表
                    detected_circles.append({
                        "position": [center_x, center_y],
                        "radius": radius,
                        "confidence": confidence,
                        "color": color
                    })
                
                # 按置信度排序
                detected_circles.sort(key=lambda x: x["confidence"], reverse=True)
                
                # 至少有一个圆被检测到
                if detected_circles:
                    best_circle = detected_circles[0]
                    
                    # 更新结果
                    result["detected"] = True
                    result["position"] = best_circle["position"]
                    result["radius"] = best_circle["radius"]
                    result["confidence"] = best_circle["confidence"]
                    result["method"] = "hough"
                    result["color"] = best_circle["color"]
                    
                    # 保存检测到的圆用于调试
                    debug_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                    center_x, center_y = best_circle["position"]
                    radius = best_circle["radius"]
                    # 根据颜色设置圆环颜色
                    circle_color = {
                        "red": (0, 0, 255),
                        "green": (0, 255, 0),
                        "blue": (255, 0, 0),
                        "unknown": (128, 128, 128)
                    }.get(best_circle["color"], (0, 255, 0))
                    
                    cv2.circle(debug_image, (center_x, center_y), radius, circle_color, 2)
                    cv2.circle(debug_image, (center_x, center_y), 2, (0, 0, 255), 3)
                    self.debug_info["detected_circle"] = debug_image
                    
        except Exception as e:
            print(f"霍夫圆检测出错: {e}")
        
        return result
    
    def _detect_circle_color(self, image: np.ndarray, center_x: int, center_y: int, radius: int) -> str:
        """
        检测圆环的颜色
        
        Args:
            image: BGR格式图像
            center_x: 圆心x坐标
            center_y: 圆心y坐标
            radius: 圆半径
            
        Returns:
            颜色名称: "red", "green", "blue" 或 "unknown"
        """
        try:
            # 转换为HSV色彩空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 创建掩码，只考虑圆环区域
            # 注意：这里我们创建一个稍微小一点的掩码，以确保只检测圆环而不是背景
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            outer_radius = radius
            inner_radius = int(radius * 0.8)  # 内圆半径稍小一些
            
            # 绘制外圆和内圆，使掩码只包含圆环区域
            cv2.circle(mask, (center_x, center_y), outer_radius, 255, -1)
            cv2.circle(mask, (center_x, center_y), inner_radius, 0, -1)
            
            # 统计每种颜色的像素点数量
            color_counts = {}
            
            # 为每种颜色创建掩码
            for color, ranges in self.color_ranges.items():
                color_mask = np.zeros_like(mask)
                
                for lower, upper in ranges:
                    # 创建特定颜色的掩码
                    range_mask = cv2.inRange(hsv, lower, upper)
                    color_mask = cv2.bitwise_or(color_mask, range_mask)
                
                # 应用圆环掩码
                color_mask = cv2.bitwise_and(color_mask, mask)
                
                # 统计颜色区域像素数
                color_counts[color] = cv2.countNonZero(color_mask)
                
                # 保存调试信息
                self.debug_info["color_masks"][color] = color_mask
            
            # 如果没有足够的颜色像素，返回未知
            if sum(color_counts.values()) < 10:
                return "unknown"
                
            # 返回最多的颜色
            max_color = max(color_counts.items(), key=lambda x: x[1])
            
            # 如果最多的颜色占比很低，也返回未知
            if max_color[1] < 20:
                return "unknown"
                
            return max_color[0]
            
        except Exception as e:
            print(f"颜色检测错误: {e}")
            return "unknown"
    
    def get_debug_info(self) -> Dict[str, Any]:
        """
        获取调试信息
        
        Returns:
            包含调试信息的字典
        """
        return self.debug_info
    
    
if __name__ == "__main__":
    # 简单测试代码
    detector = CircleDetector()
    
    # 从摄像头读取图像
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 检测圆环
        result = detector.detect(frame)
        
        # 可视化结果
        if result["detected"]:
            # 提取检测结果
            position = tuple(map(int, result["position"]))
            radius = int(result["radius"])
            confidence = result["confidence"]
            method = result["method"]
            color = result["color"]
            
            # 根据检测方法选择不同的颜色
            color_map = {
                "yolo": (0, 0, 255),      # 红色
                "hough": (0, 255, 0),     # 绿色
                "combined": (255, 0, 0)   # 蓝色
            }
            color = color_map.get(method, (255, 255, 0))
            
            # 绘制圆环和中心点
            cv2.circle(frame, position, radius, color, 2)
            cv2.circle(frame, position, 3, (0, 255, 255), -1)  # 中心点
            
            # 显示置信度和检测方法
            cv2.putText(frame, f"{method}: {confidence:.2f}", 
                       (position[0] - 50, position[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            print(f"检测到圆环，方法: {method}，置信度: {confidence:.2f}")
        
        # 显示图像
        cv2.imshow("Circle Detector", frame)
        
        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows() 