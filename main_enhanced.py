#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import json
import time
import threading
import queue
import numpy as np
import os
import argparse
import yaml
from typing import Dict, Any, List, Tuple, Optional

# 导入自定义模块
from qrcode_detector import QRCodeDetector
from color_classifier import ColorClassifier
from circle_detector import CircleDetector
from vision_debug import ParameterManager, DebugVisualizer, LoggingManager, FPSCounter
from system_enhancer import AntiInterferenceSystem, QuickStartManager, EmergencyHandler
from serial_manager import SerialManager

class EnhancedVisionPipeline:
    """增强型多任务视觉处理流水线"""
    
    def __init__(self, 
                 camera_id: int = 0, 
                 resolution: Tuple[int, int] = (640, 480),
                 debug_mode: bool = False,
                 config_file: str = "vision_params.yaml",
                 serial_port: Optional[str] = None):
        """
        初始化视觉流水线
        
        Args:
            camera_id: 摄像头ID
            resolution: 图像分辨率 (宽, 高)
            debug_mode: 是否启用调试模式
            config_file: 参数配置文件
            serial_port: 串口号，如"COM3"或"/dev/ttyUSB0"
        """
        # 初始化日志系统
        log_folder = "vision_system"
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        self.logger = LoggingManager(log_folder=log_folder)
        self.logger.log_info("初始化视觉流水线...")
        
        # 基本参数
        self.camera_id = camera_id
        self.width, self.height = resolution
        self.debug_mode = debug_mode
        self.config_file = config_file
        
        # 串口参数
        self.serial_port = serial_port
        
        # 初始化快速启动管理器
        self.quick_start = QuickStartManager(cache_folder="cache")
        
        # 收集硬件信息
        system_info = self.quick_start.check_hardware()
        self.logger.log_info(f"硬件信息: {system_info}")
        
        # 加载应急处理器参数
        fallback_config = "configs/fallback.yaml"
        if not os.path.exists(os.path.dirname(fallback_config)):
            os.makedirs(os.path.dirname(fallback_config))
        
        # 初始化应急处理系统
        self.emergency = EmergencyHandler(fallback_config_path=fallback_config)
        self.logger.log_info("初始化应急处理机制")
        
        # 加载参数配置
        self.param_manager = ParameterManager(config_file)
        
        # 初始化防干扰系统
        self.anti_interference = AntiInterferenceSystem()
        
        # 初始化各模块
        self.qr_detector = self._init_qr_detector()
        self.color_classifier = self._init_color_classifier()
        self.circle_detector = self._init_circle_detector()
        
        # 初始化串口通信
        self.serial_manager = self._init_serial_manager()
        
        # 创建摄像头对象
        self.cap = None
        self.frame = None
        self.frame_lock = threading.Lock()
        
        # 初始化可视化组件
        if debug_mode:
            self.visualizer = DebugVisualizer(self.param_manager)
        else:
            self.visualizer = None
            
        # 创建FPS计数器
        self.fps_counter = FPSCounter(avg_frames=30)
        
        # 初始化结果队列
        self.qr_results = queue.Queue()
        self.color_results = queue.Queue()
        self.circle_results = queue.Queue()
        
        # 存储当前检测结果
        self.detection_results = {
            "qrcode": {"detected": False, "content": "", "position": [], "timestamp": 0},
            "material": {"detected": False, "color": "", "confidence": 0.0, "center": [0, 0], "timestamp": 0},
            "circle": {"detected": False, "position": [0, 0], "radius": 0, "confidence": 0.0, "timestamp": 0}
        }
        
        # 调试信息
        self.debug_info = {
            "color_mask": None,
            "circle_debug": None
        }
        
        # 性能统计
        self.processing_times = {
            "qrcode": [],
            "color": [],
            "circle": [],
            "total": []
        }
        
        # 运行状态
        self.running = False
        self.qr_thread = None
        self.color_thread = None
        self.circle_thread = None
        
        # 增加比赛模式和调试模式的显示设置
        self.show_debug_view = debug_mode  # 是否显示调试视图
        self.show_competition_view = True  # 是否显示比赛视图
        
        self.logger.log_info("视觉流水线初始化完成")
    
    def _init_qr_detector(self) -> QRCodeDetector:
        """初始化二维码检测器"""
        # 尝试从缓存加载
        cached_model = self.quick_start.load_model_from_cache("qr_detector")
        if cached_model:
            self.logger.log_info("从缓存加载二维码检测器")
            return cached_model
        
        # 创建新的检测器
        self.logger.log_info("创建新的二维码检测器")
        detector = QRCodeDetector()
        
        # 缓存模型
        self.quick_start.cache_model("qr_detector", detector)
        
        return detector
    
    def _init_color_classifier(self) -> ColorClassifier:
        """初始化颜色分类器"""
        # 创建新的分类器
        self.logger.log_info("创建新的颜色分类器")
        classifier = ColorClassifier()
        
        # 应用HSV阈值配置
        classifier.color_ranges = self.param_manager.get_hsv_ranges()
        
        return classifier
    
    def _init_circle_detector(self) -> CircleDetector:
        """初始化圆环检测器"""
        # 尝试从缓存加载
        cached_model = self.quick_start.load_model_from_cache("circle_detector")
        if cached_model:
            self.logger.log_info("从缓存加载圆环检测器")
            return cached_model
        
        # 创建新的检测器
        self.logger.log_info("创建新的圆环检测器")
        
        # 获取置信度阈值
        confidence_thresholds = self.param_manager.get_confidence_thresholds()
        circle_confidence = confidence_thresholds["circle_confidence"]
        
        # 检查是否存在本地YOLOv5模型路径和是否使用YOLO
        offline_mode = False
        model_path = None
        use_yolo = True
        
        # 从配置文件加载参数
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # 检查YOLOv5模型路径
            if "yolov5_model_path" in config:
                model_path = config["yolov5_model_path"]
                if model_path:
                    offline_mode = True
                    self.logger.log_info(f"使用离线模式，YOLOv5模型路径: {model_path}")
            
            # 检查是否禁用YOLO
            if "use_yolo" in config:
                use_yolo = config.get("use_yolo", True)
                if not use_yolo:
                    self.logger.log_info("已禁用YOLO模型，将仅使用霍夫圆检测")
        except Exception as e:
            self.logger.log_error(f"读取配置文件失败: {e}")
        
        # 创建检测器，支持离线模式和无YOLO模式
        detector = CircleDetector(
            confidence_threshold=circle_confidence,
            offline_mode=offline_mode,
            model_path=model_path,
            use_yolo=use_yolo
        )
        
        # 应用霍夫参数配置
        detector.set_hough_params(self.param_manager.get_hough_params())
        
        # 缓存模型
        self.quick_start.cache_model("circle_detector", detector)
        
        return detector
    
    def _update_color_thresholds(self) -> None:
        """更新颜色阈值"""
        self.color_classifier.color_ranges = self.param_manager.get_hsv_ranges()
        self.logger.log_debug("已更新HSV颜色阈值")
    
    def _update_hough_params(self) -> None:
        """更新霍夫圆检测参数"""
        self.circle_detector.set_hough_params(self.param_manager.get_hough_params())
        self.logger.log_debug("已更新霍夫圆检测参数")
    
    def _init_serial_manager(self) -> SerialManager:
        """初始化串口管理器"""
        if self.serial_port is None:
            self.logger.log_info("未指定串口，不启用串口通信")
            return None
            
        self.logger.log_info(f"初始化串口管理器，端口: {self.serial_port}")
        serial_manager = SerialManager(
            port=self.serial_port,
            baudrate=115200,
            logger=self.logger
        )
        
        return serial_manager
    
    def start(self):
        """启动视觉处理流水线"""
        if self.running:
            self.logger.log_warning("视觉流水线已经在运行中")
            return
            
        self.running = True
        
        # 启动工作线程
        self.qr_thread = threading.Thread(target=self._qr_detection_task, daemon=True)
        self.color_thread = threading.Thread(target=self._color_classification_task, daemon=True)
        self.circle_thread = threading.Thread(target=self._circle_detection_task, daemon=True)
        
        self.qr_thread.start()
        self.color_thread.start()
        self.circle_thread.start()
        
        # 启动串口通信
        if self.serial_manager:
            self.serial_manager.start()
            
        # 打开摄像头
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # 创建窗口
        if self.debug_mode:
            cv2.namedWindow("Debug View", cv2.WINDOW_NORMAL)
            
        # 创建比赛视图窗口
        cv2.namedWindow("Competition View", cv2.WINDOW_NORMAL)
        
        self.logger.log_info("视觉处理流水线已启动")
    
    def stop(self):
        """停止视觉处理流水线"""
        if not self.running:
            return
            
        self.running = False
        
        # 等待线程结束
        if hasattr(self, 'qr_thread'):
            self.qr_thread.join(timeout=1.0)
        if hasattr(self, 'color_thread'):
            self.color_thread.join(timeout=1.0)
        if hasattr(self, 'circle_thread'):
            self.circle_thread.join(timeout=1.0)
            
        # 停止串口通信
        if self.serial_manager:
            self.serial_manager.stop()
            
        # 释放摄像头
        if self.cap is not None:
            self.cap.release()
            
        # 关闭调试窗口
        if self.visualizer:
            self.visualizer.close()
            
        # 关闭所有窗口
        cv2.destroyAllWindows()
            
        self.logger.log_info("视觉处理流水线已停止")
    
    def _qr_detection_task(self):
        """二维码检测任务线程"""
        self.logger.log_info("二维码检测任务线程已启动")
        
        while self.running:
            try:
                # 获取当前帧
                with self.frame_lock:
                    if self.frame is None:
                        time.sleep(0.01)
                        continue
                    frame = self.frame.copy()
                
                # 更新心跳
                self.emergency.register_heartbeat("qr_detector")
                
                # 执行二维码检测
                start_time = time.time()
                result = self.qr_detector.detect(frame)
                
                # 记录处理时间
                process_time = time.time() - start_time
                self.processing_times["qrcode"].append(process_time)
                if len(self.processing_times["qrcode"]) > 30:
                    self.processing_times["qrcode"].pop(0)
                
                # 将结果放入队列
                self.qr_results.put(result)
                
                # 控制检测频率
                time.sleep(0.02)
                
            except Exception as e:
                self.logger.log_exception(f"二维码检测任务出错: {e}")
                # 记录故障
                self.emergency.record_failure()
                time.sleep(0.1)
    
    def _color_classification_task(self):
        """物料颜色分类任务线程"""
        self.logger.log_info("颜色分类任务线程已启动")
        
        while self.running:
            try:
                # 获取当前帧
                with self.frame_lock:
                    if self.frame is None:
                        time.sleep(0.01)
                        continue
                    frame = self.frame.copy()
                
                # 更新心跳
                self.emergency.register_heartbeat("color_classifier")
                
                # 执行颜色分类
                start_time = time.time()
                
                # 预处理图像
                hsv_image = self.color_classifier.preprocess_image(frame)
                
                # 存储调试信息
                color_mask = None
                best_color = None
                best_contour = None
                best_confidence = 0.0
                
                # 对每种颜色进行检测
                for color in self.color_classifier.color_ranges.keys():
                    # 创建当前颜色的掩码
                    mask = self.color_classifier.create_mask_for_color(hsv_image, color)
                    
                    # 找到最大轮廓
                    contour, confidence = self.color_classifier.find_largest_contour(mask)
                    
                    # 更新最佳结果
                    if contour is not None and confidence > best_confidence:
                        best_color = color
                        best_contour = contour
                        best_confidence = confidence
                        
                        # 保存掩码用于调试
                        if self.debug_mode:
                            # 为调试目的着色掩码
                            color_mask_bgr = np.zeros_like(frame)
                            color_bgr = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}.get(color, (255, 255, 255))
                            color_mask_bgr[mask > 0] = color_bgr
                            color_mask = color_mask_bgr
                
                # 构建结果
                result = {
                    "detected": False,
                    "color": "",
                    "center": [0, 0],
                    "confidence": 0.0
                }
                
                # 检查是否找到了有效物料
                confidence_thresholds = self.param_manager.get_confidence_thresholds()
                min_confidence = confidence_thresholds["color_confidence"]
                
                if best_contour is not None and best_confidence > min_confidence:
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
                
                # 存储中间结果用于调试
                self.debug_info["color_mask"] = color_mask
                
                # 记录处理时间
                process_time = time.time() - start_time
                self.processing_times["color"].append(process_time)
                if len(self.processing_times["color"]) > 30:
                    self.processing_times["color"].pop(0)
                
                # 将结果放入队列
                self.color_results.put(result)
                
                # 控制检测频率
                time.sleep(0.02)
                
            except Exception as e:
                self.logger.log_exception(f"颜色分类任务出错: {e}")
                # 记录故障
                self.emergency.record_failure()
                time.sleep(0.1)
    
    def _circle_detection_task(self):
        """圆环检测任务线程"""
        self.logger.log_info("圆环检测任务线程已启动")
        
        while self.running:
            try:
                # 获取当前帧
                with self.frame_lock:
                    if self.frame is None:
                        time.sleep(0.01)
                        continue
                    frame = self.frame.copy()
                
                # 更新心跳
                self.emergency.register_heartbeat("circle_detector")
                
                # 执行圆环检测
                start_time = time.time()
                result = self.circle_detector.detect(frame)
                
                # 保存调试信息
                if self.debug_mode:
                    self.debug_info["circle_debug"] = self.circle_detector.get_debug_info()
                
                # 记录处理时间
                process_time = time.time() - start_time
                self.processing_times["circle"].append(process_time)
                if len(self.processing_times["circle"]) > 30:
                    self.processing_times["circle"].pop(0)
                
                # 将结果放入队列
                self.circle_results.put(result)
                
                # 控制检测频率（目标检测计算量大，降低频率）
                time.sleep(0.05)
                
            except Exception as e:
                self.logger.log_exception(f"圆环检测任务出错: {e}")
                # 记录故障
                self.emergency.record_failure()
                time.sleep(0.1)
    
    def update_results(self):
        """更新并整合各模块的检测结果"""
        changed = False
        
        # 处理二维码检测结果
        while not self.qr_results.empty():
            qr_result = self.qr_results.get()
            self.detection_results["qrcode"] = qr_result
            changed = True
        
        # 处理颜色分类结果
        while not self.color_results.empty():
            color_result = self.color_results.get()
            # 验证颜色结果可靠性
            if self.emergency.validate_color_result(color_result):
                self.detection_results["material"] = color_result
                changed = True
            elif color_result["detected"]:
                # 如果不在白名单内但检测到，记录为低置信度结果
                self.logger.log_warning(f"检测到低置信度颜色: {color_result['color']}({color_result['confidence']:.2f})")
        
        # 处理圆环检测结果
        while not self.circle_results.empty():
            circle_result = self.circle_results.get()
            self.detection_results["circle"] = circle_result
            changed = True
        
        # 如果有结果更新，通过串口发送
        if changed and self.serial_manager:
            # 发送所有检测结果
            self.serial_manager.send_all_results(self.detection_results)
        
        # 返回当前检测结果
        return self.detection_results
    
    def process_frame(self, frame=None):
        """
        处理一帧图像
        
        Args:
            frame: 要处理的图像，如果为None则从摄像头获取
            
        Returns:
            处理后的图像和检测结果
        """
        # 记录开始时间
        start_time = time.time()
        
        # 如果没有提供帧，从摄像头获取
        if frame is None:
            if self.cap is None or not self.cap.isOpened():
                if self.debug_mode:
                    # 创建一个黑色图像
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    cv2.putText(frame, "Camera not available", (50, self.height//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    return frame, self.detection_results
                else:
                    return None, self.detection_results
                
            ret, frame = self.cap.read()
            if not ret:
                return None, self.detection_results
                
        # 更新当前帧（线程安全）
        with self.frame_lock:
            self.frame = frame.copy()
            
        # 更新检测结果
        self.update_results()
        
        # 获取当前检测结果的副本
        results = {
            "qrcode": self.detection_results["qrcode"].copy(),
            "material": self.detection_results["material"].copy(),
            "circle": self.detection_results["circle"].copy()
        }
        
        # 计算处理时间
        process_time = time.time() - start_time
        self.processing_times["total"].append(process_time)
        if len(self.processing_times["total"]) > 30:
            self.processing_times["total"].pop(0)
        
        # 更新FPS
        current_fps = self.fps_counter.update()
        
        # 在调试模式下可视化结果
        if self.debug_mode and self.visualizer and self.show_debug_view:
            debug_frame = self.visualizer.show_debug_view(
                frame, 
                results["qrcode"], 
                results["material"], 
                results["circle"],
                self.debug_info["color_mask"],
                self.debug_info["circle_debug"]
            )
            cv2.imshow("Debug View", debug_frame)
            
        # 创建比赛视图
        if self.show_competition_view:
            competition_frame = self._create_competition_view(frame.copy(), results)
            cv2.imshow("Competition View", competition_frame)
        
        # 记录检测结果
        self.logger.log_detection_results(
            results["qrcode"],
            results["material"],
            results["circle"]
        )
        
        return frame, results
    
    def _create_competition_view(self, frame, results):
        """
        创建比赛视图
        
        Args:
            frame: 原始图像
            results: 检测结果
            
        Returns:
            比赛视图图像
        """
        # 基本可视化
        vis_frame = self._visualize_results(frame.copy(), results)
        
        # 在比赛视图中显示二维码结果
        self._display_qr_result(vis_frame, results["qrcode"])
        
        # 显示FPS
        cv2.putText(vis_frame, f"FPS: {self.fps_counter.get_fps():.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_frame

    def _display_qr_result(self, frame, qr_result):
        """
        在图像上显示二维码结果
        
        Args:
            frame: 要显示结果的图像
            qr_result: 二维码检测结果
            
        Returns:
            添加了二维码结果的图像
        """
        # 图像高度和宽度
        height, width = frame.shape[:2]
        
        # 创建一个白色背景
        text_bg_y1 = height - 80
        text_bg_y2 = height - 20
        text_bg_x1 = 20
        text_bg_x2 = width - 20
        
        # 绘制白色背景框
        cv2.rectangle(frame, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), (255, 255, 255), -1)
        
        # 显示二维码结果
        if qr_result["detected"]:
            qr_text = f"QR Code: {qr_result['content']}"
        else:
            qr_text = "Waiting QR Code..."
        
        # 计算文本位置，使其居中
        text_size = cv2.getTextSize(qr_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = text_bg_y1 + (text_bg_y2 - text_bg_y1 + text_size[1]) // 2
        
        # 绘制文本，使用黑色字体
        cv2.putText(frame, qr_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
                   
        return frame
    
    def _visualize_results(self, frame, results):
        """在图像上可视化检测结果"""
        # 绘制二维码区域
        if results["qrcode"]["detected"]:
            points = np.array(results["qrcode"]["position"], dtype=np.int32)
            cv2.polylines(frame, [points], True, (0, 255, 0), 2)
            cv2.putText(frame, results["qrcode"]["content"][:10], 
                       (points[0][0], points[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 绘制物料颜色区域
        if results["material"]["detected"]:
            center = tuple(map(int, results["material"]["center"]))
            color_name = results["material"]["color"]
            color_bgr = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}.get(color_name, (255, 255, 255))
            cv2.circle(frame, center, 30, color_bgr, 2)
            cv2.putText(frame, f"{color_name}", 
                       (center[0] - 20, center[1] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
        
        # 绘制圆环区域
        if results["circle"]["detected"]:
            center = tuple(map(int, results["circle"]["position"]))
            radius = int(results["circle"]["radius"])
            # 获取圆环颜色（如果有）
            color_name = results["circle"].get("color", "unknown")
            color_bgr = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}.get(color_name, (255, 255, 255))
            
            # 绘制圆环
            cv2.circle(frame, center, radius, color_bgr, 2)
            # 绘制圆心
            cv2.circle(frame, center, 4, (0, 0, 255), -1)
            # 显示圆环信息
            cv2.putText(frame, f"{color_name} r:{radius}", 
                       (center[0] - 40, center[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
        
        return frame
    
    def get_json_results(self):
        """返回JSON格式的检测结果"""
        return json.dumps(self.detection_results, ensure_ascii=False, indent=2)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        获取性能统计信息
        
        Returns:
            性能统计字典
        """
        stats = {}
        
        # 计算平均处理时间
        for key, times in self.processing_times.items():
            if times:
                avg_time = sum(times) / len(times)
                stats[f"{key}_avg_time"] = avg_time * 1000  # 转换为毫秒
            else:
                stats[f"{key}_avg_time"] = 0
        
        # 当前FPS
        fps = self.fps_counter.update()
        stats["fps"] = fps
        
        return stats


def main():
    """主程序入口"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="智能物流机器人视觉识别系统")
    parser.add_argument("--camera", type=int, default=0, help="摄像头ID")
    parser.add_argument("--width", type=int, default=640, help="图像宽度")
    parser.add_argument("--height", type=int, default=480, help="图像高度")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--config", type=str, default="vision_params.yaml", help="配置文件路径")
    parser.add_argument("--port", type=str, help="串口号，如COM3")
    parser.add_argument("--no-yolo", action="store_true", help="禁用YOLO模型，仅使用霍夫圆检测")
    
    args = parser.parse_args()
    
    # 如果--no-yolo参数被设置，修改配置文件
    if args.no_yolo:
        try:
            # 使用utf-8编码打开文件
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            # 设置use_yolo = False
            config['use_yolo'] = False
            
            # 保存修改后的配置
            with open(args.config, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            print(f"已禁用YOLO模型，使用纯霍夫圆检测模式")
        except Exception as e:
            print(f"警告: 无法修改配置文件: {e}")
            print("将直接使用无YOLO模式启动")
    
    # 初始化视觉流水线
    try:
        pipeline = EnhancedVisionPipeline(
            camera_id=args.camera,
            resolution=(args.width, args.height),
            debug_mode=args.debug,
            config_file=args.config,
            serial_port=args.port
        )
        
        # 启动流水线
        pipeline.start()
        
        try:
            while True:
                # 处理当前帧
                frame, results = pipeline.process_frame()
                
                # 处理键盘事件
                key = cv2.waitKey(1) & 0xFF
                
                # ESC键退出
                if key == 27:
                    break
                    
                # 'q'键退出
                elif key == ord('q'):
                    break
                    
                # 'd'键切换调试视图
                elif key == ord('d'):
                    pipeline.show_debug_view = not pipeline.show_debug_view
                    
                # 'c'键切换比赛视图
                elif key == ord('c'):
                    pipeline.show_competition_view = not pipeline.show_competition_view
                    
        except KeyboardInterrupt:
            print("用户中断，正在停止...")
        finally:
            # 停止流水线
            pipeline.stop()
            
    except Exception as e:
        print(f"程序出错: {e}")
    
    print("程序已退出")


if __name__ == "__main__":
    main() 