#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import yaml
import time
import os
from typing import Dict, Any, List, Tuple, Optional, Callable
import logging
import csv
import datetime

class FPSCounter:
    """FPS计数器"""
    
    def __init__(self, avg_frames: int = 30):
        """
        初始化FPS计数器
        
        Args:
            avg_frames: 计算平均FPS的帧数
        """
        self.frame_times = []
        self.avg_frames = avg_frames
        self.last_frame_time = time.time()
        self.current_fps = 0
    
    def update(self) -> float:
        """
        更新FPS计数器
        
        Returns:
            当前FPS
        """
        current_time = time.time()
        delta = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # 添加新的帧时间
        self.frame_times.append(delta)
        
        # 保持固定长度
        if len(self.frame_times) > self.avg_frames:
            self.frame_times.pop(0)
            
        # 计算FPS
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            self.current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        else:
            self.current_fps = 0
            
        return self.current_fps
        
    def get_fps(self) -> float:
        """
        获取当前FPS值
        
        Returns:
            当前FPS
        """
        return self.current_fps


class ParameterManager:
    """参数管理器"""
    
    def __init__(self, config_file: str = "vision_params.yaml"):
        """
        初始化参数管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.parameters = {
            "hsv_thresholds": {
                "red_lower_h": [0, 180, 0],    # [初始值, 最大值, 最小值]
                "red_lower_s": [100, 255, 0],
                "red_lower_v": [100, 255, 0],
                "red_upper_h": [10, 180, 0],
                "red_upper_s": [255, 255, 0],
                "red_upper_v": [255, 255, 0],
                "red_lower2_h": [160, 180, 0],
                "red_lower2_s": [100, 255, 0],
                "red_lower2_v": [100, 255, 0],
                "red_upper2_h": [180, 180, 0],
                "red_upper2_s": [255, 255, 0],
                "red_upper2_v": [255, 255, 0],
                "green_lower_h": [35, 180, 0],
                "green_lower_s": [100, 255, 0],
                "green_lower_v": [100, 255, 0],
                "green_upper_h": [85, 180, 0],
                "green_upper_s": [255, 255, 0],
                "green_upper_v": [255, 255, 0],
                "blue_lower_h": [100, 180, 0],
                "blue_lower_s": [100, 255, 0],
                "blue_lower_v": [100, 255, 0],
                "blue_upper_h": [130, 180, 0],
                "blue_upper_s": [255, 255, 0],
                "blue_upper_v": [255, 255, 0],
            },
            "hough_params": {
                "dp": [12, 20, 5],              # [初始值*10, 最大值*10, 最小值*10]
                "minDist": [50, 200, 10],
                "param1": [100, 300, 10],
                "param2": [30, 100, 5],
                "minRadius": [10, 100, 1],
                "maxRadius": [100, 300, 10],
                "blur_ksize": [7, 15, 3],
                "roi_padding": [20, 100, 0],
            },
            "confidence_thresholds": {
                "qr_confidence": [50, 100, 0],  # [初始值*100, 最大值*100, 最小值*100]
                "color_confidence": [50, 100, 0],
                "circle_confidence": [40, 100, 0],
            }
        }
        
        # 加载参数
        self.load_parameters()
        
        # 回调函数
        self.callbacks = {}
    
    def load_parameters(self) -> bool:
        """
        从文件加载参数
        
        Returns:
            是否成功加载
        """
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_params = yaml.safe_load(f)
                
                # 更新参数，保留元数据（最大值、最小值）
                if loaded_params:
                    for category, params in loaded_params.items():
                        if category in self.parameters:
                            for param_name, value in params.items():
                                if param_name in self.parameters[category]:
                                    # 更新初始值，保留最大值和最小值
                                    self.parameters[category][param_name][0] = value
                
                return True
            except Exception as e:
                print(f"加载参数文件出错: {e}")
                return False
        return False
    
    def save_parameters(self) -> bool:
        """
        保存参数到文件
        
        Returns:
            是否成功保存
        """
        try:
            # 提取当前参数值
            params_to_save = {}
            for category, params in self.parameters.items():
                params_to_save[category] = {}
                for param_name, param_data in params.items():
                    params_to_save[category][param_name] = param_data[0]
            
            # 保存到文件
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(params_to_save, f, default_flow_style=False, allow_unicode=True)
            
            return True
        except Exception as e:
            print(f"保存参数文件出错: {e}")
            return False
    
    def get_parameter(self, category: str, param_name: str) -> Any:
        """
        获取参数值
        
        Args:
            category: 参数类别
            param_name: 参数名称
            
        Returns:
            参数值
        """
        if category in self.parameters and param_name in self.parameters[category]:
            # 某些参数需要除以10或100
            if category == "hough_params" and param_name == "dp":
                return self.parameters[category][param_name][0] / 10.0
            elif category == "confidence_thresholds":
                return self.parameters[category][param_name][0] / 100.0
            else:
                return self.parameters[category][param_name][0]
        return None
    
    def set_parameter(self, category: str, param_name: str, value: int) -> None:
        """
        设置参数值
        
        Args:
            category: 参数类别
            param_name: 参数名称
            value: 参数值
        """
        if category in self.parameters and param_name in self.parameters[category]:
            # 限制在最大值和最小值范围内
            max_val = self.parameters[category][param_name][1]
            min_val = self.parameters[category][param_name][2]
            value = max(min(value, max_val), min_val)
            
            # 更新参数值
            self.parameters[category][param_name][0] = value
            
            # 触发回调函数
            if category in self.callbacks:
                for callback in self.callbacks[category]:
                    callback()
            
            # 自动保存参数
            self.save_parameters()
    
    def register_callback(self, category: str, callback: Callable[[], None]) -> None:
        """
        注册参数变更回调函数
        
        Args:
            category: 参数类别
            callback: 回调函数
        """
        if category not in self.callbacks:
            self.callbacks[category] = []
        self.callbacks[category].append(callback)
    
    def get_hsv_ranges(self) -> Dict[str, List[Dict[str, np.ndarray]]]:
        """
        获取HSV颜色范围
        
        Returns:
            颜色范围字典
        """
        color_ranges = {
            "red": [
                {
                    "lower": np.array([
                        self.get_parameter("hsv_thresholds", "red_lower_h"),
                        self.get_parameter("hsv_thresholds", "red_lower_s"),
                        self.get_parameter("hsv_thresholds", "red_lower_v")
                    ]),
                    "upper": np.array([
                        self.get_parameter("hsv_thresholds", "red_upper_h"),
                        self.get_parameter("hsv_thresholds", "red_upper_s"),
                        self.get_parameter("hsv_thresholds", "red_upper_v")
                    ])
                },
                {
                    "lower": np.array([
                        self.get_parameter("hsv_thresholds", "red_lower2_h"),
                        self.get_parameter("hsv_thresholds", "red_lower2_s"),
                        self.get_parameter("hsv_thresholds", "red_lower2_v")
                    ]),
                    "upper": np.array([
                        self.get_parameter("hsv_thresholds", "red_upper2_h"),
                        self.get_parameter("hsv_thresholds", "red_upper2_s"),
                        self.get_parameter("hsv_thresholds", "red_upper2_v")
                    ])
                }
            ],
            "green": [
                {
                    "lower": np.array([
                        self.get_parameter("hsv_thresholds", "green_lower_h"),
                        self.get_parameter("hsv_thresholds", "green_lower_s"),
                        self.get_parameter("hsv_thresholds", "green_lower_v")
                    ]),
                    "upper": np.array([
                        self.get_parameter("hsv_thresholds", "green_upper_h"),
                        self.get_parameter("hsv_thresholds", "green_upper_s"),
                        self.get_parameter("hsv_thresholds", "green_upper_v")
                    ])
                }
            ],
            "blue": [
                {
                    "lower": np.array([
                        self.get_parameter("hsv_thresholds", "blue_lower_h"),
                        self.get_parameter("hsv_thresholds", "blue_lower_s"),
                        self.get_parameter("hsv_thresholds", "blue_lower_v")
                    ]),
                    "upper": np.array([
                        self.get_parameter("hsv_thresholds", "blue_upper_h"),
                        self.get_parameter("hsv_thresholds", "blue_upper_s"),
                        self.get_parameter("hsv_thresholds", "blue_upper_v")
                    ])
                }
            ]
        }
        return color_ranges
    
    def get_hough_params(self) -> Dict[str, Any]:
        """
        获取霍夫圆检测参数
        
        Returns:
            霍夫参数字典
        """
        return {
            "dp": self.get_parameter("hough_params", "dp"),
            "minDist": self.get_parameter("hough_params", "minDist"),
            "param1": self.get_parameter("hough_params", "param1"),
            "param2": self.get_parameter("hough_params", "param2"),
            "minRadius": self.get_parameter("hough_params", "minRadius"),
            "maxRadius": self.get_parameter("hough_params", "maxRadius"),
            "blur_ksize": self.get_parameter("hough_params", "blur_ksize"),
            "roi_padding": self.get_parameter("hough_params", "roi_padding")
        }
    
    def get_confidence_thresholds(self) -> Dict[str, float]:
        """
        获取置信度阈值
        
        Returns:
            置信度阈值字典
        """
        return {
            "qr_confidence": self.get_parameter("confidence_thresholds", "qr_confidence"),
            "color_confidence": self.get_parameter("confidence_thresholds", "color_confidence"),
            "circle_confidence": self.get_parameter("confidence_thresholds", "circle_confidence")
        }
    
    def get_all_params(self) -> Dict[str, Any]:
        """
        获取所有参数的当前值
        
        Returns:
            所有参数的字典
        """
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                all_params = yaml.safe_load(f)
            return all_params if all_params else {}
        except Exception as e:
            logging.error(f"读取配置文件失败: {e}")
            return {}


class DebugVisualizer:
    """调试可视化工具"""
    
    def __init__(self, param_manager: ParameterManager):
        """
        初始化调试可视化工具
        
        Args:
            param_manager: 参数管理器
        """
        self.param_manager = param_manager
        self.window_name = "视觉系统调试面板"
        
        # 创建主窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        
        # 创建参数调节面板
        self._create_trackbars()
        
        # FPS计数器
        self.fps_counter = FPSCounter()
        
        # 上次更新时间
        self.last_update_time = time.time()
    
    def _create_trackbars(self) -> None:
        """创建参数调节滑动条"""
        # HSV颜色阈值
        for color in ["red", "red2", "green", "blue"]:
            for channel in ["h", "s", "v"]:
                for bound in ["lower", "upper"]:
                    if color == "red2":
                        param_name = f"red_lower2_{channel}" if bound == "lower" else f"red_upper2_{channel}"
                    else:
                        param_name = f"{color}_{bound}_{channel}"
                    
                    if param_name in self.param_manager.parameters["hsv_thresholds"]:
                        cv2.createTrackbar(
                            param_name,
                            self.window_name,
                            self.param_manager.parameters["hsv_thresholds"][param_name][0],
                            self.param_manager.parameters["hsv_thresholds"][param_name][1],
                            lambda x, category="hsv_thresholds", name=param_name: self.param_manager.set_parameter(category, name, x)
                        )
        
        # 霍夫圆检测参数
        for param_name in self.param_manager.parameters["hough_params"]:
            cv2.createTrackbar(
                param_name,
                self.window_name,
                self.param_manager.parameters["hough_params"][param_name][0],
                self.param_manager.parameters["hough_params"][param_name][1],
                lambda x, category="hough_params", name=param_name: self.param_manager.set_parameter(category, name, x)
            )
        
        # 置信度阈值
        for param_name in self.param_manager.parameters["confidence_thresholds"]:
            cv2.createTrackbar(
                param_name,
                self.window_name,
                self.param_manager.parameters["confidence_thresholds"][param_name][0],
                self.param_manager.parameters["confidence_thresholds"][param_name][1],
                lambda x, category="confidence_thresholds", name=param_name: self.param_manager.set_parameter(category, name, x)
            )
    
    def show_debug_view(self, 
                         frame: np.ndarray, 
                         qr_result: Dict[str, Any], 
                         color_result: Dict[str, Any], 
                         circle_result: Dict[str, Any],
                         color_mask: Optional[np.ndarray] = None,
                         circle_debug_info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        显示调试视图
        
        Args:
            frame: 原始帧
            qr_result: 二维码检测结果
            color_result: 颜色分类结果
            circle_result: 圆环检测结果
            color_mask: 颜色掩码
            circle_debug_info: 圆环检测调试信息
            
        Returns:
            组合后的调试视图
        """
        # 更新FPS
        current_fps = self.fps_counter.update()
        
        # 创建拼接画布
        h, w = frame.shape[:2]
        canvas_w = w * 2
        canvas_h = h * 2
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # 1. 左侧：原始帧+二维码框选
        vis_frame = frame.copy()
        
        # 绘制二维码检测结果
        if qr_result["detected"]:
            points = np.array(qr_result["position"], dtype=np.int32)
            cv2.polylines(vis_frame, [points], True, (0, 255, 0), 2)
            cv2.putText(vis_frame, qr_result["content"][:15], 
                       (points[0][0], points[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 2. 右上：颜色掩码+轮廓绘制
        color_vis = vis_frame.copy()
        
        # 绘制颜色分类结果
        if color_result["detected"]:
            center = tuple(map(int, color_result["center"]))
            color_name = color_result["color"]
            confidence = color_result["confidence"]
            color_bgr = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}.get(color_name, (255, 255, 255))
            cv2.circle(color_vis, center, 30, color_bgr, 2)
            cv2.putText(color_vis, f"{color_name}: {confidence:.2f}", 
                      (center[0] - 40, center[1] - 40),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
        
        # 如果有颜色掩码，则叠加显示
        if color_mask is not None:
            # 确保掩码是3通道
            if len(color_mask.shape) == 2:
                mask_vis = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
            else:
                mask_vis = color_mask
            
            # 缩放到与原图相同大小
            if mask_vis.shape[:2] != color_vis.shape[:2]:
                mask_vis = cv2.resize(mask_vis, (color_vis.shape[1], color_vis.shape[0]))
            
            # 显示50%透明度的掩码
            color_vis = cv2.addWeighted(color_vis, 0.7, mask_vis, 0.3, 0)
        
        # 3. 右下：YOLO检测+圆环标注
        circle_vis = vis_frame.copy()
        
        # 绘制圆环检测结果
        if circle_result["detected"]:
            position = tuple(map(int, circle_result["position"]))
            radius = int(circle_result["radius"])
            confidence = circle_result["confidence"]
            method = circle_result["method"]
            
            # 根据检测方法选择不同的颜色
            color_map = {
                "yolo": (0, 0, 255),      # 红色
                "hough": (0, 255, 0),     # 绿色
                "combined": (255, 0, 0)   # 蓝色
            }
            color = color_map.get(method, (255, 255, 0))
            
            # 绘制圆环和中心点
            cv2.circle(circle_vis, position, radius, color, 2)
            cv2.circle(circle_vis, position, 3, (0, 255, 255), -1)  # 中心点
            
            # 显示置信度和检测方法
            cv2.putText(circle_vis, f"{method}: {confidence:.2f}", 
                       (position[0] - 50, position[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 如果有圆环调试信息，显示YOLO框和ROI区域
        if circle_debug_info:
            # 显示YOLO检测框
            if circle_debug_info["yolo_box"] is not None:
                x1, y1, x2, y2 = circle_debug_info["yolo_box"]
                cv2.rectangle(circle_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
            
            # 缩略显示ROI图像
            if circle_debug_info["roi_image"] is not None:
                roi_img = circle_debug_info["roi_image"]
                roi_h, roi_w = roi_img.shape[:2]
                # 在右下角缩略显示
                target_h = 150  # 目标高度
                target_w = int(roi_w * (target_h / roi_h))
                roi_resized = cv2.resize(roi_img, (target_w, target_h))
                # 放置在右下角
                padding = 10
                start_x = circle_vis.shape[1] - target_w - padding
                start_y = circle_vis.shape[0] - target_h - padding
                if start_x > 0 and start_y > 0:
                    circle_vis[start_y:start_y+target_h, start_x:start_x+target_w] = roi_resized
        
        # 放置图像到画布
        # 左侧：原始帧+二维码框选
        canvas[:h, :w] = vis_frame
        # 右上：颜色掩码+轮廓绘制
        canvas[:h, w:] = color_vis
        # 右下：YOLO检测+圆环标注
        canvas[h:, w:] = circle_vis
        
        # 左下：参数状态和性能指标
        status_panel = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 绘制FPS和时间
        cv2.putText(status_panel, f"FPS: {current_fps:.1f}", (20, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(status_panel, timestamp, (20, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # 绘制当前关键参数状态
        params_to_show = [
            f"霍夫参数: dp={self.param_manager.get_parameter('hough_params', 'dp'):.1f}, "
            f"param1={self.param_manager.get_parameter('hough_params', 'param1')}, "
            f"param2={self.param_manager.get_parameter('hough_params', 'param2')}",
            
            f"半径范围: {self.param_manager.get_parameter('hough_params', 'minRadius')}-"
            f"{self.param_manager.get_parameter('hough_params', 'maxRadius')}",
            
            f"置信度阈值: 二维码={self.param_manager.get_parameter('confidence_thresholds', 'qr_confidence'):.2f}, "
            f"颜色={self.param_manager.get_parameter('confidence_thresholds', 'color_confidence'):.2f}, "
            f"圆环={self.param_manager.get_parameter('confidence_thresholds', 'circle_confidence'):.2f}"
        ]
        
        for i, param_text in enumerate(params_to_show):
            cv2.putText(status_panel, param_text, (20, 100 + 30 * i), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 250), 1)
        
        # 绘制检测状态
        status_texts = []
        if qr_result["detected"]:
            status_texts.append(f"二维码: {qr_result['content'][:20]}")
        else:
            status_texts.append("二维码: 未检测到")
            
        if color_result["detected"]:
            status_texts.append(f"颜色: {color_result['color']}, 置信度: {color_result['confidence']:.2f}")
        else:
            status_texts.append("颜色: 未检测到")
            
        if circle_result["detected"]:
            status_texts.append(f"圆环: {circle_result['method']}, 半径: {circle_result['radius']:.1f}, "
                               f"置信度: {circle_result['confidence']:.2f}")
        else:
            status_texts.append("圆环: 未检测到")
        
        for i, status_text in enumerate(status_texts):
            cv2.putText(status_panel, status_text, (20, 200 + 30 * i), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 放置状态面板到画布
        canvas[h:, :w] = status_panel
        
        # 显示画布
        cv2.imshow(self.window_name, canvas)
        
        return canvas
    
    def close(self) -> None:
        """关闭调试视图"""
        cv2.destroyWindow(self.window_name)


class LoggingManager:
    """日志记录系统"""
    
    def __init__(self, log_folder: str = "logs", csv_log_file: str = None):
        """
        初始化日志记录系统
        
        Args:
            log_folder: 日志文件夹
            csv_log_file: CSV日志文件名，None则自动生成
        """
        # 创建日志文件夹
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        
        # 初始化日志记录器
        self.logger = logging.getLogger("vision_system")
        self.logger.setLevel(logging.DEBUG)
        
        # 移除所有处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 创建文件处理器
        log_file = os.path.join(log_folder, f"vision_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 初始化CSV记录器
        if csv_log_file is None:
            self.csv_file = os.path.join(log_folder, f"results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        else:
            self.csv_file = os.path.join(log_folder, csv_log_file)
        
        # 写入CSV头
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 
                'qr_detected', 'qr_content', 'qr_confidence',
                'color_detected', 'color_name', 'color_confidence',
                'circle_detected', 'circle_method', 'circle_radius', 'circle_confidence'
            ])
        
        self.logger.info(f"日志系统初始化完成，日志文件: {log_file}, CSV文件: {self.csv_file}")
    
    def log_detection_results(self, 
                             qr_result: Dict[str, Any], 
                             color_result: Dict[str, Any], 
                             circle_result: Dict[str, Any]) -> None:
        """
        记录检测结果
        
        Args:
            qr_result: 二维码检测结果
            color_result: 颜色分类结果
            circle_result: 圆环检测结果
        """
        # 记录到日志
        log_msg = []
        if qr_result["detected"]:
            log_msg.append(f"二维码: {qr_result['content'][:20]}(置信度:{qr_result['confidence']:.2f})")
        
        if color_result["detected"]:
            log_msg.append(f"颜色: {color_result['color']}(置信度:{color_result['confidence']:.2f})")
        
        if circle_result["detected"]:
            log_msg.append(f"圆环: {circle_result['method']}(半径:{circle_result['radius']:.1f}, "
                          f"置信度:{circle_result['confidence']:.2f})")
        
        if log_msg:
            self.logger.info("检测结果: " + ", ".join(log_msg))
        else:
            self.logger.warning("未检测到任何目标")
        
        # 记录到CSV
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                qr_result["detected"], qr_result.get("content", ""), qr_result.get("confidence", 0),
                color_result["detected"], color_result.get("color", ""), color_result.get("confidence", 0),
                circle_result["detected"], circle_result.get("method", ""), 
                circle_result.get("radius", 0), circle_result.get("confidence", 0)
            ])
    
    def log_debug(self, message: str) -> None:
        """记录调试信息"""
        self.logger.debug(message)
    
    def log_info(self, message: str) -> None:
        """记录一般信息"""
        self.logger.info(message)
    
    def log_warning(self, message: str) -> None:
        """记录警告信息"""
        self.logger.warning(message)
    
    def log_error(self, message: str) -> None:
        """记录错误信息"""
        self.logger.error(message)
    
    def log_exception(self, message: str) -> None:
        """记录异常信息，包含堆栈跟踪"""
        self.logger.exception(message)


if __name__ == "__main__":
    # 简单测试
    param_manager = ParameterManager()
    visualizer = DebugVisualizer(param_manager)
    logger = LoggingManager()
    
    # 读取测试图像
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 模拟检测结果
    qr_result = {
        "detected": True,
        "content": "测试二维码",
        "position": [[50, 50], [150, 50], [150, 150], [50, 150]],
        "confidence": 0.95
    }
    
    color_result = {
        "detected": True,
        "color": "red",
        "center": [320, 240],
        "confidence": 0.85
    }
    
    circle_result = {
        "detected": True,
        "position": [400, 300],
        "radius": 50,
        "confidence": 0.75,
        "method": "combined"
    }
    
    # 生成测试数据
    cv2.putText(test_img, "测试画面", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 在测试图像上绘制一个圆
    cv2.circle(test_img, (400, 300), 50, (0, 0, 255), 2)
    
    # 创建颜色掩码
    color_mask = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(color_mask, (320, 240), 30, 255, -1)
    
    # 创建圆环调试信息
    circle_debug_info = {
        "yolo_box": (350, 250, 450, 350),
        "roi_image": test_img[250:350, 350:450].copy(),
        "processed_roi": None,
        "hough_circles": None
    }
    
    # 记录日志
    logger.log_detection_results(qr_result, color_result, circle_result)
    
    try:
        while True:
            # 显示调试视图
            visualizer.show_debug_view(
                test_img, qr_result, color_result, circle_result, 
                cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR) * 255, circle_debug_info
            )
            
            # 按'q'键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        logger.log_exception(f"调试界面出错: {e}")
    finally:
        visualizer.close() 