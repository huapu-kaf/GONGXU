#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
import os
import threading
import yaml
import pickle
import psutil
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
import logging

class AntiInterferenceSystem:
    """抗干扰系统"""
    
    def __init__(self):
        """初始化抗干扰系统"""
        # 频闪检测配置
        self.flicker_detection = {
            "enabled": True,
            "history_length": 5,   # 保存的历史帧数
            "intensity_threshold": 30,  # 像素强度变化阈值
            "pixel_percentage": 0.05,   # 变化像素占比阈值
            "frame_history": []
        }
        
        # 反光抑制配置
        self.glare_suppression = {
            "enabled": True,
            "intensity_threshold": 200,  # 高亮度阈值
            "pixel_percentage": 0.02     # 高亮度像素占比阈值
        }
        
        # 白平衡校准配置
        self.white_balance = {
            "enabled": True,
            "auto_adjust": True,
            "r_gain": 1.0,
            "g_gain": 1.0,
            "b_gain": 1.0,
            "update_interval": 30,  # 更新间隔（帧数）
            "frame_count": 0
        }
        
        # 背景差分配置
        self.background_subtraction = {
            "enabled": False,  # 默认关闭，因为这可能会移除静止的目标
            "subtractor": cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=36, detectShadows=False),
            "learning_rate": 0.01,
            "morph_kernel": np.ones((5, 5), np.uint8)
        }
        
        self.logger = logging.getLogger("vision_system.anti_interference")
    
    def detect_flicker(self, frame: np.ndarray) -> bool:
        """
        检测图像中的频闪干扰
        
        Args:
            frame: 输入图像
            
        Returns:
            是否检测到频闪
        """
        if not self.flicker_detection["enabled"] or frame is None:
            return False
        
        # 转为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 添加到历史
        self.flicker_detection["frame_history"].append(gray)
        
        # 保持固定长度
        history_length = self.flicker_detection["history_length"]
        if len(self.flicker_detection["frame_history"]) > history_length:
            self.flicker_detection["frame_history"].pop(0)
        
        # 如果历史帧不足，无法检测
        if len(self.flicker_detection["frame_history"]) < 2:
            return False
        
        # 计算当前帧与前一帧的差异
        prev_frame = self.flicker_detection["frame_history"][-2]
        frame_diff = cv2.absdiff(gray, prev_frame)
        
        # 计算变化明显的像素数量
        threshold = self.flicker_detection["intensity_threshold"]
        significant_change = np.sum(frame_diff > threshold)
        total_pixels = frame_diff.size
        change_percentage = significant_change / total_pixels
        
        # 判断是否为频闪
        flicker_detected = change_percentage > self.flicker_detection["pixel_percentage"]
        
        if flicker_detected:
            self.logger.warning(f"检测到频闪干扰，变化像素占比: {change_percentage:.4f}")
        
        return flicker_detected
    
    def suppress_glare(self, frame: np.ndarray) -> np.ndarray:
        """
        抑制图像中的反光
        
        Args:
            frame: 输入图像
            
        Returns:
            处理后的图像
        """
        if not self.glare_suppression["enabled"] or frame is None:
            return frame
        
        # 转为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 创建反光掩码（高亮度区域）
        threshold = self.glare_suppression["intensity_threshold"]
        glare_mask = (v > threshold).astype(np.uint8) * 255
        
        # 计算反光区域占比
        glare_percentage = np.sum(glare_mask) / (glare_mask.size * 255)
        
        # 如果反光区域过大，进行处理
        if glare_percentage > self.glare_suppression["pixel_percentage"]:
            self.logger.info(f"检测到反光区域，占比: {glare_percentage:.4f}")
            
            # 对反光区域进行处理
            # 1. 对V通道应用自适应直方图均衡化
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            v_equalized = clahe.apply(v)
            
            # 2. 只在反光区域应用均衡化
            v_adjusted = v.copy()
            v_adjusted[glare_mask > 0] = v_equalized[glare_mask > 0]
            
            # 合并回HSV图像
            hsv_adjusted = cv2.merge([h, s, v_adjusted])
            
            # 转回BGR颜色空间
            result = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
            return result
        
        return frame
    
    def adjust_white_balance(self, frame: np.ndarray) -> np.ndarray:
        """
        调整图像的白平衡
        
        Args:
            frame: 输入图像
            
        Returns:
            白平衡调整后的图像
        """
        if not self.white_balance["enabled"] or frame is None:
            return frame
        
        # 增加帧计数
        self.white_balance["frame_count"] += 1
        
        # 是否需要自动更新白平衡参数
        if (self.white_balance["auto_adjust"] and 
            self.white_balance["frame_count"] % self.white_balance["update_interval"] == 0):
            self._calculate_white_balance(frame)
        
        # 分离BGR通道
        b, g, r = cv2.split(frame)
        
        # 应用增益系数
        r_adjusted = cv2.convertScaleAbs(r, alpha=self.white_balance["r_gain"], beta=0)
        g_adjusted = cv2.convertScaleAbs(g, alpha=self.white_balance["g_gain"], beta=0)
        b_adjusted = cv2.convertScaleAbs(b, alpha=self.white_balance["b_gain"], beta=0)
        
        # 合并通道
        balanced_frame = cv2.merge([b_adjusted, g_adjusted, r_adjusted])
        
        return balanced_frame
    
    def _calculate_white_balance(self, frame: np.ndarray) -> None:
        """
        计算白平衡参数
        
        Args:
            frame: 输入图像
        """
        # 使用灰度世界假设来计算白平衡参数
        b, g, r = cv2.split(frame)
        
        # 计算每个通道的平均值
        r_avg = np.mean(r)
        g_avg = np.mean(g)
        b_avg = np.mean(b)
        
        # 以绿色通道为基准调整增益
        if g_avg > 0:
            self.white_balance["r_gain"] = g_avg / r_avg if r_avg > 0 else 1.0
            self.white_balance["b_gain"] = g_avg / b_avg if b_avg > 0 else 1.0
            self.white_balance["g_gain"] = 1.0
        
        # 限制增益范围，防止过度调整
        for channel in ["r_gain", "g_gain", "b_gain"]:
            self.white_balance[channel] = max(0.5, min(2.0, self.white_balance[channel]))
        
        self.logger.debug(f"更新白平衡参数: R={self.white_balance['r_gain']:.2f}, "
                        f"G={self.white_balance['g_gain']:.2f}, B={self.white_balance['b_gain']:.2f}")
    
    def remove_static_background(self, frame: np.ndarray) -> np.ndarray:
        """
        去除静态背景
        
        Args:
            frame: 输入图像
            
        Returns:
            前景掩码
        """
        if not self.background_subtraction["enabled"] or frame is None:
            return None
        
        # 应用背景减除器
        fg_mask = self.background_subtraction["subtractor"].apply(
            frame, learningRate=self.background_subtraction["learning_rate"]
        )
        
        # 应用形态学操作改善掩码质量
        kernel = self.background_subtraction["morph_kernel"]
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        return fg_mask
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        应用所有抗干扰处理
        
        Args:
            frame: 输入图像
            
        Returns:
            处理后的图像
        """
        if frame is None:
            return None
        
        # 复制输入图像
        processed = frame.copy()
        
        # 检测频闪
        flicker_detected = self.detect_flicker(processed)
        
        # 如果没有频闪，则应用其他增强
        if not flicker_detected:
            # 抑制反光
            processed = self.suppress_glare(processed)
            
            # 调整白平衡
            processed = self.adjust_white_balance(processed)
        
        return processed


class QuickStartManager:
    """快速启动管理器"""
    
    def __init__(self, cache_folder: str = "cache"):
        """
        初始化快速启动管理器
        
        Args:
            cache_folder: 缓存文件夹路径
        """
        self.cache_folder = cache_folder
        self.model_cache = {}
        self.param_cache = {}
        
        # 创建缓存文件夹
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)
        
        self.logger = logging.getLogger("vision_system.quick_start")
        self.logger.info(f"初始化快速启动管理器，缓存目录: {cache_folder}")
    
    def cache_model(self, model_name: str, model: Any) -> bool:
        """
        缓存模型到内存和磁盘
        
        Args:
            model_name: 模型名称
            model: 模型对象
            
        Returns:
            是否成功缓存
        """
        try:
            # 存储到内存缓存
            self.model_cache[model_name] = model
            
            # 存储到磁盘缓存
            model_path = os.path.join(self.cache_folder, f"{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            self.logger.info(f"模型 {model_name} 已缓存")
            return True
        except Exception as e:
            self.logger.error(f"缓存模型 {model_name} 失败: {e}")
            return False
    
    def load_model_from_cache(self, model_name: str) -> Any:
        """
        从缓存加载模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            缓存的模型对象，如果不存在则返回None
        """
        # 首先检查内存缓存
        if model_name in self.model_cache:
            self.logger.info(f"从内存加载模型 {model_name}")
            return self.model_cache[model_name]
        
        # 然后检查磁盘缓存
        model_path = os.path.join(self.cache_folder, f"{model_name}.pkl")
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                # 加载到内存缓存
                self.model_cache[model_name] = model
                
                self.logger.info(f"从磁盘加载模型 {model_name}")
                return model
            except Exception as e:
                self.logger.error(f"从磁盘加载模型 {model_name} 失败: {e}")
        
        return None
    
    def cache_parameters(self, params_name: str, params: Dict[str, Any]) -> bool:
        """
        缓存参数到内存和磁盘
        
        Args:
            params_name: 参数名称
            params: 参数字典
            
        Returns:
            是否成功缓存
        """
        try:
            # 存储到内存缓存
            self.param_cache[params_name] = params
            
            # 存储到磁盘缓存
            params_path = os.path.join(self.cache_folder, f"{params_name}.yaml")
            with open(params_path, 'w', encoding='utf-8') as f:
                yaml.dump(params, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"参数 {params_name} 已缓存")
            return True
        except Exception as e:
            self.logger.error(f"缓存参数 {params_name} 失败: {e}")
            return False
    
    def load_parameters_from_cache(self, params_name: str) -> Dict[str, Any]:
        """
        从缓存加载参数
        
        Args:
            params_name: 参数名称
            
        Returns:
            缓存的参数字典，如果不存在则返回空字典
        """
        # 首先检查内存缓存
        if params_name in self.param_cache:
            self.logger.info(f"从内存加载参数 {params_name}")
            return self.param_cache[params_name]
        
        # 然后检查磁盘缓存
        params_path = os.path.join(self.cache_folder, f"{params_name}.yaml")
        if os.path.exists(params_path):
            try:
                with open(params_path, 'r', encoding='utf-8') as f:
                    params = yaml.safe_load(f)
                
                # 加载到内存缓存
                if params:
                    self.param_cache[params_name] = params
                    self.logger.info(f"从磁盘加载参数 {params_name}")
                    return params
            except Exception as e:
                self.logger.error(f"从磁盘加载参数 {params_name} 失败: {e}")
        
        return {}
    
    def check_hardware(self) -> Dict[str, Any]:
        """
        检查硬件状态
        
        Returns:
            硬件状态信息字典
        """
        hardware_info = {
            "cpu_usage": psutil.cpu_percent(),
            "memory_available": psutil.virtual_memory().available / (1024 * 1024 * 1024),  # GB
            "disk_free": psutil.disk_usage('/').free / (1024 * 1024 * 1024),  # GB
            "camera_available": False,
            "gpu_available": False
        }
        
        # 检查摄像头
        try:
            cap = cv2.VideoCapture(0)
            hardware_info["camera_available"] = cap.isOpened()
            cap.release()
        except Exception:
            pass
        
        # 检查GPU是否可用（仅针对CUDA）
        try:
            import torch
            hardware_info["gpu_available"] = torch.cuda.is_available()
            if hardware_info["gpu_available"]:
                hardware_info["gpu_name"] = torch.cuda.get_device_name(0)
                hardware_info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)  # GB
        except (ImportError, Exception):
            pass
        
        return hardware_info
    
    def switch_configuration(self, config_name: str) -> Dict[str, Any]:
        """
        切换到备用配置
        
        Args:
            config_name: 配置名称
            
        Returns:
            加载的配置参数
        """
        return self.load_parameters_from_cache(config_name)


class EmergencyHandler:
    """应急处理机制"""
    
    def __init__(self, fallback_config_path: str = "configs/fallback.yaml"):
        """
        初始化应急处理器
        
        Args:
            fallback_config_path: 备用配置文件路径
        """
        self.fallback_config_path = fallback_config_path
        
        # 初始化颜色置信度白名单
        self.color_confidence_whitelist = {
            "red": 0.7,    # 要求红色的最低置信度
            "green": 0.7,  # 要求绿色的最低置信度
            "blue": 0.7    # 要求蓝色的最低置信度
        }
        
        # 监测计时器
        self.watchdog_timer = time.time()
        self.watchdog_timeout = 5.0  # 5秒超时
        
        # 心跳监测
        self.process_heartbeats = {}
        
        # 应急状态
        self.emergency_state = False
        
        # 故障计数
        self.failure_count = 0
        self.max_failures = 3
        
        self.logger = logging.getLogger("vision_system.emergency")
        self.logger.info("初始化应急处理机制")
    
    def reset_watchdog(self) -> None:
        """重置看门狗定时器"""
        self.watchdog_timer = time.time()
    
    def check_watchdog(self) -> bool:
        """
        检查看门狗定时器是否超时
        
        Returns:
            是否超时
        """
        elapsed = time.time() - self.watchdog_timer
        if elapsed > self.watchdog_timeout:
            self.logger.warning(f"看门狗超时！已经 {elapsed:.1f} 秒没有复位")
            self.emergency_state = True
            return True
        return False
    
    def register_heartbeat(self, process_name: str) -> None:
        """
        注册进程心跳
        
        Args:
            process_name: 进程名称
        """
        self.process_heartbeats[process_name] = time.time()
    
    def check_heartbeats(self, max_interval: float = 2.0) -> List[str]:
        """
        检查所有进程心跳
        
        Args:
            max_interval: 最大允许间隔（秒）
            
        Returns:
            超时的进程名称列表
        """
        current_time = time.time()
        timed_out = []
        
        for process, last_beat in self.process_heartbeats.items():
            if current_time - last_beat > max_interval:
                timed_out.append(process)
                self.logger.warning(f"进程 {process} 心跳超时")
        
        if timed_out:
            self.emergency_state = True
        
        return timed_out
    
    def validate_color_result(self, color_result: Dict[str, Any]) -> bool:
        """
        验证颜色分类结果是否可靠
        
        Args:
            color_result: 颜色分类结果
            
        Returns:
            结果是否可靠
        """
        if not color_result["detected"]:
            return False
        
        color = color_result["color"]
        confidence = color_result["confidence"]
        
        # 检查颜色是否在白名单中，且置信度超过阈值
        if color in self.color_confidence_whitelist:
            threshold = self.color_confidence_whitelist[color]
            return confidence >= threshold
        
        return False
    
    def load_fallback_config(self) -> Dict[str, Any]:
        """
        加载备用配置
        
        Returns:
            备用配置参数
        """
        if os.path.exists(self.fallback_config_path):
            try:
                with open(self.fallback_config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                self.logger.info("已加载备用配置")
                return config
            except Exception as e:
                self.logger.error(f"加载备用配置失败: {e}")
        
        # 返回默认备用配置
        return {
            "hsv_thresholds": {
                "red_lower_h": 0,
                "red_lower_s": 100,
                "red_lower_v": 100,
                "red_upper_h": 10,
                "red_upper_s": 255,
                "red_upper_v": 255,
                "red_lower2_h": 160,
                "red_lower2_s": 100,
                "red_lower2_v": 100,
                "red_upper2_h": 180,
                "red_upper2_s": 255,
                "red_upper2_v": 255,
                "green_lower_h": 35,
                "green_lower_s": 100,
                "green_lower_v": 100,
                "green_upper_h": 85,
                "green_upper_s": 255,
                "green_upper_v": 255,
                "blue_lower_h": 100,
                "blue_lower_s": 100,
                "blue_lower_v": 100,
                "blue_upper_h": 130,
                "blue_upper_s": 255,
                "blue_upper_v": 255
            },
            "hough_params": {
                "dp": 1.2,
                "minDist": 50,
                "param1": 100,
                "param2": 30,
                "minRadius": 10,
                "maxRadius": 100,
                "blur_ksize": 7,
                "roi_padding": 20
            }
        }
    
    def record_failure(self) -> bool:
        """
        记录一次故障，判断是否达到最大故障次数
        
        Returns:
            是否达到最大故障次数
        """
        self.failure_count += 1
        if self.failure_count >= self.max_failures:
            self.logger.error(f"故障次数达到上限 ({self.max_failures})，进入应急状态")
            self.emergency_state = True
            return True
        return False
    
    def reset_failure_count(self) -> None:
        """重置故障计数"""
        self.failure_count = 0
    
    def is_in_emergency(self) -> bool:
        """
        检查是否处于应急状态
        
        Returns:
            是否处于应急状态
        """
        return self.emergency_state
    
    def exit_emergency(self) -> None:
        """退出应急状态"""
        self.emergency_state = False
        self.reset_failure_count()
        self.reset_watchdog()
        self.logger.info("已退出应急状态")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 测试抗干扰系统
    print("测试抗干扰系统...")
    anti_interference = AntiInterferenceSystem()
    
    # 创建测试图像（模拟反光区域）
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    # 添加一个高亮区域模拟反光
    cv2.circle(test_img, (320, 240), 50, (255, 255, 255), -1)
    
    # 处理图像
    processed = anti_interference.process_frame(test_img)
    
    # 显示原始图像和处理后的图像
    cv2.imshow("Original", test_img)
    cv2.imshow("Processed", processed)
    cv2.waitKey(0)
    
    # 测试快速启动管理器
    print("测试快速启动管理器...")
    quick_start = QuickStartManager()
    
    # 检查硬件
    hw_info = quick_start.check_hardware()
    print("硬件信息:")
    for key, value in hw_info.items():
        print(f"  {key}: {value}")
    
    # 测试应急处理机制
    print("测试应急处理机制...")
    emergency = EmergencyHandler()
    
    # 注册一些进程心跳
    emergency.register_heartbeat("main")
    emergency.register_heartbeat("qr_detector")
    emergency.register_heartbeat("color_classifier")
    
    # 检查心跳
    print("等待3秒模拟心跳超时...")
    time.sleep(3)
    timed_out = emergency.check_heartbeats(max_interval=2.0)
    print(f"超时的进程: {timed_out}")
    
    # 加载备用配置
    fallback_config = emergency.load_fallback_config()
    print("备用配置已加载")
    
    cv2.destroyAllWindows() 