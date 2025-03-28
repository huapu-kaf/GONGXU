#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HSV颜色阈值调整工具
用于调整视觉识别系统中的HSV颜色阈值
"""

import cv2
import numpy as np
import argparse
import yaml
import os

class HSVTuner:
    def __init__(self, camera_id=0, resolution=(640, 480), config_file="vision_params_jetson.yaml"):
        self.camera_id = camera_id
        self.resolution = resolution
        self.config_file = config_file
        
        # 加载配置文件
        self.load_config()
        
        # 创建窗口
        cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
        
        # 创建颜色选择器
        self.create_color_selector()
        
        # 创建滑动条
        self.create_trackbars()
        
        # 打开摄像头
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        if not self.cap.isOpened():
            raise ValueError(f"无法打开摄像头 {camera_id}")
    
    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
                
            # 获取HSV阈值
            self.hsv_thresholds = self.config.get('hsv_thresholds', {})
            
            # 初始化当前颜色
            self.current_color = "red"
            
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            # 使用默认值
            self.hsv_thresholds = {
                "red_lower_h": 0, "red_lower_s": 100, "red_lower_v": 100,
                "red_upper_h": 10, "red_upper_s": 255, "red_upper_v": 255,
                "red_lower2_h": 160, "red_lower2_s": 100, "red_lower2_v": 100,
                "red_upper2_h": 180, "red_upper2_s": 255, "red_upper2_v": 255,
                "green_lower_h": 35, "green_lower_s": 80, "green_lower_v": 80,
                "green_upper_h": 85, "green_upper_s": 255, "green_upper_v": 255,
                "blue_lower_h": 90, "blue_lower_s": 80, "blue_lower_v": 80,
                "blue_upper_h": 130, "blue_upper_s": 255, "blue_upper_v": 255
            }
            self.config = {"hsv_thresholds": self.hsv_thresholds}
            self.current_color = "red"
    
    def save_config(self):
        """保存配置文件"""
        try:
            # 更新配置
            self.config['hsv_thresholds'] = self.hsv_thresholds
            
            # 保存到文件
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                
            print(f"配置已保存到 {self.config_file}")
            
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def create_color_selector(self):
        """创建颜色选择器"""
        cv2.createTrackbar("Color", "HSV Tuner", 0, 2, self.on_color_change)
        # 0: 红色, 1: 绿色, 2: 蓝色
    
    def create_trackbars(self):
        """创建HSV滑动条"""
        # 创建滑动条
        if self.current_color == "red":
            # 红色有两个范围
            # 范围1
            cv2.createTrackbar("Red Lower H", "HSV Tuner", self.hsv_thresholds["red_lower_h"], 180, self.on_trackbar_change)
            cv2.createTrackbar("Red Lower S", "HSV Tuner", self.hsv_thresholds["red_lower_s"], 255, self.on_trackbar_change)
            cv2.createTrackbar("Red Lower V", "HSV Tuner", self.hsv_thresholds["red_lower_v"], 255, self.on_trackbar_change)
            cv2.createTrackbar("Red Upper H", "HSV Tuner", self.hsv_thresholds["red_upper_h"], 180, self.on_trackbar_change)
            cv2.createTrackbar("Red Upper S", "HSV Tuner", self.hsv_thresholds["red_upper_s"], 255, self.on_trackbar_change)
            cv2.createTrackbar("Red Upper V", "HSV Tuner", self.hsv_thresholds["red_upper_v"], 255, self.on_trackbar_change)
            
            # 范围2
            cv2.createTrackbar("Red Lower2 H", "HSV Tuner", self.hsv_thresholds["red_lower2_h"], 180, self.on_trackbar_change)
            cv2.createTrackbar("Red Lower2 S", "HSV Tuner", self.hsv_thresholds["red_lower2_s"], 255, self.on_trackbar_change)
            cv2.createTrackbar("Red Lower2 V", "HSV Tuner", self.hsv_thresholds["red_lower2_v"], 255, self.on_trackbar_change)
            cv2.createTrackbar("Red Upper2 H", "HSV Tuner", self.hsv_thresholds["red_upper2_h"], 180, self.on_trackbar_change)
            cv2.createTrackbar("Red Upper2 S", "HSV Tuner", self.hsv_thresholds["red_upper2_s"], 255, self.on_trackbar_change)
            cv2.createTrackbar("Red Upper2 V", "HSV Tuner", self.hsv_thresholds["red_upper2_v"], 255, self.on_trackbar_change)
            
        elif self.current_color == "green":
            cv2.createTrackbar("Green Lower H", "HSV Tuner", self.hsv_thresholds["green_lower_h"], 180, self.on_trackbar_change)
            cv2.createTrackbar("Green Lower S", "HSV Tuner", self.hsv_thresholds["green_lower_s"], 255, self.on_trackbar_change)
            cv2.createTrackbar("Green Lower V", "HSV Tuner", self.hsv_thresholds["green_lower_v"], 255, self.on_trackbar_change)
            cv2.createTrackbar("Green Upper H", "HSV Tuner", self.hsv_thresholds["green_upper_h"], 180, self.on_trackbar_change)
            cv2.createTrackbar("Green Upper S", "HSV Tuner", self.hsv_thresholds["green_upper_s"], 255, self.on_trackbar_change)
            cv2.createTrackbar("Green Upper V", "HSV Tuner", self.hsv_thresholds["green_upper_v"], 255, self.on_trackbar_change)
            
        elif self.current_color == "blue":
            cv2.createTrackbar("Blue Lower H", "HSV Tuner", self.hsv_thresholds["blue_lower_h"], 180, self.on_trackbar_change)
            cv2.createTrackbar("Blue Lower S", "HSV Tuner", self.hsv_thresholds["blue_lower_s"], 255, self.on_trackbar_change)
            cv2.createTrackbar("Blue Lower V", "HSV Tuner", self.hsv_thresholds["blue_lower_v"], 255, self.on_trackbar_change)
            cv2.createTrackbar("Blue Upper H", "HSV Tuner", self.hsv_thresholds["blue_upper_h"], 180, self.on_trackbar_change)
            cv2.createTrackbar("Blue Upper S", "HSV Tuner", self.hsv_thresholds["blue_upper_s"], 255, self.on_trackbar_change)
            cv2.createTrackbar("Blue Upper V", "HSV Tuner", self.hsv_thresholds["blue_upper_v"], 255, self.on_trackbar_change)
    
    def on_color_change(self, value):
        """颜色选择器回调函数"""
        colors = ["red", "green", "blue"]
        self.current_color = colors[value]
        
        # 清除所有滑动条
        cv2.destroyWindow("HSV Tuner")
        cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)
        
        # 重新创建颜色选择器和滑动条
        self.create_color_selector()
        cv2.setTrackbarPos("Color", "HSV Tuner", value)
        self.create_trackbars()
    
    def on_trackbar_change(self, value):
        """滑动条回调函数"""
        # 更新HSV阈值
        if self.current_color == "red":
            self.hsv_thresholds["red_lower_h"] = cv2.getTrackbarPos("Red Lower H", "HSV Tuner")
            self.hsv_thresholds["red_lower_s"] = cv2.getTrackbarPos("Red Lower S", "HSV Tuner")
            self.hsv_thresholds["red_lower_v"] = cv2.getTrackbarPos("Red Lower V", "HSV Tuner")
            self.hsv_thresholds["red_upper_h"] = cv2.getTrackbarPos("Red Upper H", "HSV Tuner")
            self.hsv_thresholds["red_upper_s"] = cv2.getTrackbarPos("Red Upper S", "HSV Tuner")
            self.hsv_thresholds["red_upper_v"] = cv2.getTrackbarPos("Red Upper V", "HSV Tuner")
            
            self.hsv_thresholds["red_lower2_h"] = cv2.getTrackbarPos("Red Lower2 H", "HSV Tuner")
            self.hsv_thresholds["red_lower2_s"] = cv2.getTrackbarPos("Red Lower2 S", "HSV Tuner")
            self.hsv_thresholds["red_lower2_v"] = cv2.getTrackbarPos("Red Lower2 V", "HSV Tuner")
            self.hsv_thresholds["red_upper2_h"] = cv2.getTrackbarPos("Red Upper2 H", "HSV Tuner")
            self.hsv_thresholds["red_upper2_s"] = cv2.getTrackbarPos("Red Upper2 S", "HSV Tuner")
            self.hsv_thresholds["red_upper2_v"] = cv2.getTrackbarPos("Red Upper2 V", "HSV Tuner")
            
        elif self.current_color == "green":
            self.hsv_thresholds["green_lower_h"] = cv2.getTrackbarPos("Green Lower H", "HSV Tuner")
            self.hsv_thresholds["green_lower_s"] = cv2.getTrackbarPos("Green Lower S", "HSV Tuner")
            self.hsv_thresholds["green_lower_v"] = cv2.getTrackbarPos("Green Lower V", "HSV Tuner")
            self.hsv_thresholds["green_upper_h"] = cv2.getTrackbarPos("Green Upper H", "HSV Tuner")
            self.hsv_thresholds["green_upper_s"] = cv2.getTrackbarPos("Green Upper S", "HSV Tuner")
            self.hsv_thresholds["green_upper_v"] = cv2.getTrackbarPos("Green Upper V", "HSV Tuner")
            
        elif self.current_color == "blue":
            self.hsv_thresholds["blue_lower_h"] = cv2.getTrackbarPos("Blue Lower H", "HSV Tuner")
            self.hsv_thresholds["blue_lower_s"] = cv2.getTrackbarPos("Blue Lower S", "HSV Tuner")
            self.hsv_thresholds["blue_lower_v"] = cv2.getTrackbarPos("Blue Lower V", "HSV Tuner")
            self.hsv_thresholds["blue_upper_h"] = cv2.getTrackbarPos("Blue Upper H", "HSV Tuner")
            self.hsv_thresholds["blue_upper_s"] = cv2.getTrackbarPos("Blue Upper S", "HSV Tuner")
            self.hsv_thresholds["blue_upper_v"] = cv2.getTrackbarPos("Blue Upper V", "HSV Tuner")
    
    def get_mask(self, hsv_image):
        """根据当前HSV阈值获取掩码"""
        mask = None
        
        if self.current_color == "red":
            # 红色有两个范围
            lower1 = np.array([self.hsv_thresholds["red_lower_h"], 
                              self.hsv_thresholds["red_lower_s"], 
                              self.hsv_thresholds["red_lower_v"]])
            upper1 = np.array([self.hsv_thresholds["red_upper_h"], 
                              self.hsv_thresholds["red_upper_s"], 
                              self.hsv_thresholds["red_upper_v"]])
            
            lower2 = np.array([self.hsv_thresholds["red_lower2_h"], 
                              self.hsv_thresholds["red_lower2_s"], 
                              self.hsv_thresholds["red_lower2_v"]])
            upper2 = np.array([self.hsv_thresholds["red_upper2_h"], 
                              self.hsv_thresholds["red_upper2_s"], 
                              self.hsv_thresholds["red_upper2_v"]])
            
            mask1 = cv2.inRange(hsv_image, lower1, upper1)
            mask2 = cv2.inRange(hsv_image, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
            
        elif self.current_color == "green":
            lower = np.array([self.hsv_thresholds["green_lower_h"], 
                             self.hsv_thresholds["green_lower_s"], 
                             self.hsv_thresholds["green_lower_v"]])
            upper = np.array([self.hsv_thresholds["green_upper_h"], 
                             self.hsv_thresholds["green_upper_s"], 
                             self.hsv_thresholds["green_upper_v"]])
            
            mask = cv2.inRange(hsv_image, lower, upper)
            
        elif self.current_color == "blue":
            lower = np.array([self.hsv_thresholds["blue_lower_h"], 
                             self.hsv_thresholds["blue_lower_s"], 
                             self.hsv_thresholds["blue_lower_v"]])
            upper = np.array([self.hsv_thresholds["blue_upper_h"], 
                             self.hsv_thresholds["blue_upper_s"], 
                             self.hsv_thresholds["blue_upper_v"]])
            
            mask = cv2.inRange(hsv_image, lower, upper)
        
        return mask
    
    def run(self):
        """运行HSV调整器"""
        print("HSV颜色阈值调整工具")
        print("按's'保存当前配置")
        print("按'ESC'退出")
        
        while True:
            # 读取一帧
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取图像")
                break
            
            # 转换为HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 获取掩码
            mask = self.get_mask(hsv)
            
            # 应用掩码
            result = cv2.bitwise_and(frame, frame, mask=mask)
            
            # 显示结果
            cv2.imshow("HSV Tuner", frame)
            cv2.imshow("Mask", result)
            
            # 显示当前颜色和阈值
            info_image = np.zeros((200, 400, 3), dtype=np.uint8)
            cv2.putText(info_image, f"当前颜色: {self.current_color}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if self.current_color == "red":
                cv2.putText(info_image, f"范围1: H[{self.hsv_thresholds['red_lower_h']}-{self.hsv_thresholds['red_upper_h']}]", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(info_image, f"      S[{self.hsv_thresholds['red_lower_s']}-{self.hsv_thresholds['red_upper_s']}]", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(info_image, f"      V[{self.hsv_thresholds['red_lower_v']}-{self.hsv_thresholds['red_upper_v']}]", 
                           (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                           
                cv2.putText(info_image, f"范围2: H[{self.hsv_thresholds['red_lower2_h']}-{self.hsv_thresholds['red_upper2_h']}]", 
                           (200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(info_image, f"      S[{self.hsv_thresholds['red_lower2_s']}-{self.hsv_thresholds['red_upper2_s']}]", 
                           (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(info_image, f"      V[{self.hsv_thresholds['red_lower2_v']}-{self.hsv_thresholds['red_upper2_v']}]", 
                           (200, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
            elif self.current_color == "green":
                cv2.putText(info_image, f"H: [{self.hsv_thresholds['green_lower_h']}-{self.hsv_thresholds['green_upper_h']}]", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(info_image, f"S: [{self.hsv_thresholds['green_lower_s']}-{self.hsv_thresholds['green_upper_s']}]", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(info_image, f"V: [{self.hsv_thresholds['green_lower_v']}-{self.hsv_thresholds['green_upper_v']}]", 
                           (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            elif self.current_color == "blue":
                cv2.putText(info_image, f"H: [{self.hsv_thresholds['blue_lower_h']}-{self.hsv_thresholds['blue_upper_h']}]", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(info_image, f"S: [{self.hsv_thresholds['blue_lower_s']}-{self.hsv_thresholds['blue_upper_s']}]", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(info_image, f"V: [{self.hsv_thresholds['blue_lower_v']}-{self.hsv_thresholds['blue_upper_v']}]", 
                           (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.putText(info_image, "按's'保存, 'ESC'退出", (10, 170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Info", info_image)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('s'):  # 保存配置
                self.save_config()
    
    def cleanup(self):
        """清理资源"""
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="HSV颜色阈值调整工具")
    parser.add_argument("--camera", type=int, default=0, help="摄像头ID")
    parser.add_argument("--width", type=int, default=640, help="图像宽度")
    parser.add_argument("--height", type=int, default=480, help="图像高度")
    parser.add_argument("--config", type=str, default="vision_params_jetson.yaml", help="配置文件路径")
    args = parser.parse_args()
    
    try:
        tuner = HSVTuner(
            camera_id=args.camera,
            resolution=(args.width, args.height),
            config_file=args.config
        )
        
        tuner.run()
        
    except Exception as e:
        print(f"错误: {e}")
    finally:
        if 'tuner' in locals():
            tuner.cleanup()

if __name__ == "__main__":
    main() 