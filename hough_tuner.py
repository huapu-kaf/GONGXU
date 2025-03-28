#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
霍夫圆参数调整工具
用于调整视觉识别系统中的霍夫圆检测参数
"""

import cv2
import numpy as np
import argparse
import yaml
import os

class HoughTuner:
    def __init__(self, camera_id=0, resolution=(640, 480), config_file="vision_params_jetson.yaml"):
        self.camera_id = camera_id
        self.resolution = resolution
        self.config_file = config_file
        
        # 加载配置文件
        self.load_config()
        
        # 创建窗口
        cv2.namedWindow("Hough Tuner", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Circles", cv2.WINDOW_NORMAL)
        
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
                
            # 获取霍夫圆参数
            self.hough_params = self.config.get('hough_params', {})
            
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            # 使用默认值
            self.hough_params = {
                "dp": 12,
                "minDist": 30,
                "param1": 100,
                "param2": 25,
                "minRadius": 20,
                "maxRadius": 100,
                "blur_ksize": 5,
                "edge_threshold": 80,
                "max_circles": 8
            }
            self.config = {"hough_params": self.hough_params}
    
    def save_config(self):
        """保存配置文件"""
        try:
            # 更新配置
            self.config['hough_params'] = self.hough_params
            
            # 保存到文件
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                
            print(f"配置已保存到 {self.config_file}")
            
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def create_trackbars(self):
        """创建霍夫圆参数滑动条"""
        # dp参数（实际值为trackbar值/10）
        cv2.createTrackbar("dp (x10)", "Hough Tuner", self.hough_params["dp"], 30, self.on_trackbar_change)
        cv2.createTrackbar("minDist", "Hough Tuner", self.hough_params["minDist"], 200, self.on_trackbar_change)
        cv2.createTrackbar("param1", "Hough Tuner", self.hough_params["param1"], 300, self.on_trackbar_change)
        cv2.createTrackbar("param2", "Hough Tuner", self.hough_params["param2"], 100, self.on_trackbar_change)
        cv2.createTrackbar("minRadius", "Hough Tuner", self.hough_params["minRadius"], 100, self.on_trackbar_change)
        cv2.createTrackbar("maxRadius", "Hough Tuner", self.hough_params["maxRadius"], 300, self.on_trackbar_change)
        cv2.createTrackbar("blur_ksize", "Hough Tuner", self.hough_params["blur_ksize"], 15, self.on_trackbar_change)
        cv2.createTrackbar("edge_threshold", "Hough Tuner", self.hough_params["edge_threshold"], 255, self.on_trackbar_change)
        cv2.createTrackbar("max_circles", "Hough Tuner", self.hough_params["max_circles"], 20, self.on_trackbar_change)
    
    def on_trackbar_change(self, value):
        """滑动条回调函数"""
        # 更新霍夫圆参数
        self.hough_params["dp"] = cv2.getTrackbarPos("dp (x10)", "Hough Tuner")
        self.hough_params["minDist"] = cv2.getTrackbarPos("minDist", "Hough Tuner")
        self.hough_params["param1"] = cv2.getTrackbarPos("param1", "Hough Tuner")
        self.hough_params["param2"] = cv2.getTrackbarPos("param2", "Hough Tuner")
        self.hough_params["minRadius"] = cv2.getTrackbarPos("minRadius", "Hough Tuner")
        self.hough_params["maxRadius"] = cv2.getTrackbarPos("maxRadius", "Hough Tuner")
        
        # 确保blur_ksize是奇数
        blur_ksize = cv2.getTrackbarPos("blur_ksize", "Hough Tuner")
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        self.hough_params["blur_ksize"] = blur_ksize
        
        self.hough_params["edge_threshold"] = cv2.getTrackbarPos("edge_threshold", "Hough Tuner")
        self.hough_params["max_circles"] = cv2.getTrackbarPos("max_circles", "Hough Tuner")
    
    def detect_circles(self, frame):
        """使用当前参数检测圆"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 中值滤波去噪
        blur_ksize = self.hough_params["blur_ksize"]
        if blur_ksize % 2 == 0:  # 确保核大小为奇数
            blur_ksize += 1
        blurred = cv2.medianBlur(gray, blur_ksize)
        
        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # 边缘检测
        edge_threshold = self.hough_params["edge_threshold"]
        edges = cv2.Canny(enhanced, edge_threshold // 2, edge_threshold)
        
        # 霍夫圆检测
        dp = max(1.0, self.hough_params["dp"] / 10.0)  # 实际dp值
        circles = cv2.HoughCircles(
            enhanced,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=self.hough_params["minDist"],
            param1=self.hough_params["param1"],
            param2=self.hough_params["param2"],
            minRadius=self.hough_params["minRadius"],
            maxRadius=self.hough_params["maxRadius"]
        )
        
        return edges, circles
    
    def run(self):
        """运行霍夫圆参数调整器"""
        print("霍夫圆参数调整工具")
        print("按's'保存当前配置")
        print("按'ESC'退出")
        
        while True:
            # 读取一帧
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取图像")
                break
            
            # 检测圆
            edges, circles = self.detect_circles(frame)
            
            # 显示原始图像
            cv2.imshow("Hough Tuner", frame)
            
            # 显示边缘图像
            cv2.imshow("Edges", edges)
            
            # 显示检测到的圆
            result = frame.copy()
            if circles is not None:
                # 转换为整数坐标
                circles = np.uint16(np.around(circles))
                
                # 限制最大检测圆数量
                max_circles = min(self.hough_params["max_circles"], len(circles[0]))
                
                # 绘制检测到的圆
                for i, circle in enumerate(circles[0,:max_circles]):
                    center_x, center_y, radius = circle
                    
                    # 绘制圆轮廓
                    cv2.circle(result, (center_x, center_y), radius, (0, 255, 0), 2)
                    
                    # 绘制圆心
                    cv2.circle(result, (center_x, center_y), 2, (0, 0, 255), 3)
                    
                    # 显示半径
                    cv2.putText(result, f"r={radius}", (center_x - 20, center_y - radius - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            cv2.imshow("Circles", result)
            
            # 显示当前参数
            info_image = np.zeros((300, 400, 3), dtype=np.uint8)
            
            params = [
                f"dp: {self.hough_params['dp']/10:.1f}",
                f"minDist: {self.hough_params['minDist']}",
                f"param1: {self.hough_params['param1']}",
                f"param2: {self.hough_params['param2']}",
                f"minRadius: {self.hough_params['minRadius']}",
                f"maxRadius: {self.hough_params['maxRadius']}",
                f"blur_ksize: {self.hough_params['blur_ksize']}",
                f"edge_threshold: {self.hough_params['edge_threshold']}",
                f"max_circles: {self.hough_params['max_circles']}"
            ]
            
            for i, param in enumerate(params):
                cv2.putText(info_image, param, (10, 30 + i * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(info_image, "按's'保存, 'ESC'退出", (10, 280),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Parameters", info_image)
            
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
    parser = argparse.ArgumentParser(description="霍夫圆参数调整工具")
    parser.add_argument("--camera", type=int, default=0, help="摄像头ID")
    parser.add_argument("--width", type=int, default=640, help="图像宽度")
    parser.add_argument("--height", type=int, default=480, help="图像高度")
    parser.add_argument("--config", type=str, default="vision_params_jetson.yaml", help="配置文件路径")
    args = parser.parse_args()
    
    try:
        tuner = HoughTuner(
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