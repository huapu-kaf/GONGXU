#!/usr/bin/env python
# -*- coding: utf-8 -*-

import serial
import time
import threading
import queue
from typing import Optional, Dict, List, Any, Tuple, Union
import logging

class SerialManager:
    """串口通信管理器"""
    
    def __init__(self, port: Optional[str] = None, 
                 baudrate: int = 115200, 
                 timeout: float = 1.0,
                 reconnect_interval: float = 5.0,
                 logger = None):
        """
        初始化串口管理器
        
        Args:
            port: 串口号，如'COM3'或'/dev/ttyUSB0'
            baudrate: 波特率，默认115200
            timeout: 串口超时时间，默认1秒
            reconnect_interval: 断开后重连间隔，默认5秒
            logger: 日志记录器
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.reconnect_interval = reconnect_interval
        self.logger = logger or logging.getLogger("serial_manager")
        
        # 串口对象
        self.serial = None
        
        # 通信状态
        self.running = False
        self.connected = False
        
        # 发送队列
        self.send_queue = queue.Queue()
        
        # 发送线程
        self.send_thread = None
        
        # 重连线程
        self.reconnect_thread = None
        
        # 去重机制
        self.last_sent_data = None
        # 上次发送的数据（分类型）
        self.last_sent = {
            "qr": None,
            "circle": None,
            "material": None
        }
        
    def connect(self) -> bool:
        """
        连接串口
        
        Returns:
            是否连接成功
        """
        if self.connected:
            return True
            
        if self.port is None:
            self.logger.warning("未指定串口，无法连接")
            return False
            
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout
            )
            
            if self.serial.is_open:
                self.connected = True
                self.logger.info(f"成功连接到串口 {self.port}，波特率 {self.baudrate}")
                return True
            else:
                self.logger.error(f"无法打开串口 {self.port}")
                return False
                
        except Exception as e:
            self.logger.error(f"连接串口 {self.port} 时出错: {e}")
            self.connected = False
            return False
            
    def disconnect(self) -> None:
        """断开串口连接"""
        if self.serial and self.serial.is_open:
            try:
                self.serial.close()
                self.logger.info(f"串口 {self.port} 已断开")
            except Exception as e:
                self.logger.error(f"断开串口 {self.port} 时出错: {e}")
                
        self.connected = False
        
    def start(self) -> None:
        """启动串口管理器"""
        if self.running:
            return
            
        self.running = True
        
        # 启动发送线程
        self.send_thread = threading.Thread(target=self._send_task, daemon=True)
        self.send_thread.start()
        
        # 启动重连线程
        self.reconnect_thread = threading.Thread(target=self._reconnect_task, daemon=True)
        self.reconnect_thread.start()
        
        self.logger.info("串口管理器已启动")
        
    def stop(self) -> None:
        """停止串口管理器"""
        self.running = False
        
        # 等待线程结束
        if self.send_thread:
            self.send_thread.join(timeout=1.0)
            
        if self.reconnect_thread:
            self.reconnect_thread.join(timeout=1.0)
            
        # 断开连接
        self.disconnect()
        
        self.logger.info("串口管理器已停止")
        
    def send_qr_data(self, qr_content: str) -> bool:
        """
        发送二维码数据
        
        Args:
            qr_content: 二维码内容
            
        Returns:
            是否成功加入发送队列
        """
        # 去重机制，只有新数据才发送
        if qr_content == self.last_sent["qr"]:
            return False
            
        # 将数据加入发送队列
        data = {"type": "qr", "content": qr_content}
        self.send_queue.put(data)
        self.logger.debug(f"二维码数据已加入发送队列: {qr_content}")
        
        # 更新上次发送的数据
        self.last_sent["qr"] = qr_content
        
        return True
        
    def send_circle_position(self, position: List[float], radius: float, color: str) -> bool:
        """
        发送圆环位置和颜色数据
        
        Args:
            position: 圆环中心坐标 [x, y]
            radius: 圆环半径
            color: 圆环颜色
            
        Returns:
            是否成功加入发送队列
        """
        # 构建位置数据字符串
        data_str = f"C:{position[0]:.1f},{position[1]:.1f},{color}"
        
        # 去重机制
        if data_str == self.last_sent["circle"]:
            return False
            
        # 将数据加入发送队列
        data = {"type": "circle", "content": data_str}
        self.send_queue.put(data)
        self.logger.debug(f"圆环位置数据已加入发送队列: {data_str}")
        
        # 更新上次发送的数据
        self.last_sent["circle"] = data_str
        
        return True
        
    def send_material_position(self, position: List[float], color: str) -> bool:
        """
        发送物料位置数据
        
        Args:
            position: 物料中心坐标 [x, y]
            color: 物料颜色名称
            
        Returns:
            是否成功加入发送队列
        """
        # 构建位置数据字符串
        data_str = f"M:{position[0]:.1f},{position[1]:.1f},{color}"
        
        # 去重机制
        if data_str == self.last_sent["material"]:
            return False
            
        # 将数据加入发送队列
        data = {"type": "material", "content": data_str}
        self.send_queue.put(data)
        self.logger.debug(f"物料位置数据已加入发送队列: {data_str}")
        
        # 更新上次发送的数据
        self.last_sent["material"] = data_str
        
        return True
        
    def send_all_results(self, results: Dict[str, Any]) -> bool:
        """
        发送所有检测结果
        
        Args:
            results: 检测结果字典，包含qrcode、circle、material的信息
            
        Returns:
            是否成功发送
        """
        success = False
        
        # 发送二维码数据
        if results["qrcode"]["detected"]:
            if self.send_qr_data(results["qrcode"]["content"]):
                success = True
                
        # 发送圆环数据
        if results["circle"]["detected"]:
            if self.send_circle_position(
                results["circle"]["position"],
                results["circle"]["radius"],
                results["circle"]["color"] if "color" in results["circle"] else "unknown"
            ):
                success = True
                
        # 发送物料数据
        if results["material"]["detected"]:
            if self.send_material_position(
                results["material"]["center"],
                results["material"]["color"]
            ):
                success = True
                
        return success
        
    def _send_task(self) -> None:
        """发送任务线程"""
        while self.running:
            try:
                # 检查连接状态
                if not self.connected:
                    time.sleep(0.1)
                    continue
                    
                # 获取队列中的数据（非阻塞）
                try:
                    data = self.send_queue.get(block=False)
                except queue.Empty:
                    time.sleep(0.01)
                    continue
                
                # 获取数据内容
                content = data["content"]
                    
                # 编码并发送数据
                self._send_data_frame(content)
                
                # 标记任务完成
                self.send_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"发送任务出错: {e}")
                time.sleep(0.1)
                
    def _reconnect_task(self) -> None:
        """重连任务线程"""
        while self.running:
            try:
                # 如果未连接，尝试重连
                if not self.connected:
                    self.logger.info(f"尝试连接串口 {self.port}...")
                    self.connect()
                    
                # 等待重连间隔
                time.sleep(self.reconnect_interval)
                
            except Exception as e:
                self.logger.error(f"重连任务出错: {e}")
                time.sleep(0.1)
                
    def _send_data_frame(self, data: str) -> bool:
        """
        发送数据帧
        
        Args:
            data: 要发送的数据
            
        Returns:
            是否发送成功
        """
        try:
            # 转换为字节
            data_bytes = data.encode('utf-8')
            data_length = len(data_bytes)
            
            # 计算CRC8校验
            crc = self._calc_crc8(data_bytes)
            
            # 构建数据帧：0xAA [长度] [数据] [CRC] 0x55
            frame = bytearray([0xAA, data_length]) + data_bytes + bytearray([crc, 0x55])
            
            # 发送数据
            bytes_sent = self.serial.write(frame)
            
            self.logger.debug(f"已发送数据帧: {frame.hex()}, 长度: {bytes_sent}字节")
            return bytes_sent == len(frame)
            
        except Exception as e:
            self.logger.error(f"发送数据帧时出错: {e}")
            self.connected = False
            return False
            
    def _calc_crc8(self, data: bytes) -> int:
        """
        计算CRC8校验值
        
        Args:
            data: 要计算校验的数据
            
        Returns:
            CRC8校验值
        """
        crc = 0
        polynomial = 0x31  # CRC8多项式
        
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x80:
                    crc = (crc << 1) ^ polynomial
                else:
                    crc = crc << 1
                crc &= 0xFF
                
        return crc 