"""
    strategy for auto weak point aiming
"""

import numpy as np
import win32gui
from utils import mouse
from utils.yolov5.yolov5_onnx import YOLOV5_ONNX
from utils.win import get_screenshot_by_hwnd,setForeground,getChildhWnd

import os
import cv2
import time
import keyboard

import logging


# match aim_box to mouse

class Simulator(object):
    def __init__(self, hWnd, config):
        self.logger = logging.getLogger("simulator")
        self.logger.setLevel(logging.INFO)
        os.makedirs("./logs",exist_ok=True)
        f_handler = logging.FileHandler(filename='./logs/simulutor.txt',
                                    mode = 'a',
                                    encoding='utf-8')
        self.logger.addHandler(f_handler)

        self.top_hWnd = hWnd
        self.top_rect = win32gui.GetWindowRect(self.top_hWnd)
        self.logger.info('模拟器显示窗口位置：{}'.format(self.top_rect))

        self.get_ctl_hWnd()

        self.config = config
        self.detector = YOLOV5_ONNX(config)

        rect = win32gui.GetWindowRect(self.ctl_hWnd)
        self._center = np.array([int(rect[2]-rect[0])//2, int(rect[3]-rect[1])//2])
        self._offset = None
        self.ctl_rect = rect
        self.logger.info("模拟器控制窗口位置：{}".format(self.ctl_rect))
        self.h = rect[3] - rect[1]
        self.w = rect[2] - rect[0]
        self.hero_point = np.array([int(self.w)//2, int(self.h*0.92)])
        
        self.miss_alert_cnt = 0
        self.miss_aim_cnt = 0

        self.is_defend = 0
        self.is_left_down = 0
  

    def get_ctl_hWnd(self):
        self.get_ctl_hWnd = getChildhWnd(self.top_hWnd)


    def screenshot(self):
        start=time.time()
        img = get_screenshot_by_hwnd(self.top_hWnd,0,1)
        cast = time.time() - start
        self.logger.info('模拟器截图 耗时:{:.4f}ms'.format(cast*1000))
        
        # 这里我们实际要得到控制窗口的截图
        img = img[self.ctl_rect[1]-self.top_rect[1]:self.ctl_rect[3]-self.top_rect[1], self.ctl_rect[0]-self.top_rect[0]:self.ctl_rect[2]-self.top_rect[0]]
        return img

    def move_cur_center(self):
        self.move_to(np.array(self._center),0)
        self.mouse_point = self._center

    def left_down(self):
        if not self.is_left_down:
            mouse.left_down(self.ctl_hWnd, self.mouse_point[0], self.mouse_point[1])
            self.is_left_down = 1

    def left_up(self):
        mouse.left_up(self.ctl_hWnd, self.mouse_point[0], self.mouse_point[1])
        self.is_left_down = 0

    def left_click(self,target):
        mouse.left_click(self.ctl_hWnd,target[0], target[1], 0.1)

    def move_to(self,target,lbutton=1):
        if lbutton:
            mouse.mouse_drag(self.ctl_hWnd,self.mouse_point[0],self.mouse_point[1],target[0],target[1],self.config.is_smooth,self.config.duration,self.config.smooth_k)
        else:
            mouse.move_to(self.ctl_hWnd,target[0],target[1],0)
        self.mouse_point = target

    def fix_aim_offset(self):
        #setForeground(self.top_hWnd)
        # first move the mouse to center of the simulator and then press
        self.move_cur_center()
        self.left_down()
        aim_box = None
        img = self.screenshot()
        det = self.detector.infer_aim(img)
        if det is not None and len(det):
            for *xyxy,conf,cls in det:
                if int(cls) == 1: # find aim_box
                    aim_box = xyxy
                    break
        if aim_box is None:
            self.miss_aim_cnt += 1
            if self.miss_aim_cnt > 5:
                self.left_up()
                self.miss_aim_cnt = 0
            return

        self.logger.info("找到准心位置：{}".format(aim_box))
        self._offset = np.array([int(aim_box[0]+aim_box[2])//2,int(aim_box[1]+aim_box[3])//2]) - self._center

    def aim_alert(self):
        start = time.time()
        aim_box_center = self.mouse_point + self._offset
        img = self.screenshot()
        det = self.detector.infer_alert(img)
        self.logger.info('弱点检测总耗时 : ',time.time()-start)

        # find nearest alert
        x , min_dis = None, 1e9
        if det is not None and len(det):
            for *xyxy, conf, cls in det:
                c = np.array([int(xyxy[0]+xyxy[2])//2,int(xyxy[1]+xyxy[3])//2])
                if (int(cls)) == 0:
                    dis = np.linalg.norm(c - aim_box_center)
                    if dis < min_dis:
                        min_dis = dis
                        x = c

        
        if x is None:
            self.miss_alert_cnt += 1
            if self.miss_alert_cnt > 3:
                offset = self._center - aim_box_center
                offset[1] -= int(self.h * 0.05)
                target_mouse_point = self.mouse_point + offset
                self.move_to(target_mouse_point)
            
            return 0
        else:
            self.miss_alert_cnt = 0
            offset = (x - aim_box_center)
            target_mouse_point = self.mouse_point + offset
            self.move_to(target_mouse_point)

    def exit_battle(self):
        """
            not implement yet
        """
        pass

    def turnoff_auto_aiming(self):
        """
            not implement yet
        """
        pass

    def defend(self):
        if self.is_defend == 0:
            # 进入防御姿态
            self.left_up()
            self.left_click(self.hero_point)
            self.is_defend = 1
            self.logger.info('\n进入防御姿态..............')
        else:
            self.left_click(self.hero_point)
            self.left_down()
            self.is_defend = 0
            self.logger.info("\n解除防御姿态")

    def start_simulation(self):
        def key_press(key):
            if key.name == 'f':
                key_press.defend = 1
            if key.name == 'e':
                key_press.quit = 1
            if key.name == 's':
                key_press.start = 1
        key_press.defend = 0
        key_press.quit = 0
        key_press.start = 0

        keyboard.on_press(key_press)

        while True:
            if key_press.quit:
                # 结束瞄准
                self.logger.info("结束自动瞄准")
                self.left_up()
                self._offset = None
                return
            if key_press.defend:
                self.defend()
                key_press.defend = 0
            if self.is_defend:
                # 仍然在防御姿态
                time.sleep(0.2)
                continue
            if key_press.start:
                self.turnoff_auto_aiming()
                if self._offset is None:
                    self.fix_aim_offset()
                else:
                    self.aim_alert()