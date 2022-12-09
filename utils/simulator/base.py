"""
    strategy for auto weak point aiming
"""

import numpy as np
import win32gui
from utils import mouse
from utils.yolov5.yolov5_onnx import YOLOV5_ONNX
from utils.win import get_screenshot_by_hwnd,setForeground,getChildhWnd

import cv2
import time
import keyboard

# match aim_box to mouse

class Simulator(object):
    def __init__(self, hWnd, config):
        self.top_hWnd = hWnd
        self.top_rect = win32gui.GetWindowRect(self.top_hWnd)

        self.get_ctl_hWnd()

        self.config = config
        self.detector = YOLOV5_ONNX(config)

        rect = win32gui.GetWindowRect(self.ctl_hWnd)
        self.center = np.array([int(rect[2]-rect[0])//2, int(rect[3]-rect[1])//2])
        self.ctl_rect = rect
        self.h = rect[3] - rect[1]
        self.w = rect[2] - rect[0]
        self.hero_point = np.array([int(self.w)//2, int(self.h*0.92)])
        

        self.miss_alert_cnt = 0
        self.is_defend = 0
        

    def get_ctl_hWnd(self):
        return getChildhWnd(self.top_hWnd)


    def screenshot(self):
        start=time.time()
        img = get_screenshot_by_hwnd(self.top_hWnd,0,1)
        cast = time.time() - start
        #print('screenshot 耗时:{}'.format(cast))
        
        # 这里我们实际要得到控制窗口的截图
        img = img[self.ctl_rect[1]-self.top_rect[1]:self.ctl_rect[3]-self.top_rect[1], self.ctl_rect[0]-self.top_rect[0]:self.ctl_rect[2]-self.top_rect[0]]
        return img

    def move_cur_center(self):
        self.move_to(np.array(self.center),0)
        self.mouse_point = self.center

    def left_down(self):
        mouse.left_down(self.ctl_hWnd, self.mouse_point[0], self.mouse_point[1])

    def left_up(self):
        mouse.left_up(self.ctl_hWnd, self.mouse_point[0], self.mouse_point[1])

    def left_click(self,target):
        mouse.left_click(self.ctl_hWnd,target[0], target[1], 0.1)

    def move_to(self,target,lbutton=1):
        if lbutton:
            mouse.mouse_drag(self.ctl_hWnd,self.mouse_point[0],self.mouse_point[1],target[0],target[1],self.config.is_smooth,self.config.duration,self.config.smooth_k)
        else:
            mouse.move_to(self.ctl_hWnd,target[0],target[1],0)
        self.mouse_point = target

    def fix_aim_offset(self):
        setForeground(self.top_hWnd)
        # first move the mouse to center of the simulator and then press
        self.move_cur_center()
        self.left_down()
        aim_box = None
        miss_cnt = 0
        while True:
            img = self.screenshot()
            #print('screenshot img_size:', img.shape)
            det = self.detector.infer_aim(img)
            if det is not None and len(det):
                for *xyxy,conf,cls in det:
                    if int(cls) == 1: # find aim_box
                        aim_box = xyxy
                        break
            if not aim_box is None:
                break
            miss_cnt += 1
            if miss_cnt > 5:
                self.left_up()
                self.left_down()
                miss_cnt = 0
        self._offset = np.array([int(aim_box[0]+aim_box[2])//2,int(aim_box[1]+aim_box[3])//2]) - self.center

    def aim_alert(self):
        start = time.time()
        aim_box_center = self.mouse_point + self._offset
        img = self.screenshot()
        det = self.detector.infer_alert(img)
        #print('检测弱点总耗时 : ',time.time()-start)

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
                #print('未检测到弱点，准心回到屏幕中心上方一点位置')
                # alert not detected,move aim back to center
                offset = self.center - aim_box_center
                offset[1] -= int(self.h * 0.05)
                target_mouse_point = self.mouse_point + offset
                self.move_to(target_mouse_point)
            
            return 0
        else:
            #print('检测到弱点，位置 : ', x)
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
            #print('\n进入防御姿态..............')
        else:
            self.left_click(self.hero_point)
            self.left_down()
            self.is_defend = 0
            #print("\n解除防御姿态")

    def start_simulation(self):
        # first we assume the auto aiming program is off
        self.turnoff_auto_aiming()
        #print('开始寻找准心...')
        self.fix_aim_offset()
        #print("准心调整成功,开始启动自动瞄准，请勿点击游戏")
        def key_press(key):
            if key.name == 'f':
                key_press.flag = 1
            if key.name == 'c':
                key_press.quit = 1
        key_press.flag = 0
        key_press.quit = 0
        keyboard.on_press(key_press)
        while True:
            if key_press.quit:
                # 结束瞄准
                #print("结束自动瞄准")
                self.left_up()
                return 
            if key_press.flag:
                self.defend()
                key_press.flag = 0
            if self.is_defend:
                # 仍然在防御姿态
                time.sleep(0.2)
                continue
            self.aim_alert()