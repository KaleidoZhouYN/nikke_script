"""
    strategy for auto weak point aiming
"""

import numpy as np
import win32gui
from utils import mouse
from utils.win import get_screenshot_by_hwnd,setForeground,getChildhWnd
import utils.simulator.functions.battle_s as battle_s
import utils.simulator.functions.sim_room as sim_room

import os
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
                                    mode = 'w',
                                    encoding='utf-8')
        self.logger.addHandler(f_handler)


        #ch = logging.StreamHandler()
        #ch.setLevel(logging.INFO)
        #self.logger.addHandler(ch)
        

        self.top_hWnd = hWnd
        self.top_rect = win32gui.GetWindowRect(self.top_hWnd)
        self.logger.info('模拟器显示窗口位置：{}'.format(self.top_rect))

        self.get_ctl_hWnd()

        rect = win32gui.GetWindowRect(self.ctl_hWnd)
        self._center = np.array([int(rect[2]-rect[0])//2, int(rect[3]-rect[1])//2])
        self._offset = None
        self.ctl_rect = rect
        self.logger.info("模拟器控制窗口位置：{}".format(self.ctl_rect))
        self.h = rect[3] - rect[1]
        self.w = rect[2] - rect[0]

        self.config = config

        self.is_left_down = 0

        # functions
        self.battle_s = battle_s
        self.battle_s.init(self)

        self.sim_room = sim_room
        self.sim_room.init(self)

  

    def get_ctl_hWnd(self):
        self.ctl_hWnd = getChildhWnd(self.top_hWnd)


    def screenshot(self):
        start=time.time()
        img = get_screenshot_by_hwnd(self.top_hWnd,0,1)
        cast = time.time() - start
        #self.logger.info('模拟器截图 耗时:{:.4f}ms'.format(cast*1000))
        
        # 这里我们实际要得到控制窗口的截图
        img = img[self.ctl_rect[1]-self.top_rect[1]:self.ctl_rect[3]-self.top_rect[1], self.ctl_rect[0]-self.top_rect[0]:self.ctl_rect[2]-self.top_rect[0]]
        self._screenshot = img.copy()
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

    def setForeground(self):
        setForeground(self.top_hWnd)


    def start(self):
        def key_press(key):
            if key.name == 'f':
                self._defend = 1
            if key.name == 'o':
                self._start_aim = 1
            if key.name == 'p':
                self._end_aim = 1
            if key.name == '[':
                self._start_sim = 1
            if key.name == ']':
                self._end_sim = 1

        keyboard.on_press(key_press)
        while 1:
            if self._start_aim:
                self.battle_s.start(self)
            if self._start_sim:
                self.sim_room.start(self)

        """
        th_battle_s = threading.Thread(target=self.battle_s.start,args=(self))
        th_battle_s.daemon = True
        th_battle_s.start()
        """
        """
        self.logger.info("启动模拟室线程")
        th_sim_room = threading.Thread(target=self.sim_room.start,args=(self))
        th_sim_room.daemon = True
        th_sim_room.start()
        """