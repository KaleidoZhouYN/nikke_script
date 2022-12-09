from PyQt5.QtWidgets import (QWidget,  QApplication,
    QMessageBox,QPushButton, QLabel, QComboBox, QCheckBox,
    QSizePolicy)

import sys
import types
import win32gui
import os
import subprocess,signal
from PIL import Image
import numpy as np

from pynput.keyboard import Key, Controller

from utils.win import get_active_window, get_screenshot_by_hwnd,getChildhWnd

from config import config



class Nikke_Toolkit(QWidget):
    def __init__(self):
        super().__init__()
        self.keyboard = Controller()
        self.components = []
        self.setGeometry(300, 300, 400, 200)
        self.setWindowTitle('NIKKE_TOOLKIT')

        oldsize = self.size()
        self.oldsize = [oldsize.width(),oldsize.height()]
        self.initUI()


    def regist(self,com):
        # 适应字体大小
        com.adjustSize()

        # 记录下原来的rect大小并加入components组件中
        oldrect = com.geometry()
        com.oldrect = [oldrect.left(),oldrect.top(),oldrect.width(),oldrect.height()]
        self.components.append(com)

    def resizeEvent(self, a0) -> None:
        super().resizeEvent(a0)
        
        # 根据放缩的factor来对每个组件的rect进行调整
        newsize = self.size()
        newsize = [newsize.width(),newsize.height()]
        factor = [newsize[0]/self.oldsize[0], newsize[1]/self.oldsize[1]]
        for com in self.components:
            newrect = [com.oldrect[0]*factor[0], 
                        com.oldrect[1]*factor[1],
                        com.oldrect[2]*factor[0],
                        com.oldrect[3]*factor[1]]
            newrect = [int(c) for c in newrect]
            com.setGeometry(*newrect)
        

    def add_active_window(self):
        text = QLabel("请选择你的模拟器窗口,如果没有找到，\n请打开模拟器并重新点击下拉框",self)
        text.move(50,5)
        self.regist(text)
        self.active_window_combo = QComboBox(self)
        active_window_title = get_active_window()[0]
        for a in active_window_title:
            self.active_window_combo.addItem(a)
        def showPopup(self):
            # get activate windos
            self.clear()
            active_window_title = get_active_window()[0]
            for a in active_window_title:
                self.addItem(a)
            QComboBox.showPopup(self)

        self.active_window_combo.move(50,40)
        self.active_window_combo.showPopup = types.MethodType(showPopup,self.active_window_combo)
        self.active_window_combo.resize(300,20)
        self.regist(self.active_window_combo)

        def onActivated(text):
            self.select_title = text
            # 检测是否在print(text)
            self.simulator_hWnd = win32gui.FindWindow(0,text)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            if self.simulator_hWnd:
                msg.setText("已找到选择的模拟器")
                msg.exec()
            else:
                msg.setText("未找到选择的模拟器，请重试")
                msg.exec()
                return 
            
            self.select_simulator()

        self.active_window_combo.activated[str].connect(onActivated)

    def add_screenshot(self):
        btn1 = QPushButton("此处测试模拟器截图",self)
        btn1.move(50,70)
        self.regist(btn1)

        def screenshot_top():
            img = get_screenshot_by_hwnd(self.simulator_hWnd,0,1)
            PIL_image = Image.fromarray(np.uint8(img[:,:,::-1]))
            PIL_image.show()

        btn1.clicked.connect(screenshot_top)

        btn2 = QPushButton("此处测试控制窗口截图",self)
        btn2.move(200,70)
        self.regist(btn2)

        def screenshot_ctl():
            img = get_screenshot_by_hwnd(self.simulator_hWnd,0,1)
            top_rect = win32gui.GetWindowRect(self.simulator_hWnd)
            childhWnd = getChildhWnd(self.simulator_hWnd)
            ctl_rect = win32gui.GetWindowRect(childhWnd)
            img = img[ctl_rect[1]-top_rect[1]:ctl_rect[3]-top_rect[1], ctl_rect[0]-top_rect[0]:ctl_rect[2]-top_rect[0]]
            PIL_image = Image.fromarray(np.uint8(img[:,:,::-1]))
            PIL_image.show()   
        btn2.clicked.connect(screenshot_ctl)         

    def select_simulator(self):
        keywords = ['MuMu',r'雷电']
        is_match = 0
        for k in keywords:
            if k in self.select_title:
                self.simulator_name = k
                is_match = 1
                break
    
        if not is_match:
            self.simulator_name = 'other'

    def closeEvent(self, event):
        if self.battle_s_process:
            self.keyboard.press('c')
            self.keyboard.release('c')

    def add_battle_S(self):
        self.battle_s_process = None
        text = QLabel("拦截S自动瞄准,按f可以进入/解除防御状态",self)
        text.move(50,100)
        self.regist(text)

        btn1 = QPushButton("开始",self)
        btn1.move(50,120)
        self.regist(btn1)

        btn2 = QPushButton("结束",self)
        btn2.move(200,120)
        self.regist(btn2)

        # refer: https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true
        def start_battle_s():
            self.battle_s_process = subprocess.Popen(["simulator_test","--simulator_name",self.simulator_name,
             "--hWnd",str(self.simulator_hWnd)], 
                        stdin = subprocess.PIPE,
                        stdout=None, 
                       shell=True, 
                       close_fds=True,
                       creationflags=subprocess.CREATE_NEW_PROCESS_GROUP) 

        def end_battle_s():
            if self.battle_s_process:
                self.keyboard.press('c')
                self.keyboard.release('c')
            self.battle_s_process = None

        btn1.clicked.connect(start_battle_s)
        btn2.clicked.connect(end_battle_s)

        
        

        drill = QCheckBox('优先瞄准钻头(未实现）',self)
        drill.move(50,150)
        self.regist(drill)

        defend = QCheckBox('自动防御(未实现)',self)
        defend.move(200,150)
        self.regist(defend)



    def initUI(self):
        self.add_active_window()
        self.add_screenshot()
        self.add_battle_S()
        
        self.show()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Nikke_Toolkit()

    sys.exit(app.exec_())