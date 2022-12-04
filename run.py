from PyQt5.QtWidgets import (QWidget, QLabel, 
    QComboBox, QApplication,
    QHBoxLayout,QVBoxLayout,
    QMessageBox,QPushButton,
    QCheckBox)
import sys
import types
import win32gui
import os
import subprocess,signal

from pynput.keyboard import Key, Controller

from utils.win import get_active_window
# form utils.simulator import (Simulator, MuMuX, Leidian)

from config import config



class Nikke_Toolkit(QWidget):
    def __init__(self):
        super().__init__()
        self.keyboard = Controller()
        self.initUI()

    def add_active_window(self):
        text = QLabel("请选择你的模拟器窗口,如果没有找到，\n请打开模拟器并重新点击下拉框",self)
        text.move(50,5)
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
        text.move(50,70)
        btn1 = QPushButton("开始",self)
        btn2 = QPushButton("结束",self)


        # refer: https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true
        def start_battle_s():
            self.battle_s_process = subprocess.Popen(["python","simulator_test.py","--simulator_name",self.simulator_name,
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

        btn1.move(50,90)
        btn2.move(200,90)

        drill = QCheckBox('优先瞄准钻头(未实现）',self)
        drill.move(50,120)

        defend = QCheckBox('自动防御(未实现)',self)
        defend.move(200,120)



    def initUI(self):
        self.add_active_window()
        self.add_battle_S()

        self.setGeometry(300, 300, 400, 200)
        self.setWindowTitle('NIKKE_TOOLKIT')
        self.show()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Nikke_Toolkit()

    sys.exit(app.exec_())