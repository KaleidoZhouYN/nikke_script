from utils.simulator.base import Simulator
import win32gui

class MuMuX(Simulator):
    def __init__(self, hWnd, config):
        super().__init__(hWnd, config)

    def get_ctl_hWnd(self):
        def callback(hWnd,lParam):
            length = win32gui.GetWindowTextLength(hWnd)
            if (length == 0):
                return True
            #windowTitle = win32gui.GetWindowText(hWnd)
            callback._hWndList.append(hWnd)

            return True
        callback._hWndList = []
        win32gui.EnumChildWindows(self.top_hWnd,callback,None)
        self.ctl_hWnd = callback._hWndList[0]


