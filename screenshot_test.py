"""
test the python code the get the screenshot of given window title
"""

from utils.win import get_hWnd_by_name , get_screenshot_by_hwnd
import win32con,win32gui
from PIL import Image
import numpy as np


if __name__ == '__main__':
    keywords = r'MuMu模拟器'
    hWnd = get_hWnd_by_name(keywords)
    def callback(hWnd,lParam):
        length = win32gui.GetWindowTextLength(hWnd)
        if (length == 0):
            return True
        windowTitle = win32gui.GetWindowText(hWnd)
        callback._hWndList.append(hWnd)

        return True
    callback._hWndList = []
    win32gui.EnumChildWindows(hWnd,callback,None)
    ctl_hWnd = callback._hWndList[0]
    numpy_image = get_screenshot_by_hwnd(ctl_hWnd,0,1) # []
    PIL_image = Image.fromarray(np.uint8(numpy_image[:,:,::-1]))
    PIL_image.show()
    
