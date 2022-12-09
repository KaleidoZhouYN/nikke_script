import win32gui
import numpy as np

def get_active_window():
    def callback(hWnd,lParam):
        is_visible = win32gui.GetWindowLong(hWnd,win32con.GWL_STYLE) & win32con.WS_VISIBLE
        if not is_visible:
            return True
        
        length = win32gui.GetWindowTextLength(hWnd)
        if (length == 0):
            return True
        windowTitle = win32gui.GetWindowText(hWnd)
        callback._titleList.append(windowTitle)
        callback._hWndList.append(hWnd)

        return True
    callback._titleList = []
    callback._hWndList = []
    win32gui.EnumWindows(callback,None)
    return callback._titleList,callback._hWndList

def get_hWnd_by_name(name):
    def callback(hWnd,lParam):
        
        is_visible = win32gui.GetWindowLong(hWnd,win32con.GWL_STYLE) & win32con.WS_VISIBLE
        if not is_visible:
            return True
        
        length = win32gui.GetWindowTextLength(hWnd)
        if (length == 0):
            return True
        windowTitle = win32gui.GetWindowText(hWnd)
        callback._titleList.append(windowTitle)

        return True
    callback._titleList = []
    win32gui.EnumWindows(callback,None)
    titleList = callback._titleList
    
    hWnd = None
    for t in titleList:
        if name in t:
            hWnd = win32gui.FindWindow(0,t)
            print(t)
    if hWnd is None:
        win32gui.MessageBox(0,"your app with keyword '{}' not found".format(name),'Error',0)
        return 0

    return hWnd  

import win32gui
import win32api
import win32com.client
import win32con
import win32ui

def setForeground(hWnd):
    if hWnd != win32gui.GetForegroundWindow():
        # why call SendMessage: https://blog.csdn.net/weixin_30299539/article/details/96321161
        shell = win32com.client.Dispatch("WScript.Shell")
        shell.SendKeys('%')
        #win32gui.SendMessage(hWnd, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0)
        win32gui.SetForegroundWindow(hWnd)
    return 1

def getChildhWnd(hWnd):
    def callback(hWnd,lParam):
        length = win32gui.GetWindowTextLength(hWnd)
        if (length == 0):
            return True
        #windowTitle = win32gui.GetWindowText(hWnd)
        callback._hWndList.append(hWnd)

        return True
    callback._hWndList = []
    win32gui.EnumChildWindows(hWnd,callback,None)
    if len(callback._hWndList):
        return callback._hWndList[0]
    else:
        return hWnd

from PIL import ImageGrab
from ctypes import windll
from PIL import Image

def get_screenshot_by_hwnd(hWnd,call_front=False,is_backgroud=False):
    if not is_backgroud:
        if call_front and setForeground(hWnd):
            rect = win32gui.GetWindowRect(hWnd)
            img = ImageGrab.grab(rect) # [RGB]
            img = np.asarray(img)[:,:,::-1]  # [BGR]
            return img
        else:
            return None
    else:
        # to do
        # inference: https://stackoverflow.com/questions/19695214/screenshot-of-inactive-window-printwindow-win32gui
        # what is DC : https://blog.csdn.net/tc1175307496/article/details/52708832
        
        rect = win32gui.GetWindowRect(hWnd)
        #print('window rect is:' , rect)
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]

        hWndDC = win32gui.GetWindowDC(hWnd)
        mfcDC = win32ui.CreateDCFromHandle(hWndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

        saveDC.SelectObject(saveBitMap)

        result = windll.user32.PrintWindow(hWnd, saveDC.GetSafeHdc(), 1)
        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)

        im = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1
        )

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hWnd,hWndDC)
        
        im = np.asarray(im)[:,:,::-1]
        return im


