# for PyQt5 simulator,please refer :https://blog.csdn.net/robot_no1/article/details/120795278
# to Post message
import win32gui
import win32con

def left_down(hWnd, x, y):
    x,y = int(x), int(y)
    wparam = win32con.MK_LBUTTON
    lparam = y << 16 | x
    win32gui.PostMessage(hWnd, win32con.WM_LBUTTONDOWN, wparam, lparam)

def move_to(hWnd, x, y, is_ldown=False):
    x,y = int(x), int(y)
    wparam = win32con.MK_LBUTTON if is_ldown else 0
    lparam = y << 16 | x
    win32gui.PostMessage(hWnd, win32con.WM_MOUSEMOVE, wparam, lparam)    

def left_up(hWnd, x, y):
    x,y = int(x), int(y)
    wparam = 0
    lparam = y << 16 | x
    win32gui.PostMessage(hWnd, win32con.WM_LBUTTONUP, wparam, lparam)       

def __set_cursor(hWnd, msg):
    lparam = (msg << 16 ) | 1
    win32gui.PostMessage(hWnd,win32con.WM_SETCURSOR,hWnd,lparam)

def __activate_mouse(hWnd):
    lparam = (win32con.WM_LBUTTONDOWN << 16) | win32con.HTCLIENT
    win32gui.PostMessage(hWnd, win32con.WM_MOUSEACTIVATE, hWnd, lparam)

def mouse_click(hWnd, x, y, time=0.5):
    left_down(hWnd, x, y)
    move_to(hWnd, x, y)
    left_up(hWnd,x, y)

import time
def mouse_drag(hWnd, x1, y1, x2, y2, is_smooth=False,duration=0.1,smooth_k=10):
    if is_smooth:
        k1 = (x2-x1) / 10
        k2 = (y2-y1) / 10
        for i in range(10):
            move_to(hWnd, int(x1+k1*i), int(y1+k2*i), 1)
            time.sleep(duration/smooth_k)
    else:
        move_to(hWnd, x2, y2, 1)
