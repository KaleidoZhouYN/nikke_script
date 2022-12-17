from utils.yolov5.yolov5_onnx import YOLOV5_ONNX
import numpy as np
import time

def init(self):
    self.hero_point = np.array([int(self.w)//2, int(self.h*0.92)])
    self.miss_alert_cnt = 0
    self.miss_aim_cnt = 0
    self.is_defend = 0
    self._start_aim = 0
    self._end_aim = 0
    self._defend = 0
    self.detector = YOLOV5_ONNX(self.config)

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
    #self.logger.info('弱点检测总耗时 : {}'.format(time.time()-start))

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

def start(self):
    self.setForeground()
    while True:
        if self._end_aim:
            # 结束瞄准
            self.logger.info("结束自动瞄准")
            self.left_up()
            self._offset = None
            self._start_aim = 0
            self._end_aim = 0
            return
        if self._defend:
            defend(self)
            self._defend = 0
            continue
        if self.is_defend:
            # 仍然在防御姿态
            time.sleep(0.2)
            continue
  
        turnoff_auto_aiming(self)
        if self._offset is None:
            fix_aim_offset(self)
        else:
            aim_alert(self)