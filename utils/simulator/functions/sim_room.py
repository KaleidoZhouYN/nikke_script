
import time
import numpy as np
import cv2
from PIL import Image

def left_click(ins,rect):
    ins.left_click([rect[0]+rect[2]//2,rect[1]+rect[3]//2])

def init(self):
    self._start_sim = 0
    self._end_sim = 0

    self.quick_battle_rect = [1008,917,232,45]
    self.start_battle_rect = [1008,974,232,77]

    self.confirm_v3_rect = [978,900,218,53]  # 选择增益
    self.confirm_v1_rect = [978,987,174,48] # 未知
    self.confirm_v1_rect_1 = [975,651,220,50]
    self.confirm_v1_rect_2 = [851,622,220,50]

    self.confirm_v2_rect = [871,985,179,53] # 宝箱只能选择一个
    self.confirm_v2_rect_1 = [850,704,220,50]
    self.confirm_v2_rect_2 = [850,768,220,50]
    

    self.cancel_v1_rect = [765,987,174,48]
    self.cancel_v3_rect = [722,900,220,50]
    self.cancel_v3_rect_1 = [975,651,220,50]

    # 战前选择
    self.h3_rect = [[720,722,134,12],
                        [895,722,134,12],
                        [1064,722,134,12]]

    self.h3_rect_1 = [[714,618,28,20],
                     [890,618,28,20],
                     [1060,618,28,20]]

    self.h3_rect_2 = [[725,741,125,48],
                        [902,739,125,48],
                        [1070,741,125,48]]

    self.h2_rect = [[800,720,134,12],
                    [977,720,134,12]]

    self.h1_rect = [895,722,134,12]

    # 战后选择    
    self.v3_rect = [[860,457,333,12],
                    [860,629,333,12],
                    [860,800,333,12]]
    
    self.v3_rect_1 = [840,390,12,440] # 判断用的竖线

    self.v2_rect = [727,688,470,92]

    # C区域结束
    self.end_rect = [845,748,233,50]
    self.end_rect_1 = [850,650,221,50]
    self.end_rect_2 = [730,825,463,72]
    self.end_rect_3 = [977,935,220,50]
    self.end_rect_4 = [975,651,220,50]

    # 下一个区域
    self.next_stage_rect = [970,750,220,50]


    rect_list = ['quick_battle_rect','start_battle_rect',
    'h3_rect','h3_rect_1','h3_rect_2','h2_rect','h1_rect','v3_rect','v3_rect_1',
    'confirm_v3_rect','confirm_v1_rect','confirm_v1_rect_1','confirm_v1_rect_2',
    'confirm_v2_rect','confirm_v2_rect_1','confirm_v2_rect_2','v2_rect',
    'cancel_v1_rect','cancel_v3_rect','cancel_v3_rect_1',
    'end_rect','end_rect_1','end_rect_2','end_rect_3','end_rect_4',
    'next_stage_rect']

    for i in range(len(rect_list)):
        rect = getattr(self,rect_list[i])
        rect = np.array(rect).astype(np.float32) / np.array([1920,1080,1920,1080])
        rect = (rect * np.array([self.w,self.h,self.w,self.h])).astype(np.int32)
        self.logger.info(rect)
        setattr(self,rect_list[i],rect)
        

    self.enforce_num = 0

def is_quick_battle(self):
    rect = self.quick_battle_rect
    img = self._screenshot[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    per = np.logical_and(np.logical_and(img[:,:,0] > 0,img[:,:,0]<25),img[:,:,1] > 100).sum()
    per = float(per) / (img.shape[0]*img.shape[1])
    # 结束战斗时同样会出现红色
    self.logger.info("快速战斗占比:{}".format(per))
    if per > 0.7:
        if is_start_battle(self):
            return True
        else:
            return False
    else:
        return False

def quick_battle(self):
    self.logger.info("快速战斗")
    rect = self.quick_battle_rect
    left_click(self,rect)

def is_start_battle(self):
    rect = self.start_battle_rect
    img = self._screenshot[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    per = np.logical_and(np.logical_and(img[:,:,0] > 95,img[:,:,0]<135),img[:,:,1] > 100).sum()
    per = float(per) / (img.shape[0]*img.shape[1])
    self.logger.info("开始战斗占比:{}".format(per))
    if per > 0.7:
        return True
    else:
        return False

def start_battle(self):
    self.logger.info("开始战斗")
    rect = self.start_battle_rect
    left_click(self,rect)    


def is_select_h3(self):
    flag = 0
    self.logger.info("判断是否是战前状态")
    for i in range(3):
        rect = self.h3_rect[i]
        img = self._screenshot[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
        img = cv2.Canny(img,100,200)
        if ((img>0).sum(axis=1).astype(np.float32)/img.shape[1] > 0.7).sum() > 0:
            flag = True
        else:
            flag = False
            break
    return flag

def select_h3(self):
    self.logger.info("战前选择")
    min_red = 1e6
    h3_min = 0
    for i in range(3):
        rect = self.h3_rect_2[i]
        img = self._screenshot[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        if (img[:,:,2] > 200).sum() < 10:
            continue
        rect = self.h3_rect_1[i]
        img = self._screenshot[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        per = np.logical_and(np.logical_and(img[:,:,0] > 0,img[:,:,0]<35),img[:,:,1] > 100).sum()
        if min_red > per:
            min_red = per
            h3_min = i
    self.logger.info("选择第{}个".format(h3_min))
    rect = self.h3_rect[h3_min]
    left_click(self,rect)
    # 卡顿，多点击一次
    time.sleep(2)
    self.screenshot()
    if not is_start_battle(self):
        left_click(self,rect)
        

def is_select_h2(self):
    flag = 0
    for i in range(2):
        rect = self.h2_rect[i]
        img = self._screenshot[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
        img = cv2.Canny(img,100,200)
        if ((img>0).sum(axis=1).astype(np.float32)/img.shape[1] > 0.7).sum() > 0:
            flag = True
        else:
            flag = False
            break
    return flag 

def select_h2(self):
    self.logger.info("关底选择")
    rect = self.h2_rect[0]
    left_click(self,rect)     

def is_select_h1(self):
    rect = self.h1_rect
    img = self._screenshot[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
    img = cv2.Canny(img,100,200)
    return ((img>0).sum(axis=1).astype(np.float32)/img.shape[1] > 0.7).sum() > 0

def select_h1(self):
    self.logger.info("关底战斗")
    rect = self.h1_rect
    left_click(self,rect)     

def is_select_v3(self):
    # 方框的大小不固定
    """
    flag = 0
    for i in range(3):
        rect = self.v3_rect[i]
        img = self._screenshot[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
        img = cv2.Canny(img,100,200)
        if ((img>0).sum(axis=1).astype(np.float32)/img.shape[1] > 0.7).sum() > 0:
            flag = True
        else:
            flag = False
            break
    return flag 
    """
    rect = self.v3_rect_1
    img = self._screenshot[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
    img = cv2.Canny(img,100,200)
    return ((img>0).sum(axis=0).astype(np.float32)/img.shape[0] > 0.4).sum() > 0    

def select_v3(self):
    self.logger.info("战后选择")
    min_h = 1e6
    s = 0
    for i in range(3):
        rect = self.v3_rect[i]
        img = self._screenshot[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        mask = img[:,:,1] > 10
        h = min(img[mask,0].mean(),180-img[mask,0].mean())
        if min_h > h:
            min_h = h
            s = i
    self.logger.info("选择第{}个".format(s))
    rect = self.v3_rect[s]
    left_click(self,rect)
    time.sleep(1)
    if self.enforce_num < 8:  # 增益效果还没满，选择一个增益效果
        rect = self.confirm_v3_rect     
        left_click(self,rect)
        self.enforce_num += 1
        self.logger.info("目前的增益数量:{}".format(self.enforce_num))
    else:  # 增益效果满了
        rect = self.cancel_v3_rect
        left_click(self,rect)
        time.sleep(2)
        rect = self.cancel_v3_rect_1
        left_click(self,rect)
        self.logger.info("增益效果已满，退出选择")

def is_select_v1(self):
    rect = self.confirm_v1_rect
    img = self._screenshot[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    per = np.logical_and(np.logical_and(img[:,:,0] > 95,img[:,:,0]<135),img[:,:,1] > 100).sum()
    per = float(per) / (img.shape[0]*img.shape[1])
    if per > 0.7:
        return True
    else:
        return False   

def select_v1(self):
    self.logger.info("不选择任何未知选项")
    rect = self.cancel_v1_rect
    left_click(self,rect)
    time.sleep(2)
    rect = self.confirm_v1_rect_1
    left_click(self,rect)
    time.sleep(2)
    rect = self.confirm_v1_rect_2 
    left_click(self,rect)

def is_select_v2(self):
    rect = self.confirm_v2_rect
    img = self._screenshot[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    per = np.logical_and(np.logical_and(img[:,:,0] > 95,img[:,:,0]<135),img[:,:,1] > 100).sum()
    per = float(per) / (img.shape[0]*img.shape[1])
    if per > 0.7:
        return True
    else:
        return False  

def select_v2(self):
    self.logger.info("必须选择一个未知选项")
    rect = self.v2_rect
    left_click(self,rect)
    time.sleep(1)
    rect = self.confirm_v2_rect
    left_click(self,rect)
    time.sleep(2)
    rect = self.confirm_v2_rect_1
    left_click(self,rect)
    time.sleep(2)
    rect = self.confirm_v2_rect_2
    left_click(self,rect)


def is_next_stage(self):
    rect = self.next_stage_rect
    img = self._screenshot[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    per = np.logical_and(np.logical_and(img[:,:,0] > 95,img[:,:,0]<135),img[:,:,1] > 100).sum()
    per = float(per) / (img.shape[0]*img.shape[1])
    self.logger.info("下一阶段占比:{}".format(per))
    if per > 0.7:
        return True
    else:
        return False  

def next_stage(self):
    rect = self.next_stage_rect
    left_click(self,rect)
    self.logger.info("进入下一个阶段")

def is_end_sim(self):
    rect = self.end_rect
    img = self._screenshot[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    per = np.logical_and(img[:,:,1] < 10,img[:,:,2] > 240).sum()
    per = float(per) / (img.shape[0]*img.shape[1])
    if per > 0.7:
        rect[0:2] += rect[2:4]
        img = self._screenshot[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        per = np.logical_and(img[:,:,1] < 10,img[:,:,2] < 20).sum()
        per = float(per) / (img.shape[0]*img.shape[1])
        self.logger.info("结束模拟室占比:{}".format(per))
        if per > 0.7:
            return True
        else:
            return False
    else:
        return False     

def end_sim(self):
    self.logger.info("模拟室结束")
    rect = self.end_rect
    left_click(self,rect)
    time.sleep(1)
    rect = self.end_rect_1
    left_click(self,rect)
    time.sleep(1)
    rect = self.end_rect_2
    left_click(self,rect)
    time.sleep(1)        
    rect = self.end_rect_3
    left_click(self,rect)
    time.sleep(1)
    rect = self.end_rect_4
    left_click(self,rect)

def start(self):
    while 1:
        if self._end_sim:
            self._end_sim = 0
            self._start_sim = 0
            self.logger.info("模拟室结束")
            return

        time.sleep(2)
        self.screenshot()

        if is_end_sim(self):
            end_sim(self)
            self._end_sim = 0
            self._start_sim = 0
            return
        if is_next_stage(self):
            next_stage(self)
            continue
        if is_quick_battle(self):
            quick_battle(self)
            continue
        if is_start_battle(self):
            start_battle(self)
            continue
        if is_select_h3(self):
            select_h3(self)
            continue
        if is_select_h2(self):
            select_h2(self)
            continue
        if is_select_h1(self):
            select_h1(self)
            continue
        if is_select_v3(self):
            select_v3(self)
            continue
        if is_select_v2(self):
            select_v2(self)
            continue
        if is_select_v1(self):
            select_v1(self)
            continue

        self.logger.info("在战斗中或者战斗结算")
        self.left_click(self._center)
        time.sleep(5)



