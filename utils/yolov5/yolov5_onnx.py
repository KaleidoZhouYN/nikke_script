import cv2
import numpy as np
#import onnx
import onnxruntime as rt
#import torch
import time
import random

import os
import shutil
import pickle

# For multiprocessing , refer: https://zhuanlan.zhihu.com/p/410931545
# pickle 大概会占70ms

"""
class WrapInferenceSession:

    def __init__(self, onnx_path):
        onnx_bytes = onnx.load_model(onnx_path)
        self.sess = rt.InferenceSession(onnx_bytes.SerializeToString())
        self.onnx_bytes = onnx_bytes

    def run(self, *args, **kwargs):
        return self.sess.run(*args,**kwargs)

    def __getstate__(self):
        return {'onnx_bytes': self.onnx_bytes}

    def __setstate__(self, values):
        self.onnx_bytes = values['onnx_bytes']
        self.sess = rt.InferenceSession(self.onnx_bytes.SerializeToString(),providers=['CPUExecutionProvider'])
"""

class YOLOV5_ONNX(object):
    def __init__(self,config):
        # onnxruntime Session Options

        sess_options = rt.SessionOptions()
        sess_options.intra_op_num_threads = config.onnx_num_thread
        sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL


        self.aim_onnx_session=rt.InferenceSession(config.aim_onnx_path,providers=['CPUExecutionProvider'])
        #self.aim_onnx_session=pickle.dumps(WrapInferenceSession(aim_onnx_path))
        self.aim_input_name=['images']
        self.aim_output_name=['output0']

        self.alert_onnx_session=rt.InferenceSession(config.alert_onnx_path,providers=['CPUExecutionProvider'])
        #self.alert_onnx_session=pickle.dumps(WrapInferenceSession(alert_onnx_path))
        self.alert_input_name=['images']
        self.alert_output_name=['output0']


        self.aim_size = 640
        self.alert_size = 384

        self.w_croprate = 0.5
        self.h_croprate = 0.6

        self.is_record = False
        self.record_log = './detect_log'
        if os.path.exists(self.record_log):
            shutil.rmtree(self.record_log)
        os.mkdir(self.record_log)
        self.frame_cnt = 0



    def get_input_name(self,session):
        input_name=[]
        for node in session.get_inputs():
            input_name.append(node.name)

        return input_name


    def get_output_name(self,session):
        output_name=[]
        for node in session.get_outputs():
            output_name.append(node.name)

        return output_name

    def get_input_feed(self,input_name,image_tensor):
        input_feed={}
        for name in input_name:
            input_feed[name]=image_tensor

        return input_feed

    def letterbox(self,img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True,
                  stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def xywh2xyxy(self,x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)

        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

        return y


    def torchvision_nms(self, bboxes, scores, threshold=0.5):
        x1 = bboxes[:,0]
        y1 = bboxes[:,1]
        x2 = bboxes[:,2]
        y2 = bboxes[:,3]
        areas = (x2-x1)*(y2-y1)   # [N,] 每个bbox的面积
        #_, order = scores.sort(0, descending=True)    # 降序排列
        order = (-scores).argsort(0)

        keep = []
        while order.size > 0:       
            if order.size == 1:     
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()    
                keep.append(i)

            #  IOU calc
            """
            xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])
            inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]
            """

            xx1 = x1[order[1:]].clip(min=x1[i])
            yy1 = y1[order[1:]].clip(min=y1[i])
            xx2 = x2[order[1:]].clip(max=x2[i])
            yy2 = y2[order[1:]].clip(max=y2[i])
            inter = (xx2-xx1).clip(min=0) * (yy2-yy1).clip(min=0)


            iou = inter / (areas[i]+areas[order[1:]]-inter)  # [N-1,]
            idx = (iou <= threshold).nonzero()[0] # 注意此时idx为[N-1,] 而order为[N,]
            if idx.size == 0:
                break
            order = order[idx+1]  # 修补索引之间的差值
        #return torch.LongTensor(keep)   # Pytorch的索引值为LongTensor
        return np.array(keep)

    def nms(self,prediction, conf_thres=0.1, iou_thres=0.6, agnostic=False):
        """if prediction.dtype is torch.float16:
            prediction = prediction.float()  # to FP3
        """
        prediction = prediction.astype(np.float32)
        xc = prediction[..., 4] > conf_thres  # candidates
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_det = 300  # maximum number of detections per image
        output = [None] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence
            if not x.shape[0]:
                continue

            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            box = self.xywh2xyxy(x[:, :4])

            #conf, j = x[:, 5:].max(1, keepdim=True)
            conf, j = np.max(x[:,5:],1,keepdims=True), np.argmax(x[:,5:],1,keepdims=True)
            #x = torch.cat((torch.tensor(box), conf, j.float()), 1)[conf.view(-1) > conf_thres]
            x  = np.concatenate((box,conf,j.astype(np.float32)),1)
            n = x.shape[0]  # number of boxes
            if not n:
                continue
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = self.torchvision_nms(boxes, scores, iou_thres)
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            output[xi] = x[i]

        return output

    def clip_coords(self,boxes, img_shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        """
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2
        """
        boxes[:,0] = boxes[:, 0].clip(0, img_shape[1])  # x1
        boxes[:, 1] = boxes[:, 1].clip(0, img_shape[0])  # y1
        boxes[:, 2] = boxes[:, 2].clip(0, img_shape[1])  # x2
        boxes[:, 3] = boxes[:, 3].clip(0, img_shape[0])  # y2        

    def scale_coords(self,img1_shape, coords, img0_shape, ratio_pad=None):
        '''
        '''

        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new,计算缩放比率
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
                        img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding ，计算扩充的尺寸
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding，减去x方向上的扩充
        coords[:, [1, 3]] -= pad[1]  # y padding，减去y方向上的扩充
        coords[:, :4] /= gain  # 将box坐标对应到原始图像上
        self.clip_coords(coords, img0_shape)  # 边界检查
        return coords

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))



    def infer_aim(self,src_img):
        img_size = self.aim_size
        conf_thres=0.5 
        iou_thres=0.45 

        src_size=src_img.shape[:2]

        img=self.letterbox(src_img,img_size,stride=32)[0]


        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)


        img=img.astype(dtype=np.float32)
        img/=255.0
        img=np.expand_dims(img,axis=0)


        start=time.time()
        input_feed=self.get_input_feed(self.aim_input_name,img)
        #pred=pickle.loads(self.aim_onnx_session).run(output_names=self.aim_output_name,input_feed=input_feed)
        pred=self.aim_onnx_session.run(output_names=self.aim_output_name,input_feed=input_feed)

        #results = torch.tensor(pred)
        results = np.array(pred)
        cast = time.time() - start
        #print("瞄准网络耗时:{}".format(cast))


        results = self.nms(results, conf_thres, iou_thres)
        cast=time.time()-start
        #print("瞄准检测耗时:{}".format(cast))

        img_shape=img.shape[2:]
        #print(img_size)
        det = results[0]
        if det is not None and len(det):
            det[:, :4] = self.scale_coords(img_shape, det[:, :4],src_size).round()
            #det[:, :4] = self.xyxy2xywh(det[:,:4])
        
        if self.is_record:
            self.draw(src_img, det)
        
        return det

    def infer_alert(self,src_img):
        img_size = self.alert_size
        conf_thres=0.7
        iou_thres=0.45 

        # process src_img
        src_size=src_img.shape[:2]
        h,w = src_size
        left_pad = int(w*(1-self.w_croprate)/2)
        crop_img = src_img[0:int(h*self.h_croprate), int(w*(1-self.w_croprate)/2): int(w*(1-self.w_croprate)/2) + int(w*self.w_croprate)]
        crop_size = crop_img.shape[:2]

        img=self.letterbox(crop_img,img_size,stride=32)[0]


        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)


        img=img.astype(dtype=np.float32)
        img/=255.0
        img=np.expand_dims(img,axis=0)


        start=time.time()
        input_feed=self.get_input_feed(self.alert_input_name,img)
        #pred=pickle.loads(self.alert_onnx_session).run(output_names=self.alert_output_name,input_feed=input_feed)
        pred=self.alert_onnx_session.run(output_names=self.alert_output_name,input_feed=input_feed)

        #results = torch.tensor(pred)
        results = np.array(pred)
        cast = time.time() - start
        #print("弱点网络耗时:{}".format(cast))


        results = self.nms(results, conf_thres, iou_thres)
        cast=time.time()-start
        #print("弱点检测耗时:{}".format(cast))

        img_shape=img.shape[2:]
        #print(img_size)
        det = results[0]
        if det is not None and len(det):
            det[:, :4] = self.scale_coords(img_shape, det[:, :4],crop_size).round()
            #det[:, :4] = self.xyxy2xywh(det[:,:4])

            # add left padding
            det[:, [0,2]] += left_pad
        if self.is_record:
            self.draw(src_img, det)

        return det

    def plot_one_box(self,x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 2  # line/font thickness
        #print(tl)
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        #print(c1,c2)
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def draw(self,img, boxinfo):
        label_map = {0:'alert',1:'aim_box',2:'drill'}
        img = img.astype(np.uint8)
        colors = [[0, 0, 255],[0,255,0],[255,0,0]]
        if not boxinfo is None and len(boxinfo):
            for *xyxy, conf, cls in boxinfo:
                label = '%s %.2f' % (label_map[int(cls)], conf)
                #print('xyxy: ', xyxy)
                self.plot_one_box(xyxy, img, label=label, color=colors[int(cls)], line_thickness=1)

        cv2.imwrite(os.path.join(self.record_log,'{}.jpg'.format(self.frame_cnt)), img)
        self.frame_cnt += 1
        return 0


if __name__=="__main__":
    from easydict import EasyDict as edict
    config = edict()
    config.aim_onnx_path = "./yolov5n_640.onnx"
    config.alert_onnx_path = './yolov5t2_384.onnx'
    config.onnx_num_thread=2
    model=YOLOV5_ONNX(config)
    img_path="1_371.jpg"
    img = cv2.imread(img_path) # BGR
    for i in range(100):
        print(model.infer_aim(img))
        print(model.infer_alert(img))