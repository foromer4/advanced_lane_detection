import cv2
import numpy as np
from builtins import str


class DenseOpticalFlow:

    def __init__(self):
        self.prev = None
        self.hsv = None
        self.threshold = 3

    def processFrame(self,original_frame: np.ndarray, in_frame: np.ndarray, reset : bool):
        frame = in_frame.copy()
        if frame.dtype == np.float64:
            frame = np.asarray(frame * 255, dtype=np.uint8)

        if self.prev is not None and not reset:
            if len(frame.shape) == 3:
                next = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            else:
                next = frame
            flow = cv2.calcOpticalFlowFarneback(prev=self.prev,next=next, flow=None, pyr_scale=0.6,
                                                levels=3,winsize=10, iterations=3, poly_n=5, poly_sigma=.2, flags=0)
            flow_x = flow[...,0]
            _, flow_green = cv2.threshold(src=flow_x,thresh=self.threshold,maxval=255, type=cv2.THRESH_TOZERO)
            _, flow_red = cv2.threshold(src=-1. * flow_x, thresh=self.threshold, maxval=255, type=cv2.THRESH_TOZERO)



            mask = np.zeros_like(frame)
            mask[...,1] = np.asarray(flow_green * 20, dtype=np.uint8)
            mask[...,2] = np.asarray(flow_red * 20, dtype=np.uint8)

            # mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            # self.hsv[...,0] = ang*180/np.pi/2
            # self.hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            #mask = cv2.cvtColor(self.hsv,cv2.COLOR_HSV2BGR)
            result = cv2.addWeighted(src1=mask, alpha=0.4, src2=frame, beta=0.6, gamma=0)
            self.prev = next
        else:
            result = frame
            self.prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.hsv = np.zeros_like(frame)
            self.hsv[..., 1] = 255
            mask = np.zeros_like(frame)

        stacked = np.vstack((original_frame, in_frame, result , mask))
        final_result = cv2.resize(src=stacked,dsize=None, fx = 0.25, fy=0.25,  interpolation=cv2.INTER_CUBIC)

        cv2.imshow('frame', result)
        cv2.waitKey(30) & 0xff
        return result







