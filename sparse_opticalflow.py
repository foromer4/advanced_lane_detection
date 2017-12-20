import cv2
import numpy as np
from builtins import str


class SparseOpticalFlow:

    def __init__(self):
        self.old_gray = None
        self.p0 = None

    def processFrame(self,original_frame: np.ndarray, in_frame: np.ndarray, reset: bool):
        frame = in_frame.copy()
        if  frame.dtype == np.float64:
            frame = np.asarray(frame * 256, dtype=np.uint8)

        if self.p0 is not None and not reset:
            mask = np.zeros_like(frame)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)
            if p1 is None:
                print(err)
                p1 = self.p0
            # Select good points
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]
            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(img=mask, pt1=(a,b), pt2=(c, d), color=self.color[i].tolist(), thickness=5)
                frame = cv2.circle(img=frame, center=(a, b), radius=25, color=self.color[i].tolist(),thickness=8)
            result = cv2.add(frame, mask)


            # Now update the previous frame and previous points
            self.old_gray = frame_gray
            self.p0 = good_new.reshape(-1, 1, 2)
            stacked = np.vstack((original_frame,in_frame, result, mask))
            final_result = cv2.resize(src=stacked, dsize=None, fx=0.25, fy=0.25,interpolation=cv2.INTER_CUBIC)
            cv2.imshow('frame', final_result)
            cv2.waitKey(30) & 0xff
            return result

        else:
            self.feature_params = dict(maxCorners=100,
                                  qualityLevel=0.3,
                                  minDistance=7,
                                  blockSize=7)
            # Parameters for lucas kanade optical flow
            self.lk_params = dict(winSize=(15, 15),
                             maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            self.old_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
            self.color = np.random.randint(0, 255, (100, 3))
            return np.zeros_like(frame)







