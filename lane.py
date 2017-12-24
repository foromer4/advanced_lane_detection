import numpy as np
import cv2
class Lane:

    def __init__(self):
        pass


    def process_image(self, image: np.ndarray):
        bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.__find_bright_pixles(bw_image, 15)

    def __find_bright_pixles(self, image: np.ndarray, margin: int):
        result = np.zeros_like(image)
        for y in range(0, result.shape[0]):
            for x in range(margin, result.shape[1] - margin):
                marg_dif = abs((int)(image[y, x-margin]) - (int)(image[y, x+margin]))
                bright_dif = 2 * (int)(image[y, x]) - (int)(image[y, x - margin]) - (int)(image[y, x + margin]) - marg_dif
                brightness = min(255, bright_dif)
                if brightness > 0:
                    result[y,x] = brightness
        return result



    def __find_white_yellow_masks(self, image: np.ndarray):
        pass


    def __hough_transform(self, image: np.ndarray):
        pass