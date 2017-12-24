import numpy as np
import cv2
class Lane:

    def __init__(self):
        pass


    def process_image(self, image: np.ndarray):
        bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bright_pixels = self.__find_bright_pixles(bw_image, 15)
        yellow_mask, white_mask = self.__find_white_yellow_masks(image)
        result = self.__integrate_masks(bright_pixels=bright_pixels,yellow_mask=yellow_mask, white_mask=white_mask)
        return result * 255


    def __integrate_masks(self, bright_pixels: np.ndarray, yellow_mask: np.ndarray, white_mask: np.ndarray):
        white_or_yellow = cv2.bitwise_or(white_mask, yellow_mask)
        filtered_img = cv2.bitwise_and(bright_pixels,white_or_yellow)
        final_image = cv2.bitwise_or(filtered_img, yellow_mask)
        return final_image

    def __find_bright_pixles(self, image: np.ndarray, margin: int):
        bright_pixels = np.zeros_like(image)
        for y in range(0, bright_pixels.shape[0]):
            for x in range(margin, bright_pixels.shape[1] - margin):
                marg_dif = abs((int)(image[y, x-margin]) - (int)(image[y, x+margin]))
                bright_dif = 2 * (int)(image[y, x]) - (int)(image[y, x - margin]) - (int)(image[y, x + margin]) - marg_dif
                brightness = min(255, bright_dif)
                if brightness > 0:
                    bright_pixels[y,x] = brightness
        _, result_binary = cv2.threshold(src= bright_pixels, thresh=0.05, maxval=1, type=cv2.THRESH_BINARY)
        return result_binary



    def __find_white_yellow_masks(self, image: np.ndarray):
        hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        hsv_image = np.asarray(hsv_image, dtype=np.float64) / 255;

        hue_min = 0.1
        hue_max = 0.2;
        sat_min = 0.35;
        val_min = 0.5;
        white_val_min = 0.4
        white_sat_max = 0.15

        im_mask_yellow = np.asarray(np.zeros((hsv_image.shape[0],hsv_image.shape[1])), dtype=np.uint8)
        im_mask_white = np.zeros_like(im_mask_yellow)
        for y in range(0, hsv_image.shape[0]):
            for x in range(0, hsv_image.shape[1]):
                pixel_hsv = hsv_image[y,x]
                if pixel_hsv[0]  < hue_min and pixel_hsv[0] > hue_max and pixel_hsv[1] > sat_min and pixel_hsv[2] > val_min:
                    im_mask_yellow[y,x] = 1
                if pixel_hsv[1] < white_sat_max and pixel_hsv[2] > white_val_min:
                    im_mask_white[y,x] = 1

        return im_mask_yellow, im_mask_white




    def __hough_transform(self, image: np.ndarray):
        pass