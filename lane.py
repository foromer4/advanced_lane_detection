import math
import numpy as np
import cv2
class Lane:

    def __init__(self):
        self.max_lines = 5


    def find_lines(self, image: np.ndarray):
        bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bright_pixels = self.__find_bright_pixles(bw_image, 15)
        yellow_mask, white_mask = self.__find_white_yellow_masks(image)
        result = self.__integrate_masks(bright_pixels=bright_pixels,yellow_mask=yellow_mask, white_mask=white_mask)
        result = result * 255
        lines , mask= self.__hough_transform(result)
        return lines , mask

    def draw_lines_on_image(self, lines, image: np.ndarray):
        if lines is not None:
            for line in lines:
                cv2.line(img=image, pt1=(line[2], line[3]), pt2=(line[4], line[5]), color=(0, 0, 255),
                         thickness=2)
        return image


    def __integrate_masks(self, bright_pixels: np.ndarray, yellow_mask: np.ndarray, white_mask: np.ndarray):
        white_or_yellow = cv2.bitwise_or(white_mask, yellow_mask)
        filtered_img = cv2.bitwise_and(bright_pixels,white_or_yellow)
        final_image = cv2.bitwise_or(filtered_img, yellow_mask)
        return final_image

    def __find_bright_pixles(self, image: np.ndarray, margin: int):
        threshold = 13 # 0.05 * 255
        bright_pixels = np.zeros_like(image)
        for y in range(0, bright_pixels.shape[0]):
            for x in range(margin, bright_pixels.shape[1] - margin):
                marg_dif = abs((int)(image[y, x-margin]) - (int)(image[y, x+margin]))
                bright_dif = 2 * (int)(image[y, x]) - (int)(image[y, x - margin]) - (int)(image[y, x + margin]) - marg_dif
                brightness = min(255, bright_dif)
                if brightness > threshold:
                    bright_pixels[y,x] = 1
        return bright_pixels



    def __find_white_yellow_masks(self, image: np.ndarray):
        hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        hsv_image = np.asarray(hsv_image, dtype=np.float64) / 255

        hue_min = 0.1
        hue_max = 0.2
        sat_min = 0.35
        val_min = 0.5
        white_sat_max = 0.15
        white_val_min = 0.4

        im_mask_yellow = np.asarray(np.zeros((hsv_image.shape[0],hsv_image.shape[1])), dtype=np.uint8)
        im_mask_white = np.zeros_like(im_mask_yellow)
        for y in range(0, hsv_image.shape[0]):
            for x in range(0, hsv_image.shape[1]):
                pixel_hsv = hsv_image[y,x]
                if pixel_hsv[0] > hue_min and pixel_hsv[0] < hue_max and pixel_hsv[1] > sat_min and pixel_hsv[2] > val_min:
                    im_mask_yellow[y, x] = 1
                if pixel_hsv[1] < white_sat_max and pixel_hsv[2] > white_val_min:
                    im_mask_white[y, x] = 1
        return im_mask_yellow, im_mask_white

    def __hough_transform(self, image: np.ndarray):
        found_lines = []
        img_with_lines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        lines = cv2.HoughLines(image=image, rho=20, theta=2 * np.pi / 180,threshold=10, min_theta=355 * np.pi / 180, max_theta=365 * np.pi / 180)
        if lines is not None:
            max_line_to_take = min(500, len(lines))
            for i in range(0, max_line_to_take):
                line = lines[i]
                rho = line[0][0]
                theta = line[0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                self.__update_lines(found_lines,(x0, y0, x1,y1,x2,y2), img_with_lines)
            self.draw_lines_on_image(found_lines, img_with_lines)
        return found_lines, img_with_lines


    def __update_lines(self, lines: list, new_line: tuple, image: np.ndarray):
        if len(lines) >= self.max_lines:
            return
        found_close_line = False
        for existing_line in lines:
            if self.__lines_are_close(existing_line, new_line):
                found_close_line = True
                break

        if not found_close_line:
            lines.append(new_line)

    def __lines_are_close(self, line1 , line2):
        #x0,y0, x1,y1,x2,y2 , compare by x2
        close_line_threshold = 50
        return abs(line1[0] - line2[0]) < close_line_threshold or abs(line1[2] - line2[2]) < close_line_threshold\
               or abs(line1[4] - line2[4]) < close_line_threshold


