import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import json
from perspective_transform import perspective_transform

# Global variables (just to make the moviepy video annotation work)
with open('calibrate_camera.p', 'rb') as f:
	save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']


# MoviePy video annotation will call this function
def annotate_image(data):
	# Undistort, threshold, perspective transform
	in_img = cv2.imread(data['in_file'])
	undist = cv2.undistort(in_img, mtx, dist, None, mtx)
	#img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(undist)
	img = undist
	binary_warped, binary_unwarped, m, m_inv = perspective_transform(img, src=np.float32(data['src']), dst=np.float32(data['dest']))
	result = np.vstack((undist, binary_warped))
	save_and_display_image(result, data)


def save_and_display_image(image: np.ndarray, data):
	cv2.imwrite(data['out_file'], image)
	plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	plt.show()


def main():
	with open('def.json', 'r') as myfile:
		str = myfile.read().replace('\n', '')
		data = json.loads(str)
		print(data)
		annotate_image(data)


if __name__ == '__main__':
	main()


