import numpy as np
import cv2
import pickle
import json
from perspective_transform import perspective_transform
from Line import Line
from moviepy.editor import VideoFileClip
from dense_opticalflow import DenseOpticalFlow
from sparse_opticalflow import SparseOpticalFlow
from line_fit import combined_thresh
from lane import Lane
import matplotlib.pyplot as plt



optical_flow_sparse = SparseOpticalFlow()
optical_flow_dense = DenseOpticalFlow()
counter = 0

# Global variables (just to make the moviepy video annotation work)
with open('calibrate_camera.p', 'rb') as f:
	save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']
window_size = 5  # how many frames for line smoothing
left_line = Line(n=window_size)
right_line = Line(n=window_size)
detected = False  # did the fast line fit detect the lines?
left_curve, right_curve = 0., 0.  # radius of curvature for left and right lanes
left_lane_inds, right_lane_inds = None, None  # for calculating curvature
lane = Lane()

# MoviePy video annotation will call this function
def annotate_image(img_in):
	global counter
	counter += 1
	reset = counter % 8 == 0
	img_bgr = cv2.cvtColor(img_in,cv2.COLOR_RGB2BGR)
	# Undistort, threshold, perspective transform
	undist = cv2.undistort(img_bgr, mtx, dist, None, mtx)
	warped = data['warped']
	if not warped:
		binary_warped, binary_unwarped, m, m_inv = perspective_transform(undist, src=np.float32(data['src']), dst=np.float32(data['dest']))
		lines, mask = lane.find_lines(binary_warped)
		lane.draw_lines_on_image(lines, binary_warped)
		unwarped_laned_image = cv2.warpPerspective(binary_warped, m_inv, (binary_warped.shape[1],
																		  binary_warped.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG

		result = np.vstack((img_in,mask, binary_warped, unwarped_laned_image))
		result = cv2.resize(src=result, dsize=None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
		cv2.imshow('frame', result)
		cv2.waitKey(30) & 0xff
	else:
		img_out = img_in.copy()
		lines, mask = lane.find_lines(img_in)
		lane.draw_lines_on_image(lines, img_out)
		result = np.vstack((img_in, mask, img_out))
		result = cv2.resize(src=result, dsize=None, fx=0.33, fy=0.33, interpolation=cv2.INTER_LINEAR)
		cv2.imshow('frame', result)
		cv2.waitKey(30) & 0xff
	return result



def annotate_video():
	with open('def.json', 'r') as myfile:
		str = myfile.read().replace('\n', '')
		global data
		data= json.loads(str)
		in_video = data['in_video']
		out_video = data['out_video']

	""" Given input_file video, save annotated video to output_file """
	video = VideoFileClip(in_video)
	annotated_video = video.fl_image(annotate_image)
	annotated_video.write_videofile(out_video, audio=False)


if __name__ == '__main__':
	# Annotate the video
	annotate_video()


