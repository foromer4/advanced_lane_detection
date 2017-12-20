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


# MoviePy video annotation will call this function
def annotate_image(img_in):
	global counter
	counter += 1
	reset = counter % 8 == 0
	img_bgr = cv2.cvtColor(img_in,cv2.COLOR_RGB2BGR)
	# Undistort, threshold, perspective transform
	undist = cv2.undistort(img_bgr, mtx, dist, None, mtx)

	#img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(undist)
	img = undist
	binary_warped, binary_unwarped, m, m_inv = perspective_transform(img, src=np.float32(data['src']), dst=np.float32(data['dest']))


	optical_Result = optical_flow_dense.processFrame(img, binary_warped, reset= reset)
	#optical_Result = optical_flow_sparse.processFrame(img, binary_warped, reset= reset)

	return cv2.cvtColor(optical_Result,cv2.COLOR_BGR2RGB)



def annotate_video(input_file, output_file):
	with open('def.json', 'r') as myfile:
		str = myfile.read().replace('\n', '')
		global data
		data= json.loads(str)

	""" Given input_file video, save annotated video to output_file """
	video = VideoFileClip(input_file)
	annotated_video = video.fl_image(annotate_image)
	annotated_video.write_videofile(output_file, audio=False)


if __name__ == '__main__':
	# Annotate the video
	annotate_video('/home/omer/work/temp/direct_video.mp4', 'out.mp4')


