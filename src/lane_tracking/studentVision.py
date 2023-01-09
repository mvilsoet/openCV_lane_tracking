import time
import math
import numpy as np
import cv2
import rospy

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology



class lanenet_detector():
	def __init__(self):

		self.bridge = CvBridge()
		# NOTE
		# Uncomment this line for lane detection of GEM car in Gazebo
		self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
		# Uncomment this line for lane detection of videos in rosbag
		# self.sub_image = rospy.Subscriber('camera/image_raw', Image, self.img_callback, queue_size=1)
		self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
		self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
		self.pub_grad = rospy.Publisher("lane_detection/grad_thresh", Image, queue_size=1)
		self.pub_color = rospy.Publisher("lane_detection/color_thresh", Image, queue_size=1)
		self.pub_combined = rospy.Publisher("lane_detection/combined_thresh", Image, queue_size=1)
		self.left_line = Line(n=5)
		self.right_line = Line(n=5)
		self.detected = False
		self.hist = True


	def img_callback(self, data):

		try:
			# Convert a ROS image message into an OpenCV image
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		raw_img = cv_image.copy()
		# cv2.imwrite('1_raw_img.png',raw_img)
		mask_image, bird_image, color_image, gradient_image, combined_image = self.detection(raw_img)
		# cv2.imwrite('1_mask_image.png',mask_image)
		# cv2.imwrite('1_bird_image.png',bird_image)

		if mask_image is not None and bird_image is not None:
			# Convert an OpenCV image into a ROS image message
			out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
			out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')
			out_grad_msg = self.bridge.cv2_to_imgmsg(gradient_image.astype(np.uint16), 'mono16')
			out_color_msg = self.bridge.cv2_to_imgmsg(color_image.astype(np.uint16), 'mono16')
			out_combined_msg = self.bridge.cv2_to_imgmsg(combined_image.astype(np.uint16), 'mono16')

			# Publish image message in ROS
			self.pub_image.publish(out_img_msg)
			self.pub_bird.publish(out_bird_msg)
			self.pub_grad.publish(out_grad_msg)
			self.pub_color.publish(out_color_msg)
			self.pub_combined.publish(out_combined_msg)




	def gradient_thresh(self, img, thresh_min=230, thresh_max=255):#test_realworld picture:150, 180 gazebo:230, 255 rosbag: 200,255
		"""
		Apply sobel edge detection on input image in x, y direction
		"""
		#1. Convert the image to gray scale
		#2. Gaussian blur the image
		#3. Use cv2.Sobel() to find derievatives for both X and Y Axis
		#4. Use cv2.addWeighted() to combine the results
		#5. Convert each pixel to unint8, then apply threshold to get binary image

		## TODO
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #1

		gray_blurred_img = cv2.GaussianBlur(gray_img,ksize=(5,5),sigmaX=0)  #2

		grad_x = cv2.Sobel(gray_blurred_img, cv2.CV_64F,1,0,ksize=5)  #3
		grad_y = cv2.Sobel(gray_blurred_img, cv2.CV_64F,0,1,ksize=5)

		abs_grad_x = cv2.convertScaleAbs(grad_x)
		abs_grad_y = cv2.convertScaleAbs(grad_y)
		grad = cv2.addWeighted(abs_grad_x, .5, abs_grad_y, .5, 0)  #4
		img_uint8 = grad
		# print(img_uint8.shape)
		binary_output = np.zeros(img_uint8.shape)  #5
		binary_output = ((img_uint8[:,:] >= thresh_min).astype(np.uint8) & (img_uint8[:,:] <= thresh_max).astype(np.uint8))

		# for column in range(img_uint8.shape[0]):
		# 	for row in range(img_uint8.shape[1]):
		# 		# print(img_uint8[column][row])
		# 		binary_output[column][row] = 1 if img_uint8[column][row]>=thresh_min and img_uint8[column][row]<=thresh_max else 0
		
		# binary_output_255 = 255 * binary_output
		# cv2.imwrite('1_gardient_thresh.png',binary_output_255)
		return binary_output

	def color_thresh(self, img, thresh=(130, 160)): #test real world picture: 100,255 gazebo: 130,160 rosbag: 210, 250
		"""thresh=(100, 255)
		Convert RGB to HSL and threshold to binary image using S channel
		"""
		#1. Convert the image from RGB to HSL
		img_HSL = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
		#2. Apply threshold on S channel to get binary image
		img_h  = img_HSL.shape[0]
		img_w = img_HSL.shape[1]
		binary_output = np.zeros([img_h, img_w])
		binary_output = (img_HSL[:, :, 1] > thresh[0]).astype(np.uint8)
		# print(binary_output)

		# for i in range(img_h):
		# 	for j in range(img_w):
		# 		if img_HSL[i][j][1] < thresh[0]:
		# 			binary_output[i][j] = 0
		# 		else:
		# 			binary_output[i][j] = 1 
		
		# binary_output_255 = 255 * binary_output
		# cv2.imwrite('1_color_thresh.png',binary_output_255)
		return binary_output


	def combinedBinaryImage(self, img):
		"""
		Get combined binary image from color filter and sobel filter
		"""
		#1. Apply sobel filter and color filter on input image
		#2. Combine the outputs
		## Here you can use as many methods as you want.
		ColorOutput = self.color_thresh(img)
		SobelOutput = self.gradient_thresh(img)
		binaryImage = np.zeros(SobelOutput.shape)
		binaryImage[(ColorOutput==1) | (SobelOutput==1)] = 1
		# Remove noise from binary image
		binaryImage = (morphology.remove_small_objects(binaryImage.astype('bool'),min_size=50,connectivity=2)).astype(np.uint8) 
		return binaryImage, SobelOutput, ColorOutput


	def perspective_transform(self, img, verbose=False):
		"""
		Get bird's eye view from input image
		"""
		#1. Visually determine 4 source points and 4 destination points
		# All points are in format [cols, rows]
		# pt_A = [187, 144]
		# pt_B = [81, 210]
		# pt_C = [393, 210]
		# pt_D = [236, 144]
		
		#for test real world picture
		# pt_A = [149, 168]
		# pt_B = [81, 208]
		# pt_C = [388, 208]
		# pt_D = [292, 168]

		# #for test gazebo picture
		pt_A = [255, 275]
		pt_B = [38, 406]
		pt_C = [633,406]
		pt_D = [395, 275]		
		
		# pt_A = [278, 275]
		# pt_D = [405, 275]	
		
		#for gazebo bag picture
		# pt_A = [500, 241]
		# pt_B = [350, 348]
		# pt_C = [800 ,348]
		# pt_D = [740, 241]	
		#2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
		# Here, I have used L2 norm. You can use L1 also.
		width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
		width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
		maxWidth = max(int(width_AD), int(width_BC))


		height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
		height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
		maxHeight = max(int(height_AB), int(height_CD))

		input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
		output_pts = np.float32([[0, 0], [0, maxHeight - 1], [maxWidth - 1, maxHeight - 1],[maxWidth - 1, 0]])

		# Compute the perspective transform M
		M = cv2.getPerspectiveTransform(input_pts,output_pts)

		#3. Generate warped image in bird view using cv2.warpPerspective()
		out_img = cv2.warpPerspective(img,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
		
		
		# out_img_255 = 255 * out_img
		# cv2.imwrite('1_out_img.png',out_img_255)
		# cv2.imshow('output_img',out_img)
		# cv2.waitKey(0)

		return out_img, np.linalg.inv(M)


	def detection(self, img):

		region_of_interest_img, SobelOutput, ColorOutput = self.combinedBinaryImage(img)
		img_birdeye, Minv = self.perspective_transform(region_of_interest_img)
		# cv2.imshow('output_img',region_of_interest_img)
		# cv2.waitKey(0)

		if not self.hist:
			# Fit lane without previous result
			ret = line_fit(img_birdeye)
			left_fit = ret['left_fit']
			right_fit = ret['right_fit']
			nonzerox = ret['nonzerox']
			nonzeroy = ret['nonzeroy']
			left_lane_inds = ret['left_lane_inds']
			right_lane_inds = ret['right_lane_inds']

		else:
			# Fit lane with previous result
			if not self.detected:
				ret = line_fit(img_birdeye)

				if ret is not None:
					left_fit = ret['left_fit']
					right_fit = ret['right_fit']
					nonzerox = ret['nonzerox']
					nonzeroy = ret['nonzeroy']
					left_lane_inds = ret['left_lane_inds']
					right_lane_inds = ret['right_lane_inds']

					left_fit = self.left_line.add_fit(left_fit)
					right_fit = self.right_line.add_fit(right_fit)

					self.detected = True

			else:
				left_fit = self.left_line.get_fit()
				right_fit = self.right_line.get_fit()
				ret = tune_fit(img_birdeye, left_fit, right_fit)

				if ret is not None:
					left_fit = ret['left_fit']
					right_fit = ret['right_fit']
					nonzerox = ret['nonzerox']
					nonzeroy = ret['nonzeroy']
					left_lane_inds = ret['left_lane_inds']
					right_lane_inds = ret['right_lane_inds']

					left_fit = self.left_line.add_fit(left_fit)
					right_fit = self.right_line.add_fit(right_fit)

				else:
					self.detected = False

			# Annotate original image
			bird_fit_img = None
			combine_fit_img = None
			if ret is not None:
				bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)
				combine_fit_img = final_viz(img, left_fit, right_fit, Minv)
			else:
				print("Unable to detect lanes")
			return combine_fit_img, bird_fit_img, ColorOutput, SobelOutput, region_of_interest_img


if __name__ == '__main__':
	# init args
	
	rospy.init_node('lanenet_node', anonymous=True)
	lanenet_detector()
	while not rospy.core.is_shutdown():
		rospy.rostime.wallsleep(0.5)
	
	# process = lanenet_detector()
	# img  = cv2.imread('/home/yuchenc2/Desktop/test')
	# output_img = process.color_thresh(img, 150)

	# img  = cv2.imread('/home/yuchenc2/Desktop/ECE484/gazebo_test.png')
	# region_of_interest_img, img_birdeye = process.detection(img)

	# cv2.imwrite('raw_img.png',raw_img)
	# cv2.imshow('output_img',region_of_interest_img)
	# cv2.imwrite('output_img1.png',img_birdeye)
	# cv2.imshow('output_img1',img_birdeye)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()