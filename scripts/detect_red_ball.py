#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import collections
import os
import datetime
import glob
import time

class image_converter:
	def __init__(self, color_topic, depth_topic):
		# self.image_pub = rospy.Publisher("image_topic_2",Image)
		self.bridge = CvBridge()
		# self.lower_red = np.array([150, 100, 100])
		# self.upper_red = np.array([190, 255, 255])
		
		self.Lower_hsv1 = np.array([0, 32, 95])
		self.Upper_hsv1 = np.array([10,255,255])
		self.Lower_hsv2 = np.array([170, 32, 90])
		self.Upper_hsv2 = np.array([185,255,255])

		self.threshold = 0.005
		sensor_rate = 5
		AVE_TIME = 1  # lengh of the averaging in seconds
		self.AVE_SIZE = int(AVE_TIME * sensor_rate)                        # size of the averaging sample
		self.average_list = collections.deque(maxlen = self.AVE_SIZE)
		self.empty_image_counter = 0   # only save every empty_image_ignore-th empty image
		self.empty_image_ignore = 5

		self.color_topic = color_topic
		self.depth_topic = depth_topic
		self.color_image = None
		self.depth_image = None
		self.image_sub = rospy.Subscriber(self.color_topic, Image,self.colorCallBack)
		self.image_sub = rospy.Subscriber(self.depth_topic, Image,self.depthCallBack)		

	def colorCallBack(self,data):
		try:
			self.color_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

	def depthCallBack(self, data):
		try:
			# print(data.encoding)
			self.depth_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
		except CvBridgeError as e:
			print(e)
		self.redBallDection()

	def redBallDection(self):
		# print("callBack")
		if self.color_image is None or self.depth_image is None:
			rospy.logerr("None pict")
			return 
		else:
			# rospy.logwarn("haha, ok!")
			# Convert images to numpy arrays
			depth_image = np.asanyarray(self.depth_image)
			color_image = np.asanyarray(self.color_image)
			image_size = color_image.shape[0] * color_image.shape[1]
			hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
			mask_hsv1 = cv2.inRange(hsv, self.Lower_hsv1, self.Upper_hsv1)
			# mask_hsv1 = cv2.GaussianBlur(mask_hsv1, (9,9), 2, 2)
			# cv2.imshow('mask_hsv1', mask_hsv1)
			# cv2.imwrite('mask_hsv1.jpg', mask_hsv1)
			percent_detect1 = 1.0*cv2.countNonZero(mask_hsv1)  / image_size # percentage of the image that contains the expected colors
			# if only 1 set of color is used, we compute only 1 mask
			if self.Lower_hsv2[0] != 0 or self.Upper_color_hsv2[0] != 0:
				mask_hsv2 = cv2.inRange(hsv, self.Lower_hsv2, self.Upper_hsv2)
				# mask_hsv2 = cv2.GaussianBlur(mask_hsv2, (9,9), 2, 2)
				# cv2.imwrite('mask_hsv2.jpg', mask_hsv2)
				cv2.imshow('mask_hsv2', mask_hsv2)
				percent_detect2 = 1.0*cv2.countNonZero(mask_hsv2)  / image_size # percentage of the image that contains the expected colors
			else:
				percent_detect2 = 0

			print("sum = ", percent_detect1 + percent_detect2)
			if percent_detect1 + percent_detect2  >= self.threshold: 
				circles1 = None
				circles2 = None
				circles1 = cv2.HoughCircles(mask_hsv1, cv2.HOUGH_GRADIENT, 1, 1000, param1=200, param2=15, minRadius=20, maxRadius=500)

				if percent_detect2 != 0:
					
					circles2 = cv2.HoughCircles(mask_hsv2, cv2.HOUGH_GRADIENT, 1, 1000, param1=200, param2=5, minRadius=20, maxRadius=500)

				if circles1 is not None and circles2 is not None:
					# rospy.logwarn("")
					
					# if circles1 is not None:
					# 	# find the max radius
					# 	circle1 = circles1[0, np.argmax(circles1[0, :, 2]), :]
					# 	i1 = np.uint16(np.around(circle1))
					# 	image1 = cv2.circle(color_image,(i1[0],i1[1]),i1[2],(255,0,0),5)
					# 	image1 = cv2.circle(image1,(i1[0],i1[1]),2,(255,0,255),10)
					# 	image1 = cv2.rectangle(image1,(i1[0]-i1[2],i1[1]+i1[2]),(i1[0]+i1[2],i1[1]-i1[2]),(255,255,0),5)
					# 	cv2.imwrite("image1.jpg", image1)
					# 	# cv2.imshow("image1", image1)
					# 	#打印圆心位置 和 圆形的距离 单位mm
					# 	print("center, radius",i1[0],i1[1],i1[2])

					if circles2 is not None:
						self.average_list.append(1)
						circle2 = circles2[0, np.argmax(circles2[0, :, 2]), :]
						i2 = np.uint16(np.around(circle2))
						cv2.circle(color_image,(i2[0],i2[1]),i2[2],(255,0,0),5)
						cv2.circle(color_image,(i2[0],i2[1]),2,(255,0,255),10)
						cv2.rectangle(color_image,(i2[0]-i2[2],i2[1]+i2[2]),(i2[0]+i2[2],i2[1]-i2[2]),(255,255,0),5)
						# cv2.imwrite("image2.jpg", image2)
						print("center, radius",i2[0],i2[1],i2[2])  

					# else:
					# 	self.average_list.append(0)

				else:
					self.average_list.append(0)
			else:
				self.average_list.append(0)
			cv2.imshow("image2", color_image)
			cv2.waitKey(100)
			if 1.0*sum(self.average_list)/self.AVE_SIZE > 0.5:
				msg = 'detected'
				rospy.logwarn(msg)
				filename = os.path.expanduser('~/camera_detected_obstacle_{:%Y-%m-%d_%H:%M:%S_%f}.jpg'.format(datetime.datetime.now()))
			# cv2.imwrite(filename, image)
			else:
				msg = 'nothing'
				# rospy.logwarn(msg)
				self.empty_image_counter +=1
				if self.empty_image_counter > self.empty_image_ignore:
					filename = os.path.expanduser('~/camera_detected_obstacle_{:%Y-%m-%d_%H:%M:%S_%f}_nothing.jpg'.format(datetime.datetime.now()))
					# cv2.imwrite(filename, image)
					self.empty_image_counter = 0

 
def main(args):
	color_topic = '/camera/color/image_raw'
	depth_topic = '/camera/depth/image_rect_raw'
	ic = image_converter(color_topic, depth_topic)
	rospy.init_node('image_converter', anonymous=True)
	# rospy.rate(10)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
		cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)