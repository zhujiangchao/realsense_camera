#!/usr/bin/python
# READY FOR MIT
import collections
import time
import numpy as np
import cv2
import datetime
import os
import glob

class Camera_detection():
    """
        Node that publishes the string 'detected' every time an obstacle is detected;
        otherwise, the string 'nothing' is published.

        An obstacle is detected when more than half of the images within the given averaging time 
        have a detection area larger than the threshold. 
        The detection area is the fraction of the total frame that is covered
        by a colour within the given colour range.
        both the color range and the minimum needed area for obstacle detection come from ROS param.

        All photos taken for the obstacle detection are saved in ~/camera_detect_obstacle_$datetime
    """
    def __init__(self):
        
        sensor_rate = 5

        self.Lower_hsv1 = np.array([0, 32, 95])
        self.Upper_hsv1 = np.array([10,255,255])
        self.Lower_hsv2 = np.array([170, 32, 90])
        self.Upper_hsv2 = np.array([185,255,255])

        self.threshold = 0.04
        AVE_TIME = 1  # lengh of the averaging in seconds
        self.AVE_SIZE = int(AVE_TIME * sensor_rate)                        # size of the averaging sample
        self.average_list = collections.deque(maxlen = self.AVE_SIZE)
        self.empty_image_counter = 0   # only save every empty_image_ignore-th empty image
        self.empty_image_ignore = 5
        self.detection()

    def detection(self, dir):
        imagelist = glob.glob('/home/zuzu/pict' + '*.jpg') + glob.glob('/home/zuzu/pict' + '*.jpeg')

        if len(imagelist) == 0:
            print('error!!!')
            return
        for fn in imagelist:
            image = cv2.imread(fn)
            image_size = image.shape[0]*image.shape[1]
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask_hsv1 = cv2.inRange(hsv, self.Lower_hsv1, self.Upper_hsv1)
            cv2.imshow('mask_hsv1', mask_hsv1)
            percent_detect1 = 1.0*cv2.countNonZero(mask_hsv1)  / image_size # percentage of the image that contains the expected colors
            print("percent_detect1 = {}".format(percent_detect1))
            # if only 1 set of color is used, we compute only 1 mask
            if self.Lower_hsv2[0] != 0 or self.Upper_color_hsv2[0] != 0:
                mask_hsv2 = cv2.inRange(hsv, self.Lower_hsv2, self.Upper_hsv2)
                cv2.imshow("mask_hsv2", mask_hsv2)
                percent_detect2 = 1.0*cv2.countNonZero(mask_hsv2)  / image_size # percentage of the image that contains the expected colors
                print("percent_detect2 = {}".format(percent_detect2))            
            else:
                percent_detect2 = 0
            
            if percent_detect1 + percent_detect2  >= self.threshold: 
                self.average_list.append(1)
            else:
                self.average_list.append(0)
            
            if 1.0*sum(self.average_list)/self.AVE_SIZE > 0.5:
                msg = 'detected'
                filename = os.path.expanduser('~/camera_detected_obstacle_{:%Y-%m-%d_%H:%M:%S_%f}.jpg'.format(datetime.datetime.now()))
                cv2.imwrite(filename, image)
            else:
                msg = 'nothing'
                self.empty_image_counter +=1
                if self.empty_image_counter > self.empty_image_ignore:
                    filename = os.path.expanduser('~/camera_detected_obstacle_{:%Y-%m-%d_%H:%M:%S_%f}_nothing.jpg'.format(datetime.datetime.now()))
                    cv2.imwrite(filename, image)
                    self.empty_image_counter = 0
    


if __name__ == '__main__':
    Camera_detection()


