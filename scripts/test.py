import numpy as np
import cv2
from collections import deque
import argparse
  
# 3 10 12
lower_red = np.array([3, 10, 100])
upper_red = np.array([190, 255, 255])
 
img = cv2.imread("/home/zuzu/pict/color.jpg")
# Display the resulting frame
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

red_mask = cv2.inRange(hsv, lower_red, upper_red)
cv2.imshow('mask', red_mask)
red_mask = cv2.erode(red_mask, None, iterations=2)
cv2.imshow('erode', red_mask)
red_mask = cv2.dilate(red_mask, None, iterations=2)
cv2.imshow('dilate', red_mask)
image, contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.waitKey(0)
# circles = None
# circles = cv2.HoughCircle
# print(hierarchy)
# for i in range(len(contours)):
#     print(len(contours[i]))
#     print('area of {} is {}'.format(i, cv2.contourArea(contours[i])))
#     print('length of {} is {}'.format(i, cv2.arcLength(contours[i], True)))

# if len(contours) > 0:
#     c = max(contours, key = cv2.contourArea)
#     ((x, y), radius) = cv2.minEnclosingCircle(c)
#     center = (int(x), int(y))
#     radius = int(radius)
#     circle = cv2.circle(img, center, radius, (0, 255, 0), 2)
#     cv2.imshow('circle', circle)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     print('center is = {}'.format(center))
#     print('radius = {}'.format(radius))

# cv2.drawContours(img,contours,-1,(0,0,255),3)  
# cv2.imshow("img", img)  
# cv2.waitKey(0)

# center = None
# if len(cnts) > 0:
#     c = max(cnts, key=cv2.contourArea)
#     ((x, y), radius) = cv2.minEnclosingCircle(c)
    # M = cv2.moments(c)
    # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    # if cv2.waitKey(1) & 0xFF == ord('q'):
        
#     if radius > 10:
#         cv2.circle(frame, (int(x), int(y)), int(radius),
#                    (0, 255, 255), 2)
#         cv2.circle(frame, center, 5, (0, 0, 255), -1)
# pts.appendleft(center)
# for i in xrange(1, len(pts)):
#     if pts[i - 1] is None or pts[i] is None:
#         continue

#     thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
#     cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
# cv2.imshow("Frame", frame)
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break
# cap.release()
# cv2.destroyAllWindows()