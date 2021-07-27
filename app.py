from model_process import process_detect
import numpy as np
import cv2
import imutils
import math

VIDEO_SOURCE = "video/example_02.mp4"


vs = cv2.VideoCapture(VIDEO_SOURCE)
_, frame = vs.read()
frame = imutils.resize(frame, width=640)
(H, W) = frame.shape[:2]

UP_x0, UP_y0 = 0, H//2
UP_x1, UP_y1 = W, H//2

DOWN_x0, DOWN_y0 = 0, H//2
DOWN_x1,  DOWN_y1 = W, H//2



p0_up = np.array((UP_x0, UP_y0))
p1_up = np.array((UP_x1, UP_y1))
p0_down = np.array((DOWN_x0, DOWN_y0))
p1_down = np.array((DOWN_x1, DOWN_y1))

#show_lines(frame, p0_up, p1_up, p0_down, p1_down)

deltaY = UP_y1 - UP_y0
deltaX = UP_x1 - UP_x0
y_line = False
angle = abs(math.atan2(deltaY, deltaX)*180/math.pi)

if (angle >= 0 and angle < 45) or (angle > 135 and angle <= 180):
    y_line=True
else:
    y_line=False
    
process_detect(vs, p0_up, p1_up, p0_down, p1_down, y_line)


