import numpy as np
import random
import Tool_Functions.Functions as Functions

import cv2  # importing cv
import imutils

# read an image as input using OpenCV
image = Functions.convert_png_to_np_array('/home/zhoul0a/Desktop/codes_for_home_work/AMCS329/2d_absolute_error_map.png')
print(np.shape(image))
image = image[:, :, 2]

Rotated_image = imutils.rotate(image, center=(480, 640), angle=45, scale=0.5)
Rotated1_image = imutils.rotate(image, angle=90)
print(np.shape(Rotated_image))
Functions.image_show(Rotated_image)
exit()
# display the image using OpenCV of
# angle 45
cv2.imshow("Rotated", Rotated_image)
exit()

# display the image using OpenCV of
# angle 90
cv2.imshow("Rotated", Rotated1_image)

# This is used for To Keep On Displaying
# The Image Until Any Key is Pressed
cv2.waitKey(0)