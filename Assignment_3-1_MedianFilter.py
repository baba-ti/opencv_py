import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

image_1 = cv2.imread("image/fce(salt_pepper noise).bmp")
image_2 = cv2.imread("image/Copy of Lena-Gaussian-noise2.jpg")

cv2.imshow("image 1", image_1)
cv2.imshow("image 2", image_2)  


