import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

image_1 = cv2.imread("image/fce(salt_pepper noise).bmp")
image_2 = cv2.imread("image/Copy of Lena-Gaussian-noise2.jpg")

cv2.imshow("image 1", image_1)
cv2.imshow("image 2", image_2)  

for ksize in (3, 7, 11, 17):
    re_image_1 = cv2.GaussianBlur(image_1, (ksize, ksize), 1.0)
    re_image_2 = cv2.GaussianBlur(image_2, (ksize, ksize), 1.0)

    desc = "Mean : {}x{},".format(ksize,ksize)
    cv2.putText(re_image_1, desc, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
    cv2.putText(re_image_2, desc, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
    


    cv2.imshow("Gaussian Filter 1", re_image_1)
    cv2.imshow("Gaussian Filter 2", re_image_2)
    
    cv2.waitKey()

cv2.destroyAllWindows()