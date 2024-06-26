import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

image_1 = cv2.imread("image/fce(salt_pepper noise).bmp")
image_2 = cv2.imread("image/Copy of Lena-Gaussian-noise2.jpg")

re_image_1_average = cv2.blur(image_1,(5,5))
re_image_2_average = cv2.blur(image_2,(5,5))
re_image_1_median = cv2.medianBlur(image_1, 5)
re_image_2_median = cv2.medianBlur(image_2, 5)

image_1_merge = np.hstack((image_1, re_image_1_average, re_image_1_median))
image_2_merge = np.hstack((image_2, re_image_2_average, re_image_2_median))

cv2.imshow("Original/ Average/ Median", image_1_merge)
cv2.imshow("Original_2/ Average2/ Median2", image_2_merge)

#분산도 변경
for sigma in (2, 4, 7, 10):
    re_image_1_gaussian = cv2.GaussianBlur(image_1, (5,5), sigma)
    re_image_2_gaussian = cv2.GaussianBlur(image_2, (5,5), sigma)

    desc = "Sigma : {} ".format(sigma)
    cv2.putText(re_image_1_gaussian, desc, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
    cv2.putText(re_image_2_gaussian, desc, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
    
    cv2.imshow("Gaussian Filter 1", re_image_1_gaussian)
    cv2.imshow("Gaussian Filter 2", re_image_2_gaussian)
    
    cv2.waitKey()
cv2.destroyAllWindows()