import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

image_1 = cv2.imread("image/fce(salt_pepper noise).bmp")
image_2 = cv2.imread("image/Copy of Lena-Gaussian-noise2.jpg")
image_3 = cv2.imread("image/300px-Kodim17_noisy.jpg")

cv2.imshow("Original 1", image_1)
cv2.imshow("Original 2", image_2)
cv2.imshow("Original 3", image_3)

cv2.waitKey()
cv2.destroyAllWindows()

# 평균값 필터 3 by 3
ksize = 3
image_1_AverageFilter = cv2.blur(image_1,(ksize,ksize))
image_2_AverageFilter = cv2.blur(image_2,(ksize,ksize))
image_3_AverageFilter = cv2.blur(image_3,(ksize,ksize))
    
text = "Mean : {} * {}".format(ksize,ksize)
    
cv2.putText(image_1_AverageFilter, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
cv2.putText(image_2_AverageFilter, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
cv2.putText(image_3_AverageFilter, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
    
cv2.imshow("Average Filter 1", image_1_AverageFilter)
cv2.imshow("Average Filter 2", image_2_AverageFilter)
cv2.imshow("Average Filter 3", image_3_AverageFilter)

cv2.waitKey()
cv2.destroyAllWindows()

# 중간값 필터 3 by 3
image_1_MedianFilter = cv2.medianBlur(image_1, ksize)
image_2_MedianFilter = cv2.medianBlur(image_2, ksize)
image_3_MedianFilter = cv2.medianBlur(image_3, ksize)


desc = "Mean : {}x{}".format(ksize,ksize)
cv2.putText(image_1_MedianFilter, desc, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
cv2.putText(image_2_MedianFilter, desc, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
cv2.putText(image_3_MedianFilter, desc, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)


cv2.imshow("Median Filter 1", image_1_MedianFilter)
cv2.imshow("Median Filter 2", image_2_MedianFilter)
cv2.imshow("Median Filter 3", image_3_MedianFilter)

cv2.waitKey()
cv2.destroyAllWindows()

#가우시안 필터 3 by 3 분산도 1
image_1_GaussianFiltering = cv2.GaussianBlur(image_1, (ksize, ksize), 1.0)
image_2_GaussianFiltering = cv2.GaussianBlur(image_2, (ksize, ksize), 1.0)
image_3_GaussianFiltering = cv2.GaussianBlur(image_3, (ksize, ksize), 1.0)

desc = "Mean : {}x{},".format(ksize,ksize)
cv2.putText(image_1_GaussianFiltering, desc, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
cv2.putText(image_2_GaussianFiltering, desc, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
cv2.putText(image_3_GaussianFiltering, desc, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)

cv2.imshow("Gaussian Filter 1", image_1_GaussianFiltering)
cv2.imshow("Gaussian Filter 2", image_2_GaussianFiltering)
cv2.imshow("Gaussian Filter 3", image_3_GaussianFiltering)

cv2.waitKey()
cv2.destroyAllWindows()

