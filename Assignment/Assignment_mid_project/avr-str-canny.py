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
image_1_avr = cv2.blur(image_1,(ksize,ksize))
image_2_avr = cv2.blur(image_2,(ksize,ksize))
image_3_avr = cv2.blur(image_3,(ksize,ksize))
    
text = "Mean : {} * {}".format(ksize,ksize)
    
cv2.putText(image_1_avr, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
cv2.putText(image_2_avr, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
cv2.putText(image_3_avr, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
    
cv2.imshow("image_1_avr", image_1_avr)
cv2.imshow("image_2_avr", image_2_avr)
cv2.imshow("image_3_avr", image_3_avr)

cv2.waitKey()
cv2.destroyAllWindows()

image_1_avr_str = cv2.normalize(image_1_avr,None,0,255,cv2.NORM_MINMAX) 
image_2_avr_str = cv2.normalize(image_2_avr,None,0,255,cv2.NORM_MINMAX)
image_3_avr_str = cv2.normalize(image_3_avr,None,0,255,cv2.NORM_MINMAX)

hist1 = cv2.calcHist([image_1_avr],[0],None,[256],[0,256])
hist2 = cv2.calcHist([image_1_avr_str],[0],None,[256],[0,256])
hist3 = cv2.calcHist([image_2_avr],[0],None,[256],[0,256])
hist4 = cv2.calcHist([image_2_avr_str],[0],None,[256],[0,256])
hist5 = cv2.calcHist([image_3_avr],[0],None,[256],[0,256])
hist6 = cv2.calcHist([image_3_avr_str],[0],None,[256],[0,256])

cv2.imshow("image_1_avr_str", image_1_avr_str)
cv2.imshow("image_2_avr_str", image_2_avr_str)
cv2.imshow("image_3_avr_str", image_3_avr_str)

# plt.figure(1)
# plt.plot(hist1, label = "image_1_avr")
# plt.plot(hist2, label = "image_1_avr_str")
# plt.legend(loc=2)

# plt.figure(2)
# plt.plot(hist3, label = "image_2_avr")
# plt.plot(hist4, label = "image_2_avr_str")
# plt.legend(loc=2)

# plt.figure(3)
# plt.plot(hist5, label = "image_3_avr")
# plt.plot(hist6, label = "image_3_avr_str")
# plt.legend(loc=2)
 
# plt.show()
cv2.waitKey()
cv2.destroyAllWindows()
