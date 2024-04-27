import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt


image_1 = cv2.imread("image/fce(salt_pepper noise).bmp")
image_2 = cv2.imread("image/Copy of Lena-Gaussian-noise2.jpg")
image_3 = cv2.imread("image/300px-Kodim17_noisy.jpg")

image_1_gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
image_2_gray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
image_3_gray = cv2.cvtColor(image_3, cv2.COLOR_BGR2GRAY)


cv2.imshow("Original 1", image_1_gray)
cv2.imshow("Original 2", image_2_gray)
cv2.imshow("Original 3", image_3_gray)

cv2.waitKey()
cv2.destroyAllWindows()

# 중간값 필터 3 by 3 mid 필터
ksize = 3
image_1_mid = cv2.medianBlur(image_1_gray, ksize)
image_2_mid = cv2.medianBlur(image_2_gray, ksize)
image_3_mid = cv2.medianBlur(image_3_gray, ksize)


desc = "Mean : {}x{}".format(ksize,ksize)
cv2.putText(image_1_mid, desc, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
cv2.putText(image_2_mid, desc, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
cv2.putText(image_3_mid, desc, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)


cv2.imshow("image_1_mid", image_1_mid)
cv2.imshow("image_2_mid", image_2_mid)
cv2.imshow("image_3_mid", image_3_mid)

cv2.waitKey()
cv2.destroyAllWindows()


#히스토그램 스트레칭
image_1_mid_str = cv2.normalize(image_1_mid,None,0,255,cv2.NORM_MINMAX) 
image_2_mid_str = cv2.normalize(image_2_mid,None,0,255,cv2.NORM_MINMAX)
image_3_mid_str = cv2.normalize(image_3_mid,None,0,255,cv2.NORM_MINMAX)

# hist1 = cv2.calcHist([image_1_mid],[0],None,[256],[0,256])
# hist2 = cv2.calcHist([image_1_mid_str],[0],None,[256],[0,256])
# hist3 = cv2.calcHist([image_2_mid],[0],None,[256],[0,256])
# hist4 = cv2.calcHist([image_2_mid_str],[0],None,[256],[0,256])
# hist5 = cv2.calcHist([image_3_mid],[0],None,[256],[0,256])
# hist6 = cv2.calcHist([image_3_mid_str],[0],None,[256],[0,256])

cv2.imshow("image_1_mid_str", image_1_mid_str)
cv2.imshow("image_2_mid_str", image_2_mid_str)
cv2.imshow("image_3_mid_str", image_3_mid_str)

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

#케니 엣지
image_1_mid_str_canny = cv2.Canny(image_1_mid_str,100,200)
image_2_mid_str_canny = cv2.Canny(image_2_mid_str,200,300) 
image_3_mid_str_canny = cv2.Canny(image_3_mid_str,100,200) 

cv2.imshow("image_1_mid_str_canny",image_1_mid_str_canny)
cv2.imshow("image_2_mid_str_canny",image_2_mid_str_canny)
cv2.imshow("image_3_mid_str_canny",image_3_mid_str_canny)


cv2.imwrite("mid_project_result/image_1_mid_str_canny.jpg", image_1_mid_str_canny)
cv2.imwrite("mid_project_result/image_2_mid_str_canny.jpg", image_2_mid_str_canny)
cv2.imwrite("mid_project_result/image_3_mid_str_canny.jpg", image_3_mid_str_canny)

cv2.waitKey()
cv2.destroyAllWindows() 


