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

#가우시안 필터 3 by 3 분산도 1
ksize = 3
image_1_gau = cv2.GaussianBlur(image_1_gray, (ksize, ksize), 1.0)
image_2_gau = cv2.GaussianBlur(image_2_gray, (ksize, ksize), 1.0)
image_3_gau = cv2.GaussianBlur(image_3_gray, (ksize, ksize), 1.0)

desc = "Mean : {}x{},".format(ksize,ksize)
cv2.putText(image_1_gau, desc, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
cv2.putText(image_2_gau, desc, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
cv2.putText(image_3_gau, desc, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)

cv2.imshow("image_1_gau", image_1_gau)
cv2.imshow("image_2_gau", image_2_gau)
cv2.imshow("image_3_gau", image_3_gau)

cv2.waitKey()
cv2.destroyAllWindows()

#히스토그램 스트레칭
image_1_gau_str = cv2.normalize(image_1_gau,None,0,255,cv2.NORM_MINMAX) 
image_2_gau_str = cv2.normalize(image_2_gau,None,0,255,cv2.NORM_MINMAX)
image_3_gau_str = cv2.normalize(image_3_gau,None,0,255,cv2.NORM_MINMAX)

cv2.imshow("image_1_gau_str", image_1_gau_str)
cv2.imshow("image_2_gau_str", image_2_gau_str)
cv2.imshow("image_3_gau_str", image_3_gau_str)

cv2.waitKey()
cv2.destroyAllWindows()

#케니 엣지
image_1_gau_str_canny = cv2.Canny(image_1_gau_str,100,200)
image_2_gau_str_canny = cv2.Canny(image_2_gau_str,200,300) 
image_3_gau_str_canny = cv2.Canny(image_3_gau_str,100,200) 

cv2.imshow("image_1_gau_str_canny",image_1_gau_str_canny)
cv2.imshow("image_2_gau_str_canny",image_2_gau_str_canny)
cv2.imshow("image_3_gau_str_canny",image_3_gau_str_canny)

cv2.imwrite("mid_project_result/image_1_gau_str_canny.jpg", image_1_gau_str_canny)
cv2.imwrite("mid_project_result/image_2_gau_str_canny.jpg", image_2_gau_str_canny)
cv2.imwrite("mid_project_result/image_3_gau_str_canny.jpg", image_3_gau_str_canny)

cv2.waitKey()
cv2.destroyAllWindows() 


