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


#평활화
image_1_mid_equ = cv2.equalizeHist(image_1_mid)
image_2_mid_equ = cv2.equalizeHist(image_2_mid)
image_3_mid_equ = cv2.equalizeHist(image_3_mid)

cv2.imshow("image_1_mid_equ", image_1_mid_equ)
cv2.imshow("image_2_mid_equ", image_2_mid_equ)
cv2.imshow("image_3_mid_equ", image_3_mid_equ)

# plt.figure("image 1")
# img1_hist1 = cv2.calcHist(image_1_gray,[0],None,[256],[0,256])
# plt.subplot(2,1,1), plt.plot(img1_hist1)
# img1_hist2 = cv2.calcHist(image_1_equalize,[0],None,[256],[0,256])
# plt.subplot(2,1,2), plt.plot(img1_hist2)

# plt.figure("image 2")
# img2_hist1 = cv2.calcHist(image_2_gray,[0],None,[256],[0,256])
# plt.subplot(2,1,1), plt.plot(img2_hist1)
# img2_hist2 = cv2.calcHist(image_2_equalize,[0],None,[256],[0,256])
# plt.subplot(2,1,2), plt.plot(img2_hist2)

# plt.show()
cv2.waitKey()
cv2.destroyAllWindows()

#케니 엣지
image_1_mid_equ_canny = cv2.Canny(image_1_mid_equ,100,200)
image_2_mid_equ_canny = cv2.Canny(image_2_mid_equ,200,300) 
image_3_mid_equ_canny = cv2.Canny(image_3_mid_equ,100,200) 

cv2.imshow("image_1_mid_equ_canny",image_1_mid_equ_canny)
cv2.imshow("image_2_mid_equ_canny",image_2_mid_equ_canny)
cv2.imshow("image_3_mid_equ_canny",image_3_mid_equ_canny)

cv2.imwrite("mid_project_result/image_1_mid_equ_canny.jpg", image_1_mid_equ_canny)
cv2.imwrite("mid_project_result/image_2_mid_equ_canny.jpg", image_2_mid_equ_canny)
cv2.imwrite("mid_project_result/image_3_mid_equ_canny.jpg", image_3_mid_equ_canny)

cv2.waitKey()
cv2.destroyAllWindows() 


