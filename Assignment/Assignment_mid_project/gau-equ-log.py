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

#평활화
image_1_gau_equ = cv2.equalizeHist(image_1_gau)
image_2_gau_equ = cv2.equalizeHist(image_2_gau)
image_3_gau_equ = cv2.equalizeHist(image_3_gau)

cv2.imshow("image_1_gau_equ", image_1_gau_equ)
cv2.imshow("image_2_gau_equ", image_2_gau_equ)
cv2.imshow("image_3_gau_equ", image_3_gau_equ)

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

#LOG 에지 검출
image_1_gau_equ_gau = cv2.GaussianBlur(image_1_gau_equ,(3,3),1)
image_1_gau_equ_gau_lap = cv2.Laplacian(image_1_gau_equ_gau,-1,1)
image_2_gau_equ_gau = cv2.GaussianBlur(image_2_gau_equ,(3,3),1)
image_2_gau_equ_gau_lap = cv2.Laplacian(image_2_gau_equ_gau,-1,1)
image_3_gau_equ_gau = cv2.GaussianBlur(image_3_gau_equ,(3,3),1)
image_3_gau_equ_gau_lap = cv2.Laplacian(image_3_gau_equ_gau,-1,1)

image_1_gau_equ_gau_lap_final = image_1_gau_equ_gau_lap/image_1_gau_equ_gau_lap.max()
image_2_gau_equ_gau_lap_final = image_2_gau_equ_gau_lap/image_2_gau_equ_gau_lap.max()
image_3_gau_equ_gau_lap_final = image_3_gau_equ_gau_lap/image_3_gau_equ_gau_lap.max()

cv2.imshow("image_1_gau_equ_log",image_1_gau_equ_gau_lap_final)
cv2.imshow("image_2_gau_equ_log",image_2_gau_equ_gau_lap_final)
cv2.imshow("image_3_gau_equ_log",image_3_gau_equ_gau_lap_final)
cv2.waitKey()
cv2.destroyAllWindows()

