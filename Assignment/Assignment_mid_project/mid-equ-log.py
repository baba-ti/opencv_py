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

#LOG 에지 검출
image_1_mid_equ_gau = cv2.GaussianBlur(image_1_mid_equ,(3,3),1)
image_1_mid_equ_gau_lap = cv2.Laplacian(image_1_mid_equ_gau,-1,1)
image_2_mid_equ_gau = cv2.GaussianBlur(image_2_mid_equ,(3,3),1)
image_2_mid_equ_gau_lap = cv2.Laplacian(image_2_mid_equ_gau,-1,1)
image_3_mid_equ_gau = cv2.GaussianBlur(image_3_mid_equ,(3,3),1)
image_3_mid_equ_gau_lap = cv2.Laplacian(image_3_mid_equ_gau,-1,1)

image_1_mid_equ_gau_lap_final = image_1_mid_equ_gau_lap/image_1_mid_equ_gau_lap.max()
image_2_mid_equ_gau_lap_final = image_2_mid_equ_gau_lap/image_2_mid_equ_gau_lap.max()
image_3_mid_equ_gau_lap_final = image_3_mid_equ_gau_lap/image_3_mid_equ_gau_lap.max()

cv2.imshow("image_1_mid_equ_log",image_1_mid_equ_gau_lap_final)
cv2.imshow("image_2_mid_equ_log",image_2_mid_equ_gau_lap_final)
cv2.imshow("image_3_mid_equ_log",image_3_mid_equ_gau_lap_final)

# LOG 에지 검출 후 저장 전에 수정 -> 정규화 하여 0,1의 값으로만 나와 이미지 저장시 다 0으로 변환
image_1_mid_equ_gau_lap_final = np.uint8(255 * image_1_mid_equ_gau_lap_final)
image_2_mid_equ_gau_lap_final = np.uint8(255 * image_2_mid_equ_gau_lap_final)
image_3_mid_equ_gau_lap_final = np.uint8(255 * image_3_mid_equ_gau_lap_final)

cv2.imwrite("mid_project_result/image_1_mid_equ_log.jpg", image_1_mid_equ_gau_lap_final)
cv2.imwrite("mid_project_result/image_2_mid_equ_log.jpg", image_2_mid_equ_gau_lap_final)
cv2.imwrite("mid_project_result/image_3_mid_equ_log.jpg", image_3_mid_equ_gau_lap_final)


cv2.waitKey()
cv2.destroyAllWindows()


