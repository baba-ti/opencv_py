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

# 평균값 필터 3 by 3 avr 필터
ksize = 3
image_1_avr = cv2.blur(image_1_gray,(ksize,ksize))
image_2_avr = cv2.blur(image_2_gray,(ksize,ksize))
image_3_avr = cv2.blur(image_3_gray,(ksize,ksize))
    
text = "Mean : {} * {}".format(ksize,ksize)
    
cv2.putText(image_1_avr, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
cv2.putText(image_2_avr, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
cv2.putText(image_3_avr, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)
    
cv2.imshow("image_1_avr", image_1_avr)
cv2.imshow("image_2_avr", image_2_avr)
cv2.imshow("image_3_avr", image_3_avr)

cv2.waitKey()
cv2.destroyAllWindows()

#히스토그램 스트레칭
image_1_avr_str = cv2.normalize(image_1_avr,None,0,255,cv2.NORM_MINMAX) 
image_2_avr_str = cv2.normalize(image_2_avr,None,0,255,cv2.NORM_MINMAX)
image_3_avr_str = cv2.normalize(image_3_avr,None,0,255,cv2.NORM_MINMAX)

# hist1 = cv2.calcHist([image_1_avr],[0],None,[256],[0,256])
# hist2 = cv2.calcHist([image_1_avr_str],[0],None,[256],[0,256])
# hist3 = cv2.calcHist([image_2_avr],[0],None,[256],[0,256])
# hist4 = cv2.calcHist([image_2_avr_str],[0],None,[256],[0,256])
# hist5 = cv2.calcHist([image_3_avr],[0],None,[256],[0,256])
# hist6 = cv2.calcHist([image_3_avr_str],[0],None,[256],[0,256])

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

#LOG 에지 검출
image_1_avr_str_gau = cv2.GaussianBlur(image_1_avr_str,(3,3),1)
image_1_avr_str_gau_lap = cv2.Laplacian(image_1_avr_str_gau,-1,1)
image_2_avr_str_gau = cv2.GaussianBlur(image_2_avr_str,(3,3),1)
image_2_avr_str_gau_lap = cv2.Laplacian(image_2_avr_str_gau,-1,1)
image_3_avr_str_gau = cv2.GaussianBlur(image_3_avr_str,(3,3),1)
image_3_avr_str_gau_lap = cv2.Laplacian(image_3_avr_str_gau,-1,1)

image_1_avr_str_gau_lap_final = image_1_avr_str_gau_lap/image_1_avr_str_gau_lap.max()
image_2_avr_str_gau_lap_final = image_2_avr_str_gau_lap/image_2_avr_str_gau_lap.max()
image_3_avr_str_gau_lap_final = image_3_avr_str_gau_lap/image_3_avr_str_gau_lap.max()

cv2.imshow("image_1_avr_str_log",image_1_avr_str_gau_lap_final)
cv2.imshow("image_2_avr_str_log",image_2_avr_str_gau_lap_final)
cv2.imshow("image_3_avr_str_log",image_3_avr_str_gau_lap_final)

# LOG 에지 검출 후 저장 전에 수정 -> 정규화 하여 0,1의 값으로만 나와 이미지 저장시 다 0으로 변환
image_1_avr_str_gau_lap_final = np.uint8(255 * image_1_avr_str_gau_lap_final)
image_2_avr_str_gau_lap_final = np.uint8(255 * image_2_avr_str_gau_lap_final)
image_3_avr_str_gau_lap_final = np.uint8(255 * image_3_avr_str_gau_lap_final)

cv2.imwrite("mid_project_result/image_1_avr_str_log.jpg", image_1_avr_str_gau_lap_final)
cv2.imwrite("mid_project_result/image_2_avr_str_log.jpg", image_2_avr_str_gau_lap_final)
cv2.imwrite("mid_project_result/image_3_avr_str_log.jpg", image_3_avr_str_gau_lap_final)
    
cv2.waitKey()
cv2.destroyAllWindows()

