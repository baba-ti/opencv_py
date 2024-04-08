import cv2 
import numpy as np
import sys
import matplotlib.pyplot as plt

image_1 = cv2.imread("image/cat.bmp")
image_2 = cv2.imread("image/Copy of Lena-Gaussian-noise2.jpg")

image_1_gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)

canny_image_1 = cv2.Canny(image_1_gray,100,200)
canny_image_2 = cv2.Canny(image_2,100,200)

cv2.imshow("Original",image_1_gray)
cv2.imshow("Canny",canny_image_1)

cv2.imshow("Original_2",image_2)
cv2.imshow("Canny_2",canny_image_2)
cv2.waitKey()
cv2.destroyAllWindows()

laplacian_image_1 = cv2.Laplacian(image_1_gray,-1)
laplacian_image_2 = cv2.Laplacian(image_2,-1)

cv2.imshow("Original",image_1_gray)
cv2.imshow("laplasian",laplacian_image_1)

cv2.imshow("Original_2",image_2)
cv2.imshow("laplasian_2",laplacian_image_2)

cv2.waitKey()
cv2.destroyAllWindows()

LoG_gaussian = cv2.GaussianBlur(image_1_gray,(3,3),1)
LoG_laplacian = cv2.Laplacian(LoG_gaussian,-1,1)
LoG_gaussian_2 = cv2.GaussianBlur(image_2,(3,3),1)
LoG_laplacian_2 = cv2.Laplacian(LoG_gaussian_2,-1,1)

LoG_image_1 = LoG_laplacian/LoG_laplacian.max()
LoG_image_2 = LoG_laplacian_2/LoG_laplacian_2.max()

cv2.imshow("Original",image_1_gray)
cv2.imshow("LOG ",LoG_image_1)

cv2.imshow("Original_2",image_2)
cv2.imshow("LOG_2",LoG_image_2)
cv2.waitKey()
cv2.destroyAllWindows()
