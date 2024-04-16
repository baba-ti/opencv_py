import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

image_1 = cv2.imread("image/cat.bmp")
image_2 = cv2.imread("image/Copy of Lena-Gaussian-noise2.jpg")
image_1_gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)

#Roberts 엣지
roberts_horizontal = np.array([[-1,0,0],[0,1,0],[0,0,0]])
roberts_vertical = np.array([[0,0,-1],[0,1,0],[0,0,0]])
roberts_edge_horizontal = cv2.filter2D(image_1_gray, -1, roberts_horizontal)
roberts_edge_vertical = cv2.filter2D(image_1_gray, -1, roberts_vertical)
roberts_edge_horizontal_2 = cv2.filter2D(image_2, -1, roberts_horizontal)
roberts_edge_vertical_2 = cv2.filter2D(image_2, -1, roberts_vertical)
image_1_gray_merged_robert = np.hstack((image_1_gray, roberts_edge_horizontal + roberts_edge_vertical))
image_2_merged_robert = np.hstack((image_2, roberts_edge_horizontal_2 + roberts_edge_vertical_2))
 
#소벨 엣지는 라이브러리가 제공하기도 한다.
sobel_horizontal = np.array([[-1,-2,-1],[0,0,0],[1,2,1]]) 
sobel_vertical = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobel_edge_horizontal = cv2.filter2D(image_1_gray,-1,sobel_horizontal)
sobel_edge_vertical = cv2.filter2D(image_1_gray, -1 ,sobel_vertical)
sobel_edge_horizontal_2 = cv2.filter2D(image_2,-1,sobel_horizontal)
sobel_edge_vertical_2 = cv2.filter2D(image_2,-1,sobel_vertical)
image_1_merged_sobel = np.hstack((image_1_gray, sobel_edge_horizontal + sobel_edge_horizontal ))
image_2_merged_sobel = np.hstack((image_2, sobel_edge_horizontal_2 + sobel_edge_vertical_2 ))

# #Sobel 엣지 api이용 방법
# sobel_dx = cv2.Sobel(image_1_gray, -1, 1, 0, ksize=3)
# sobel_dy = cv2.Sobel(image_1_gray,-1,0,1,ksize=3)
# sobel_dx2 = cv2.Sobel(image_2, -1, 1, 0, ksize=3)
# sobel_dy2 = cv2.Sobel(image_2, -1, 1, 0, ksize=3)
# image_1_merged_sobel_api = np.hstack((image_1_gray, sobel_dx+sobel_dy))
# image_2_merged_sobel_api = np.hstack((image_2, sobel_dx2 + sobel_dy2))

# cv2.imshow("Sobel API",image_1_merged_sobel_api)
# cv2.imshow("Sobel API 2",image_2_merged_sobel_api)
# cv2.waitKey()
# cv2.destroyAllWindows()

#Prewitt 엣지
prewitt_horizontal = np.array([[-1,-1,-1],[0,1,0],[1,1,1]])
prewitt_vertical = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
prewitt_edge_horizontal = cv2.filter2D(image_1_gray,-1,prewitt_horizontal)
prewitt_edge_vertical = cv2.filter2D(image_1_gray, -1 ,prewitt_vertical)
prewitt_edge_horizontal_2 = cv2.filter2D(image_2,-1,prewitt_horizontal)
prewitt_edge_vertical_2 = cv2.filter2D(image_2,-1,prewitt_vertical)
image_1_merged_prewitt = np.hstack((image_1_gray, prewitt_edge_horizontal + prewitt_edge_vertical))
image_2_merged_prewitt = np.hstack((image_2, prewitt_edge_horizontal_2 + prewitt_edge_vertical_2))

cv2.imshow("Roberts edge ", image_1_gray_merged_robert) 
cv2.imshow("Roberts edge 2", image_2_merged_robert)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow("Sobel edge ", image_1_merged_sobel)
cv2.imshow("Sobel edge 2", image_2_merged_sobel)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow("Prewitt edge ", image_1_merged_prewitt)
cv2.imshow("Prewitt edge 2", image_2_merged_prewitt)
cv2.waitKey()
cv2.destroyAllWindows()
