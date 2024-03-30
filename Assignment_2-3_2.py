import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

image_1 = cv2.imread("image/cat.bmp")
image_2 = cv2.imread("image/airplane.bmp")

if image_1 is None and image_2 is None:
    sys.exit("파일을 찾을 수 없습니다.")

image_1_gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
image_2_gray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

cv2.imshow("original image 1", image_1)
cv2.imshow("original image 2",image_2)
cv2.imshow("image 1 gray", image_1_gray)
cv2.imshow("image 2 gray", image_2_gray)

ret, th1 = cv2.threshold(image_1_gray,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(image_1_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,2)
th3 = cv2.adaptiveThreshold(image_1_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,2)

ret_2, th1_2 = cv2.threshold(image_2_gray,127,255,cv2.THRESH_BINARY)
th2_2 = cv2.adaptiveThreshold(image_2_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,2)
th3_2 = cv2.adaptiveThreshold(image_2_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,2)

titles = ["Original","Global","Mean","Gaussian","Original","Global","Mean","Gaussian"]
images = [image_1_gray,th1,th2,th3,image_2_gray,th1_2,th2_2,th3_2]

for i in range(8):
    plt.subplot(2,4,i+1), plt.imshow(images[i],"gray")
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
    
plt.show() 
cv2.waitKey()
cv2.destroyAllWindows()
