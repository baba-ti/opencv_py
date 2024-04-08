import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt


image_1 = cv2.imread("image/cat.bmp")
image_2 = cv2.imread("image/airplane.bmp")

if image_1 is None and image_2 is None:
    sys.exit("파일을 찾을 수 없습니다.")

image_1_gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
image_2_gray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
cv2.imshow("image 1 gray", image_1_gray)
cv2.imshow("image 2 gray", image_2_gray)

image_normalize = cv2.normalize(image_1_gray,None,0,255,cv2.NORM_MINMAX) 
image_normalize2 = cv2.normalize(image_2_gray,None,0,255,cv2.NORM_MINMAX)

hist1 = cv2.calcHist([image_1_gray],[0],None,[256],[0,256])
hist2 = cv2.calcHist([image_normalize],[0],None,[256],[0,256])
hist3 = cv2.calcHist([image_2_gray],[0],None,[256],[0,256])
hist4 = cv2.calcHist([image_normalize2],[0],None,[256],[0,256])


cv2.imshow("image 1 normalize", image_normalize)
cv2.imshow("image 2 normalize", image_normalize2)

plt.figure(1)
plt.plot(hist1, label = "image 1 gray")
plt.plot(hist2, label = "normalize")
plt.legend(loc=2)

plt.figure(2)
plt.plot(hist3, label = "image 2 gray")
plt.plot(hist4, label = "normalize")
plt.legend(loc=2)

plt.show()
cv2.waitKey()
cv2.destroyAllWindows()
