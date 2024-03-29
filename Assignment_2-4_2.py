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

image_1_equalize = cv2.equalizeHist(image_1_gray)
image_2_equalize = cv2.equalizeHist(image_2_gray)

cv2.imshow("image 1 equalize", image_1_equalize)
cv2.imshow("image 2 equalize", image_2_equalize)

plt.figure(1)
img1_hist1 = cv2.calcHist(image_1_gray,[0],None,[256],[0,256])
plt.subplot(2,1,1), plt.plot(img1_hist1)
img1_hist2 = cv2.calcHist(image_1_equalize,[0],None,[256],[0,256])
plt.subplot(2,1,2), plt.plot(img1_hist2)

plt.figure(2)
img2_hist1 = cv2.calcHist(image_2_gray,[0],None,[256],[0,256])
plt.subplot(2,1,1), plt.plot(img2_hist1)
img2_hist2 = cv2.calcHist(image_2_equalize,[0],None,[256],[0,256])
plt.subplot(2,1,2), plt.plot(img2_hist2)

plt.show()
cv2.waitKey()
cv2.destroyAllWindows()