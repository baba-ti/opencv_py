import cv2
import sys
import numpy as np

image = cv2.imread("image/cat.bmp")
if image is None:
    sys.exit("파일을 찾을 수 없습니다.")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(hsv)

cv2.imshow("HSV_image",hsv)
cv2.imshow("H_image",h)
cv2.imshow("S_image",s)
cv2.imshow("V_image",v)

print("H channel pixel:\n",h)
print("S channel pixel:\n",s)
print("V channel pixel:\n",v)

cv2.waitKey()
cv2.destroyWindow()