import cv2
import numpy as np
import sys

image = cv2.imread("image/cat.bmp")
if image is None:
    sys.exit("파일을 찾을 수 없습니다.")

gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Gray_scale", gray_scale)

print("Gray_scale pixel:\n",gray_scale)

cv2.waitKey()
cv2.destroyAllWindows()