import cv2
import numpy as py
import sys

image_1 = cv2.imread("image/a.bmp")
image_2 = cv2.imread("image/abcdef.bmp")
if image_1 is None and image_2 is None:
    sys.exit("파일을 찾을 수 없습니다.")

cv2.imshow("image 1", image_1)
cv2.imshow("image 2", image_2)

bitwis_and = cv2.bitwise_and(image_1, image_2)
bitwis_or = cv2.bitwise_or(image_1, image_2)
bitwis_xor = cv2.bitwise_xor(image_1, image_2)

cv2.imshow("AND", bitwis_and)
cv2.imshow("or",bitwis_or)
cv2.imshow("xor",bitwis_xor)

cv2.waitKey()
cv2.destroyAllWindows()