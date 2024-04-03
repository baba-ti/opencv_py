import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

image_1 = cv2.imread("image/cat.bmp")
image_2 = cv2.imread("image/airplane.bmp")

if image_1 is None and image_2 is None:
    sys.exit("파일을 찾을 수 없습니다.")

image_1_gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
image_2_gray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
cv2.imshow("original image 1", image_1), cv2.imshow("original image 2",image_2)
cv2.imshow("image 1 gray", image_1_gray), cv2.imshow("image 2 gray", image_2_gray)

 #(입력이미지, 지정입계값, 픽섹적용 최대값, 타입)
ret, thresh1 = cv2.threshold(image_1_gray, 127, 255, cv2.THRESH_BINARY) #임계값 넘으면 maxval로 지정, 못하면 0으로 지정
ret,thresh2 = cv2.threshold(image_1_gray, 127, 255, cv2.THRESH_BINARY_INV) #Binary 설정의 반대
ret, thresh3 = cv2.threshold(image_1_gray, 127, 255, cv2.THRESH_TRUNC) #임계값을 넘으면 maxval로 지정, 못하면 원래값 유지
ret, thresh4 = cv2.threshold(image_1_gray, 127, 255, cv2.THRESH_TOZERO) #임계값을 넘으면 값 유지, 못하면 0으로 지정
ret, thresh5 = cv2.threshold(image_1_gray, 127, 255, cv2.THRESH_TOZERO_INV) #Tozero 설정의 반대

titles = ["original","binary","binary_inv","trunc","tozero","tozero_inv"]
images = [image_1_gray,thresh1,thresh2,thresh3,thresh4,thresh5]

for i in range(6):
    plt.subplot(2, 3,i+1), plt.imshow(images[i],"gray")
    plt.title(titles[i])
    plt.xticks([]), plt.xticks([])

plt.show()
cv2.waitKey()
cv2.destroyAllWindows()


