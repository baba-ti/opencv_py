import cv2
import numpy as np

img1 = cv2.imread("image/star.bmp") #이미지 읽어오기
img2 = cv2.imread("image/butterfly.bmp")
img3 = cv2.imread("image/camera.bmp")

h1, w1, c1 =img1.shape #크기 읽어오는 함수
h2, w2, c2 =img2.shape
h3, w3, c3 =img3.shape

print("im1 height=", h1,"width=", w1,"channl=",c1)
print("im1 height=", h2,"width=", w2,"channl=",c2)
print("im1 height=", h3,"width=", w3,"channl=",c3)


print(type(img1))   #변수 타입
print(img1)         #각픽셀의 정보를 담은 배열/행열 값

print(type(img2))
print(img2)

print(type(img3))
print(img3)

#ex) blue 0 gree 1 red 2
print("brg 값 (100,100)번쨰 픽셀 값 =", img1[100,100])
print("blue 값={}, gree값={}, red값={}".format(img2[100,100,0],img2[100,100,1],img2[100,100,2]))


cv2.imshow("show1",img1)    #이미지 보여주기
cv2.imshow("show2",img2)
cv2.imshow("show3",img3)
cv2.waitKey()
cv2.destroyAllWindows()






