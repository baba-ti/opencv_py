import cv2
import numpy as np

# 이미지 읽기
img_1 = cv2.imread('image/im0.png')
img_2 = cv2.imread('image/im1.png')
img_3 = cv2.imread('image/im2.png')

gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
gray_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY)

# SIFT 객체 생성
sift_1 = cv2.SIFT_create()
sift_2 = cv2.SIFT_create()
sift_3 = cv2.SIFT_create()

# 키포인트와 디스크립터 계산
keypoints_1, descriptors_1 = sift_1.detectAndCompute(gray_1, None)
keypoints_2, descriptors_2 = sift_2.detectAndCompute(gray_2, None)
keypoints_3, descriptors_3 = sift_3.detectAndCompute(gray_3, None)

print('keypoint_1:',len(keypoints_1), 'descriptors_1:', descriptors_1.shape)
print('keypoint_2:',len(keypoints_2), 'descriptors_2:', descriptors_2.shape)
print('keypoint_3:',len(keypoints_3), 'descriptors:_3', descriptors_3.shape)

# 키포인트 시각화
img_keypoints_1 = cv2.drawKeypoints(gray_1, keypoints_1, img_1)
img_keypoints_2 = cv2.drawKeypoints(gray_2, keypoints_2, img_2)
img_keypoints_3 = cv2.drawKeypoints(gray_3, keypoints_3, img_3)

# 결과 출력
cv2.imshow('SIFT_1', img_keypoints_1)
cv2.imwrite("result/SIFT_1.png", img_keypoints_1)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('SIFT_2', img_keypoints_2)
cv2.imwrite("result/SIFT_2.png", img_keypoints_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('SIFT_3', img_keypoints_3)
cv2.imwrite("result/SIFT_3.png", img_keypoints_3)
cv2.waitKey(0)
cv2.destroyAllWindows()

