import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

img_1 = cv2.imread('image_set/skates1.png')
img_2 = cv2.imread('image_set/skates2.png')

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img_1, None)
kp2, des2 = sift.detectAndCompute(img_2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
         good_matches.append(m)
    
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
# Fundamental Matrix 계산
fundamental_matrix, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC)

print("Fundamental Matrix:")
print(fundamental_matrix)