import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

img_1 = cv2.imread('image_set/skates1.png')
img_2 = cv2.imread('image_set/skates2.png')

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img_1, None)
kp2, des2 = sift.detectAndCompute(img_2, None)

# FLANN 매쳐 설정
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

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

# 매치 포인트 그리기
img_matches = cv2.drawMatches(img_1, kp1, img_2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 에피라인 그리기
h1, w1 = img_1.shape[:2]
h2, w2 = img_2.shape[:2]
pts = np.float32([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]]).reshape(-1, 1, 2)
lines1 = cv2.computeCorrespondEpilines(pts, 2, fundamental_matrix)
lines1 = lines1.reshape(-1, 3)

img1_with_lines = np.copy(img_1)
for line in lines1:
    x0, y0 = map(int, [0, -line[2]/line[1]])
    x1, y1 = map(int, [w1, -(line[2]+line[0]*w1)/line[1]])
    cv2.line(img1_with_lines, (x0, y0), (x1, y1), (0, 0, 255), 1)

img2_with_lines = np.copy(img_2)
lines2 = cv2.computeCorrespondEpilines(dst_pts, 1, fundamental_matrix)
lines2 = lines2.reshape(-1, 3)
for line in lines2:
    x0, y0 = map(int, [0, -line[2]/line[1]])
    x1, y1 = map(int, [w2, -(line[2]+line[0]*w2)/line[1]])
    cv2.line(img2_with_lines, (x0, y0), (x1, y1), (0, 0, 255), 1)

# 결과 이미지 출력
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(img_matches)
ax[0].set_title('Matched Keypoints')
ax[1].imshow(img1_with_lines)
ax[1].set_title('Image 1 with Epipolar Lines')
ax[2].imshow(img2_with_lines)
ax[2].set_title('Image 2 with Epipolar Lines')
plt.show()
