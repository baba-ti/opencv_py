import cv2 
import numpy as np
import time

img_1 = cv2.imread("image/5.jpg")
gray_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
img_2 = cv2.imread("image/6.jpg")
gray_2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)

sift_1 = cv2.SIFT_create(
sift_2 = cv2.SIFT_create()

keypoints_1, descriptors_1 = sift_1.detectAndCompute(gray_1, None)
keypoints_2, descriptors_2 = sift_2.detectAndCompute(gray_2, None)
print('keypoint_1:',len(keypoints_1), 'descriptors_1:', descriptors_1.shape)
print('keypoint_2:',len(keypoints_2), 'descriptors_2:', descriptors_2.shape)

start = time.time()
flann_matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
knn_match = flann_matcher.knnMatch(descriptors_1,descriptors_2,2)

T = 0.7
good_match=[]
for nearest1, nearest2 in knn_match:
    if(nearest1.distance/nearest2.distance)<T:
        good_match.append(nearest1)
print("매칭에 걸린 시간:",time.time()-start)

img_match = np.empty((max(img_1.shape[0],img_2.shape[0]),img_1.shape[1]+img_2.shape[1],3),dtype=np.uint8)
cv2.drawMatches(img_1,keypoints_1,img_2,keypoints_2,good_match,img_match,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Matches_FLANN",img_match)
cv2.imwrite("result/FLANN_match_1.png", img_match)
cv2.waitKey(0)
cv2.destroyAllWindows()

