import cv2  
import numpy as np
import sys

image = cv2.imread("image/cat.bmp")
if image is None:
    sys.exit("파일을 찾을 수 없습니다.")

cv2.imshow("original_RGB",image)

print("original pixel(R,G,B):\n",image)

cv2.imshow("R channel", image[:,:,2])
cv2.imshow("G channel", image[:,:,1])
cv2.imshow("B channel", image[:,:,0])

print("R channel pixel:\n",image[:,:,2])
print("G channel pixel:\n",image[:,:,1])
print("B channel pixel:\n",image[:,:,0])

cv2.waitKey()
cv2.destroyAllWindows()




# for channel in range(3):
#     tmp = np.zeros(image.shape, dtype=np.uint8)
#     tmp[:,:,channel] = image[:,:,channel]
#     cv2.imshow("channel_"+str(channel),tmp)

# for channel in range(image.shape[2]):
#     print("channel", channel)
#     channel_values=image[:,:,channel]
#     print(channel_values)
