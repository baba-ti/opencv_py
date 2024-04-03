import cv2
import dlib

image = cv2.imread("image/people.jpg")

image_re = cv2.resize(image,(640,480))
hog_face_detector = dlib.get_frontal_face_detector() 
face_detection = hog_face_detector(image_re,2)

print(face_detection)

for face_detection in face_detection:
    left, top, right, bottom = face_detection.left(), face_detection.top(), face_detection.right(), face_detection.bottom()
    cv2.rectangle(image_re,(left, top), (right,bottom), (0,255,0),2)


cv2.imshow("show", image_re)    #선글라스 낀사람은 추출 실패

cv2.waitKey()
cv2.destroyAllWindows()
