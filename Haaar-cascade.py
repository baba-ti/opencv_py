import cv2 

image = cv2.imread("image/people.jpg")

h1, w1, c1 =image.shape     #사진의 가로, 세로, 색 채널 검출
print("im height=", h1,"width=", w1,"channl=",c1)
image_re = cv2.resize(image,(800,600))   #사진 크기 재설정
h2, w2, c2 = image_re.shape                 
print("im height=", h2,"width=", w2,"channl=",c2)

casecade_face_detector = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml") #사전 훈련된 얼굴 검출기 데이터
face_detections = casecade_face_detector.detectMultiScale(image_re, scaleFactor=1.07, minNeighbors=4) #scalefactor 기본값 1.1 최소값 > 1

casecade_eye_detecter = cv2.CascadeClassifier("data/haarcascade_eye.xml") #사전 훈련된 눈 검출기 데이터
eye_detections = casecade_eye_detecter.detectMultiScale(image_re, scaleFactor=1.073, minNeighbors=6, minSize=(10,10), maxSize=(30,30)) #눈 인식

for(x,y,w,h) in face_detections:
    cv2.rectangle(image_re,(x,y),(x+w, y+h),(0,255,0),2)

for(x,y,w,h) in eye_detections:
    cv2.rectangle(image_re,(x,y),(x+w,y+h),(255,255,255),2)

print("얼굴 좌표:\n{}".format(face_detections)) #1열: x좌표 2열: y좌표 3열: 경계박스 너비 4열 경계박스 높이
print("--------------------------------")
print("눈좌표:\n{}".format(eye_detections))


cv2.imshow("show",image_re)
cv2.waitKey()
cv2.destroyAllWindows()


#1.13값 5명 나옴
#sacleFactor 큰 얼굴은 작제 작은 얼굴은 크게 조정 후 감지 기본값 1.1
#miNeighbors 파라미터는 최종 경계 박스를 선택하기 위해 얼굴 주변에 존재해야 하는 최소 후보 경계 박스 개수 ex)5라면 최소 5개의 경계 후보가 있어야함
