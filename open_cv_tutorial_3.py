import cv2
import numpy as np

# haarcascade_frontalface_default.xml
# this import a face classier
# face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

img2 = cv2.imread("ai_face.jpg")
img = cv2.imread("fake_ai_faces.png")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(imgGray, 1.1, 4)
eyes = eye_classifier.detectMultiScale(imgGray, 1.1, 50)
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 1)
for (x,y,w,h) in eyes:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)

cv2.imshow("basic_face", img)
cv2.waitKey()


'''# face tracking with video
video_cap = cv2.VideoCapture("video_test.mp4")  
while True:
    success, img = video_cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(imgGray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break'''