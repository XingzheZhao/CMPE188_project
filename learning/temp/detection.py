import cv2

img = cv2.imread('../Photos/lady.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Haar Cascades Classifier
face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Histogram of Oriented Gradients (HOG)
import face_recognition
import cv2

image = face_recognition.load_image_file('../Photos/lady.jpg')
face_locations = face_recognition.face_locations(image, model='hog')
for (top,right,bottom,left), landmarks in zip(face_locations,face_landmarks):
    cv2.rectangle(image,(left,bottom),(right,top),(255,0,0),2)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Multi-task Cascaded Convolutional Networks (MTCNN)
from mtcnn.mtcnn import MTCNN
import face_recognition
import cv2

image = face_recognition.load_image_file('../Photos/lady.jpg')
detector = MTCNN()
face_locations = detector.detect_faces(image)
for face in zip(face_locations):
    (x, y, w, h) = face[0]['box']
    landmarks = face[0]['keypoints']
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    for key, point in landmarks.items():
        cv2.circle(image, point, 2, (255, 0, 0), 6)

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
