import cv2 as cv

img = cv.imread('Photos/lady.jpg')
# cv.imshow('Group of 5 people', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray People', gray)

# ! Face detect classifier
haar_cascade = cv.CascadeClassifier('haar_face.xml')
# ! Eye detect classifier
# haar_cascade = cv.CascadeClassifier('eye_tree_eye_glass.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
print(f'Number of faces found = {len(faces_rect)}')
for (x,y,w,h) in faces_rect:
  cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)
cv.waitKey(0)
