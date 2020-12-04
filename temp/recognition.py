import face_recognition
import cv2
import time
from imutils import paths
import os
import numpy as np

knownFace = "known.jpg"
image = face_recognition.load_image_file(knownFace)
face_locations = face_recognition.face_locations(image, model="hog")
face_landmarks = face_recognition.face_landmarks(image)

# after alignment we have to resize the image so we have to give
# width and height of the output aligned face.
(top,right,bottom,left) = face_locations[0]
desiredWidth = (right-left)
desiredHeight = (bottom-top)

# use the code snippet for face alignment (from the previous section)
# and create a function alignFace. set the desiredWidth and desiredHeight
# to face width and height
align_f = alignFace(image, face_locations, face_landmarks, desiredWidth, desiredHeight)

# calculate face encodings of align face. It is array of 128 length.
known_face_encoding = face_recognition.face_encodings(align_f, num_jitters=10)[0]

unknownFace = "unknown.jpg"
image = face_recognition.load_image_file(unknownFace)
face_locations = face_recognition.face_locations(image, model="hog")
face_landmarks = face_recognition.face_landmarks(image)

# after alignment we have to resize the image so we have to give
# width and height of the output aligned face.
(top,right,bottom,left) = face_locations[0]
desiredWidth = (right-left)
desiredHeight = (bottom-top)

# use the code snippet for face alignment (from the previous section)
# and create a function alignFace. set the desiredWidth and desiredHeight
# to face width and height
align_f = alignFace(image, face_locations, face_landmarks, desiredWidth, desiredHeight)

# calculate face encodings of align face. It is array of 128 length.
unknown_face_encoding = face_recognition.face_encodings(align_f, num_jitters=10)[0]

# calculate the distance between known and unknown face.
# distance range is 0 to 1. If two faces are maching then value
# of distance is near to zeor else it is much away from zero or near
# to one.
distance = face_recognition.face_distance([known_face_encoding], unknown_face_encoding)[0]
print("Distance : {}".format(distance))