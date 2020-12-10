import cv2
import pickle
import pyautogui
import numpy as np

from PIL import Image
from numpy import load
from numpy import expand_dims
from numpy import asarray
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from keras.models import load_model

# load the face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Face detection function
def detect_face(img):
    face_rects = face_cascade.detectMultiScale(img,scaleFactor=1.2, minNeighbors=5)
    return face_rects

# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return asarray(yhat[0])

# label encode targets
data = load('faces-embeddings.npz')
out_encoder = LabelEncoder()
out_encoder.fit(data['arr_1'])

# load the recognizer model from disk
filename = 'recognizer.sav'
recognizer = pickle.load(open(filename, 'rb'))

# load the facenet model
model = load_model('facenet_keras.h5')

# Connecting to video
cap = cv2.VideoCapture('test_vid.mkv')

# Grab width and height from video feed
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get screen resolution
screenW, screenH = pyautogui.size()

# Set display resolution
dispW = int(screenW/2)
dispH = int(screenH/2)

cv2.namedWindow('frame',cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_NORMAL)
cv2.moveWindow('frame',0,0)

required_size = (160,160)

print("Press q to quit\nStreaming..")
while True:
    
    # Capture frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Perform face detection
    rois = detect_face(frame)

    for face_rects in rois:    
        x,y,w,h = face_rects[0], face_rects[1], face_rects[2], face_rects[3]

        face = frame[y:y+h,x:x+w]

        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_pixels = asarray(image)
        face_emb = get_embedding(model, face_pixels)    

        # prediction for the face
        samples = expand_dims(face_emb, axis=0)
        yhat_class = recognizer.predict(samples)
        yhat_prob = recognizer.predict_proba(samples)
        
        # get name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.putText(frame, predict_names[0], (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)

    # Display the resulting frame
    cv2.resizeWindow('frame',dispW,dispH)
    cv2.imshow('frame',frame)
    
    # Quit if 'q' button is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
