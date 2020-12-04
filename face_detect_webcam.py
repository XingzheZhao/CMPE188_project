import cv2
import pyautogui

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
screenW, screenH = pyautogui.size()
dispW = int(screenW/2)
dispH = int(screenH/2)

cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_NORMAL)
cv2.moveWindow('frame', 0, 0)

# Face detection function
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.2, minNeighbors=5) 
    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (0,255,0), 10)
    return face_img

print("Press q to quit\nStreaming..")
while True:
    # Capture frame
    ret, frame = cap.read()
    # Perform face detection
    frame = detect_face(frame)
    # Display the resulting frame
    cv2.resizeWindow('frame', dispW, dispH)
    cv2.imshow('frame', frame)
    # Quit if 'q' button is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
