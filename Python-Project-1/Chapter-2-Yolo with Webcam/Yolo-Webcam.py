from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0) #for webcam
cap.set(3,1280)
cap.set(4, 720)

while(True):
    success, img = cap.read()

    cv2.imshow('Image',img)
    if cv2.waitKey(1) & 0xFF==('a'):
        break
cv2.destroyAllWindows()