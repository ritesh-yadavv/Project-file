from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0) #for webcam
cap.set(3,1280)
cap.set(4, 720)

model = YOLO("../Yolo-Weights/yolov8n.pt")

while(True):
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

            w, h = x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))

           # # Confidence
           #  conf = math.ceil((box.conf[0]*100))/100
           #  print(conf)
           #  cvzone.putTextRect(img, f'{conf}',(max(0,x1),max(35,y1)))
           #
           # # Class Name
           #  cls = int(box.cls[0])
           #
           #  cvzone.putTextRect(img,f'{className[cls]}{conf}',(max(0,x1),max(35,y1)) ,thickness=1)


    cv2.imshow('Image',img)
    if cv2.waitKey(1) & 0xFF==('a'):
        break
cv2.destroyAllWindows()