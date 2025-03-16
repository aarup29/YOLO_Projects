import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO("yolov8n.pt")

video_path = "CarParking.mp4"
cap = cv2.VideoCapture(video_path)

drawing = False
no_parking_zone = [(0, 0), (0, 0)]
zone_selected = False

def draw_rectangle(event, p, q, flags, param):
    global drawing, no_parking_zone, zone_selected 
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        no_parking_zone[0] = (p, q)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            no_parking_zone[1] = (p, q)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        no_parking_zone[1] = (p, q)
        zone_selected = True

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", draw_rectangle)

while cap.isOpened():
    ret, frame = cap.read() 
    print(ret)
    if ret == False:
        break
    
    results = model(frame)
    cv2.rectangle(frame, (no_parking_zone[0][0],  no_parking_zone[0][1]), (no_parking_zone[1][0], no_parking_zone[1][1]), (0, 0, 255), 1)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            clss = int(box.cls[0].item())            
            
            if clss in [2, 3, 5, 7]:
                # cv2.rectangle(frame, (x1,  y1), (x2, y2), (0, 255, 0), 1)
                
                if (no_parking_zone[0][0] < x1 < no_parking_zone[1][0] and
                    no_parking_zone[0][1] < y1 < no_parking_zone[1][1]):
                    cv2.putText(frame, "WRONG PARKING", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()