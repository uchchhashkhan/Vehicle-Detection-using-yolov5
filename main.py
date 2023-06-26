import cv2
import torch
import numpy as np
from tracker import *


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap=cv2.VideoCapture('/home/asus/Desktop/Office/win11vehiclecount-main/highway.mp4')

count=0
tracker = Tracker()




def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)


area1=[(366,436),(346,457),(526,482),(524,449)]
area2=[(600,436),(619,472),(818,463),(785,440)]
area_1=set()
area_2=set()
while True:
    ret,frame=cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,600))
    
    results=model(frame)
    list=[]
    for index,rows in results.pandas().xyxy[0].iterrows():
        x=int(rows[0])
        y=int(rows[1])
        x1=int(rows[2])
        y1=int(rows[3])
        b=str(rows['name'])
        list.append([x,y,x1,y1])
    idx_bbox=tracker.update(list)
    for bbox in idx_bbox:
        x2,y2,x3,y3,id=bbox
        cv2.rectangle(frame,(x2,y2),(x3,y3),(0,0,225),2)
        cv2.putText(frame,str(id),(x2,y2),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
        cv2.circle(frame,(x3,y3),3,(0,225,0),-1) 
        result=cv2.pointPolygonTest(np.array(area1,np.int32),((x3,y3)),False)
        result1=cv2.pointPolygonTest(np.array(area2,np.int32),((x3,y3)),False)
        if result>0:
            area_1.add(id)

        if result1>0:
            area_2.add(id)


        

       
      
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,255,255),3)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,105,180),3)
    a1=len(area_1)
    cv2.putText(frame,('Coming : ')+str(a1),(36,36),cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),2)

    a2=len(area_2)
    cv2.putText(frame,('Going - ')+str(a2),(669,36),cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),2)


    

    cv2.imshow("FRAME",frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
