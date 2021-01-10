import cv2
import numpy as np
import urllib.request
import urllib.parse
import random

face_detect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0);
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer\\trainingData.yml")
ide=0
font=cv2.FONT_HERSHEY_COMPLEX_SMALL
while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_detect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img ,(x,y),(x+w,y+h),(0,255,0),2)
        ide,conf=rec.predict(gray[y:y+h,x:x+w])
        if(conf < 70):
            if(ide==7):
                ide="thej"
            elif(ide==8):
                ide="deep"
            elif(ide==9):
                ide="sai"
        else:
            ide="Unknown"
        cv2.putText(img,str(ide),(x,y+h),font,2,255,2);
    cv2.imshow("Face",img);
    if(cv2.waitKey(1)==ord('q')):
        break

cam.release()
cv2.destroyAllWindows()
