# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 16:23:17 2020

@author: 503107711
"""


import cv2

cv2.startWindowThread()
cv2.namedWindow("preview",0)
video_capture=cv2.VideoCapture(0)
while True:
    _,frame=video_capture.read()
    face_classifier=cv2.CascadeClassifier(r'C:\Users\503107711\Desktop\Project for GIT\OpenCV\FACEPREDICT\Computer-Vision-Tutorial-master\Haarcascades\haarcascade_frontalface_default.xml')
    eye_classifier=cv2.CascadeClassifier(r'C:\Users\503107711\Desktop\Project for GIT\OpenCV\FACEPREDICT\Computer-Vision-Tutorial-master\Haarcascades\haarcascade_eye.xml')
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(grey,1.3,5)
    if faces in ():
        print('no face found')
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(127,0,255),2)
        roi_frame=frame[x:x+w,y:y+h]
        roi_grey=grey[x:x+w,y:y+h]
        eyes=eye_classifier.detectMultiScale(roi_grey,1.1,5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_frame,(ex,ey),(ex+ew,ey+eh),(127,22,232),2)
    cv2.imshow('preview',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
                                    
video_capture.release()
cv2.destroyAllWindows()
