from __future__ import print_function
import cv2
import argparse
import time
import pafy
import youtube_dl
import numpy as np
import os
from os import path
if __name__ == '__main__':
    import sys
    sys.path.append(path.join(path.dirname(__file__), '..'))
    from resources import read


def tryload():
    #-- Info CPU, Optimization OPENCV
    print("-"*100)
    print("Threads:",cv2.getNumThreads())
    print("Cores:",cv2.getNumberOfCPUs())
    print("Optimized:",cv2.useOptimized())
    print("-"*100)
    #-- Load the cascades
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)
    if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
        print('--(!)Error loading eyes cascade')
        exit(0)


def detectAndDisplay(frame):
    global faces_detect, faces_count
    frame_color = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_color)

    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
    for (x, y, w, h) in faces:
        center = (x, y)
        faceROI = frame_gray[y:y+h, x:x+w]
        faceColor = frame[y:y+h, x:x+w]
        #-- Capture image from Frame
        """if not compareframe(faceROI):
            path = 'D:\\Usuarios\\cristian.molina\\Desktop\\Instituto\\OCVCODE-master\\ocv_youtube\\photos'
            cv2.imwrite(os.path.join(path, "face%d.jpg" % faces_count), faceColor)"""
        frame = cv2.rectangle(frame, center, (x+w,y+h), (0, 255, 0), 2)
        faces_xywh = str(x),str(y),str(w),str(h)
        faces_str = str(faces_xywh)
        cv2.putText(frame,faces_str, (x-5,y-5),cv2.FONT_HERSHEY_PLAIN,1, (255, 0, 255),1)
        
        #-- Detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int((w2 + h2)*0.25)
            cv2.circle(frame, eye_center, radius, (255, 0, 0), 2)

    cv2.imshow('Capture - Face detection', frame)


def playvideo():
    cap = cv2.VideoCapture(read.readvideo(url))
    cap.set(cv2.CAP_PROP_FPS, 60)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second (video): {0}".format(fps))
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        cv2.useOptimized()
        detectAndDisplay(frame)
        if cv2.waitKey(1) == ord('q'):
            break


# Define arguments parse
parser = argparse.ArgumentParser(
    description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', 
                    default='D:\\opencv\\opencv\\sources\\data\\lbpcascades\\lbpcascade_frontalface_improved.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', 
                    default='D:\\OpenCV\\opencv\\sources\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument(
    '--urlvideo', help='Video streaming.', default='https://www.youtube.com/watch?v=4PkcfQtibmU')

args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
url = args.urlvideo
face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()

tryload()
playvideo()

        
