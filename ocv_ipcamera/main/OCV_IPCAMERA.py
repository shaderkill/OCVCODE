from __future__ import print_function
import numpy as np
import os.path
import cv2
import argparse
import time
import urllib
from urllib.request import urlopen
from PIL import Image
import os
from os import path
if __name__ == '__main__':
    import sys
    sys.path.append(path.join(path.dirname(__file__), '..'))
#-- Llamar imports fuera de la carpeta main


def numpy2pil(np_array: np.ndarray) -> Image:
    assert_msg = 'Input'
    assert isinstance(np_array, np.ndarray), assert_msg
    assert len(np_array.shape) == 3, assert_msg
    assert np_array.shape[2] == 3, assert_msg
    img = Image.fromarray(np_array, 'RGB')
    return img

def detectAndDisplay(frame):
    frame_color = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_color)

    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
    for (x, y, w, h) in faces:
        center = (x, y)
        faceROI = frame_gray[y:y+h, x:x+w]
        faceColor = frame[y:y+h, x:x+w]
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
    cv2.imshow('Face detection', frame)
#-- Función modificada por Cristian 04-06-2019

def tryload():
    #-- Info CPU, Optimization OPENCV
    print("-"*100)
    print("Threads:",cv2.getNumThreads())
    print("Cores:",cv2.getNumberOfCPUs())
    print("Optimized:",cv2.useOptimized())
    print("-"*100)
    #Info CPU modificado por Cristian 04-06-2019
    #-- Load the cascades
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)
    if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
        print('--(!)Error loading eyes cascade')
        exit(0)
    if not smile_cascade.load(cv2.samples.findFile(smile_cascade_name)):
        print('--(!)Error loading smile cascade')
        exit(0)

def readvideo():
    #-- Try read video streaming
    success = False
    count = 1
    while success == False:
        try:
            imgResp = urllib.request.urlopen(url)
            imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
            img = cv2.imdecode(imgNp,-1)
            success = True
            return img
        except:
            print('IP Camera disconnected, trying (%d/10)' % count)
            success = False
            count += 1
            if count > 10:
                return
#-- Función modificada por Cristian 04-06-2019

def playvideo():
    while True:
        img = readvideo()
        if img is None:
            print('--(!) No captured frame -- Break!')
            print('Streaming ending')
            break
        cv2.useOptimized()
        detectAndDisplay(img)
        if cv2.waitKey(1) == ord('q'):
            break
#-- Función modificada por Cristian 04-06-2019

#-- Define arguments parse
parser = argparse.ArgumentParser(
    description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', 
                    default='D:\\opencv\\opencv\\sources\\data\\lbpcascades\\lbpcascade_frontalface_improved.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', 
                    default='D:\\OpenCV\\opencv\\sources\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--smile_cascade', help='Path to smile cascade.', 
                    default='D:\\OpenCV\\opencv\\sources\\data\\haarcascades\\haarcascade_smile.xml')
parser.add_argument(
    '--urlvideo', help='Video streaming.', default='http://10.113.1.63:8080/shot.jpg')

#-- Proceso principal
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
smile_cascade_name = args.smile_cascade
url = args.urlvideo
face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()
smile_cascade = cv2.CascadeClassifier()
tryload()
playvideo()