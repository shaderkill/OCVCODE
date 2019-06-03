from __future__ import print_function
import numpy as np
import os.path
import cv2
import argparse
import time
import urllib
from urllib.request import urlopen

def numpy2pil(np_array: np.ndarray) -> Image:
    assert_msg = 'Input shall be a HxWx3 ndarray'
    assert isinstance(np_array, np.ndarray), assert_msg
    assert len(np_array.shape) == 3, assert_msg
    assert np_array.shape[2] == 3, assert_msg

    img = Image.fromarray(np_array, 'RGB')
    return img

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray, 1.05, 5)
    for (x, y, w, h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv2.ellipse(frame, center, (w//1, h//1),
                           0, 0, 360, (255, 0, 255), 2)
        faceROI = frame_gray[y:y+h, x:x+w]
        faceCOLOR = frame[y:y+h, x:x+w]
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI, 1.2, 5)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 2)
    cv2.imshow('Capture - Face detection', frame)


parser = argparse.ArgumentParser(
    description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', 
                    default='D:\\opencv\\opencv\\sources\\data\\lbpcascades\\lbpcascade_frontalface_improved.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', 
                    default='D:\\OpenCV\\opencv\\sources\\data\\haarcascades\\haarcascade_eye.xml')
parser.add_argument(
    '--urlvideo', help='Video streaming.', default='http://10.113.1.63:8080/shot.jpg')
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
url = args.urlvideo
face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()
#-- 1. Load the cascades
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
while True:
    #-- 2. Read the video stream
    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img = cv2.imdecode(imgNp,-1)
    if img is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(img)
    time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break