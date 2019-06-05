try:
    import cv2
    import argparse
    import time
    import pafy
    import youtube_dl
    import numpy as np
    import os
    from array import array
    from os import path
    if __name__ == '__main__':
        import sys
        sys.path.append(path.join(path.dirname(__file__), '..'))
        from resources import stream
except ImportError as e:
    print(e)
    print('Para poder instalar el modulo use el siguiente comando "pip install <modulo>"')
#-- Llamar imports fuera de la carpeta main

def tryload():
    #-- Info CPU, Optimization OPENCV
    print("-"*100)
    print("Hilos:",cv2.getNumThreads())
    print("Nucleos:",cv2.getNumberOfCPUs())
    print("OpenCV optimizado:",cv2.useOptimized())
    print("-"*100)
    #Info CPU modificado por Cristian 04-06-2019
    #-- Load the cascades
    file = face_cascades.read()
    if not face_cascade.load(cv2.samples.findFile(file)):
        print('--(!)Error cargando "face cascade"')
        exit(0)
    file = eyes_cascades.read()
    if not eyes_cascade.load(cv2.samples.findFile(file)):
        print('--(!)Error cargando "eyes cascade"')
        exit(0)
    file = smile_cascades.read()
    if not smile_cascade.load(cv2.samples.findFile(file)):
        print('--(!)EError cargando "smile cascade"')
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
        #-- Capture image from Frame (experimental)
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
#Modulo modificado por Cristian 04-06-2019

def playvideo():
    cap = cv2.VideoCapture(stream.readvideo(url.load()))
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

#-- Función modificada por Cristian 04-06-2019

class video:
    def __init__(self,url_video):
        self.url = url_video

    def load(self):
        return self.url

class cascada:
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def read(self):
        return self.dir_path

#-- Proceso principal
face_cascades = cascada('D:\\opencv\\opencv\\sources\\data\\lbpcascades\\lbpcascade_frontalface_improved.xml')
eyes_cascades = cascada('D:\\OpenCV\\opencv\\sources\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml')
smile_cascades = cascada('D:\\OpenCV\\opencv\\sources\\data\\haarcascades\\haarcascade_smile.xml')
face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()
smile_cascade = cv2.CascadeClassifier()
url = video(input('Inserte el url de un video de youtube:\n'))
tryload()
playvideo()
        
