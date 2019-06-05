try:
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
except ImportError as e:
    print(e)
    print('Para poder instalar el modulo use el siguiente comando "pip install <modulo>"')
#-- Llamar imports fuera de la carpeta main


#-- Función archivada, Numpy to PIL para manejo de imagenes
"""def numpy2pil(np_array: np.ndarray) -> Image:
    assert_msg = 'Input'
    assert isinstance(np_array, np.ndarray), assert_msg
    assert len(np_array.shape) == 3, assert_msg
    assert np_array.shape[2] == 3, assert_msg
    img = Image.fromarray(np_array, 'RGB')
    return img"""


def detectAndDisplay(frame):
    frame_color = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_color)
    #-- Detectar caras
    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
    for (x, y, w, h) in faces:
        top_left = (x, y)
        botom_right = (x+w,y+h)
        faceROI = frame_gray[y:y+h, x:x+w]
        faceColor = frame[y:y+h, x:x+w]
        frame = cv2.rectangle(frame, top_left, botom_right, (43, 248, 243), 1)
        faces_xywh = str(x),str(y),str(w),str(h)
        faces_str = str(faces_xywh)
        cv2.putText(frame,faces_str, (x-15,y-15),cv2.FONT_HERSHEY_PLAIN,1, (255, 0, 255),1)
        #-- Detectar ojos
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int((w2 + h2)*0.25)
            cv2.circle(frame, eye_center, radius, (43, 248, 243), 1)
        #-- Detectar sonrisas
        smiles = smile_cascade.detectMultiScale(faceROI, 1.3, 26)
        for (x3, y3, w3, h3) in smiles:
            top_left = (x3, y3)
            botom_right = (x3 + w3, y3 + h3)
            cv2.rectangle(faceColor, top_left, botom_right, (43, 248, 243), 1)
            cv2.putText(frame,"Sonriendo", (x-15, y + 250),cv2.FONT_HERSHEY_PLAIN,1, (255, 0, 255),1)
    cv2.imshow('IP Camera - Detección de caras', frame)
#-- Función modificada por Cristian 05-06-2019


def tryload():
    #-- Info CPU, Optimization OPENCV
    print("-"*100)
    print("Hilos:",cv2.getNumThreads())
    print("Nucleos:",cv2.getNumberOfCPUs())
    print("OpenCV optimizado:",cv2.useOptimized())
    print("-"*100)
    print("-"*100)
    #Info CPU modificado por Cristian 04-06-2019
    #-- Load the cascades
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
        print('--(!)Error cargando "face cascade"')
        exit(0)
    if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
        print('--(!)Error cargando "eyes cascade"')
        exit(0)
    if not smile_cascade.load(cv2.samples.findFile(smile_cascade_name)):
        print('--(!)EError cargando "smile cascade"')
        exit(0)


def readvideo():
    #-- Try read video streaming
    success = False
    count = 1
    while success == False:
        try:
            imgResp = urllib.request.urlopen(url.load())
            imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
            img = cv2.imdecode(imgNp,-1)
            success = True
            return img
        except:
            print('IP Camera desconectada, reintentando conectar (%d/6)' % count)
            count += 1
            if count > 6:
                return
#-- Función modificada por Cristian 04-06-2019

def playvideo():
    while True:
        img = readvideo()
        if img is None:
            print('--(!)No se ha logrado capturar una imagen')
            print('Streaming finalizado')
            break
        cv2.useOptimized()
        detectAndDisplay(img)
        if cv2.waitKey(1) == ord('q'):
            break
#-- Función modificada por Cristian 04-06-2019

class video:
    def __init__(self, url_video):
        self.url = url_video

    def load(self):
        return self.url

class cascada:
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def load(self):
        return self.dir_path

#-- Proceso principal
face_cascade = cascada('D:\\opencv\\opencv\\sources\\data\\lbpcascades\\lbpcascade_frontalface_improved.xml')
eyes_cascade = cascada('D:\\OpenCV\\opencv\\sources\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cascada('D:\\OpenCV\\opencv\\sources\\data\\haarcascades\\haarcascade_smile.xml')
url = video('http://10.113.1.63:8080/shot.jpg')
face_cascade_name =face_cascade.load()
eyes_cascade_name = eyes_cascade.load()
smile_cascade_name = smile_cascade.load()
face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()
smile_cascade = cv2.CascadeClassifier()
tryload()
playvideo()