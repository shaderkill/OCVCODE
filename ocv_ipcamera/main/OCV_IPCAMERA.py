# -- Imports
import urllib.request
from os.path import normpath, realpath

import cv2
import numpy as np


# -- Funciones
# -- Función encargada de detectar y dibujar las cascadas
def detect_and_display(frame):
    frame_color = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_color)
    # -- Detectar caras de frente
    ffaces = frontalface_cascade.detectMultiScale(frame_gray, 1.25, 5)
    if len(ffaces) > 0:
        for (x, y, w, h) in ffaces:
            top_left = (x, y)
            botom_right = (x+w, y+h)
            faceROI = frame_gray[y:y+h, x:x+w]
            faceColor = frame[y:y+h, x:x+w]
            frame = cv2.rectangle(frame,
                                  top_left, botom_right, (43, 248, 243), 1)
            faces_xywh = str(x), str(y), str(w), str(h)
            faces_str = str(faces_xywh)
            cv2.putText(frame, faces_str,
                        (x-15, y-15), cv2.FONT_HERSHEY_PLAIN,
                        1, (255, 0, 255), 1)
            # -- Detectar ojos
            eyes = eyes_cascade.detectMultiScale(faceROI)
            for (x2, y2, w2, h2) in eyes:
                eye_center = (x + x2 + w2//2, y + y2 + h2//2)
                radius = int((w2 + h2)*0.25)
                cv2.circle(frame,
                           eye_center, radius, (43, 248, 243), 1)
            # -- Detectar sonrisas
            smiles = smile_cascade.detectMultiScale(faceROI, 1.3, 26)
            for (x3, y3, w3, h3) in smiles:
                top_left = (x3, y3)
                botom_right = (x3 + w3, y3 + h3)
                cv2.rectangle(faceColor,
                              top_left, botom_right, (43, 248, 243), 1)
                cv2.putText(frame, "Sonriendo",
                            (x-15, y + 250), cv2.FONT_HERSHEY_PLAIN,
                            1, (255, 0, 255), 1)
    else:
        # -- Detectar caras de frente
        pfaces = profileface_cascade.detectMultiScale(frame_gray, 1.3, 2)
        for (x, y, w, h) in pfaces:
            top_left = (x, y)
            botom_right = (x+w, y+h)
            faceROI = frame_gray[y:y+h, x:x+w]
            faceColor = frame[y:y+h, x:x+w]
            frame = cv2.rectangle(frame,
                                  top_left, botom_right, (43, 248, 243), 1)
            faces_xywh = str(x), str(y), str(w), str(h)
            faces_str = str(faces_xywh)
            cv2.putText(frame, faces_str,
                        (x-15, y-15), cv2.FONT_HERSHEY_PLAIN,
                        1, (255, 0, 255), 1)
    cv2.imshow('IP Camera - Deteccion de caras', frame)
# -- Función modificada por Cristian 05-06-2019


# -- Cargar cascadas
def tryload():
    # -- Info CPU, Optimization OPENCV
    print("-"*100)
    print("Hilos:", cv2.getNumThreads())
    print("Nucleos:", cv2.getNumberOfCPUs())
    print("OpenCV optimizado:", cv2.useOptimized())
    print("-"*100)
    # -- Info CPU modificado por Cristian 04-06-2019
    # -- Load the cascades
    file = frontalface_cascades.read()
    print(file)
    if not frontalface_cascade.load(cv2.samples.findFile(file)):
        print(' -- (!)Error cargando "face cascade"')
        exit(0)
    file = profileface_cascades.read()
    if not profileface_cascade.load(cv2.samples.findFile(file)):
        print(' -- (!)Error cargando "face cascade"')
        exit(0)
    file = eyes_cascades.read()
    if not eyes_cascade.load(cv2.samples.findFile(file)):
        print(' -- (!)Error cargando "eyes cascade"')
        exit(0)
    file = smile_cascades.read()
    if not smile_cascade.load(cv2.samples.findFile(file)):
        print(' -- (!)EError cargando "smile cascade"')
        exit(0)


# -- Intentar capturar video streaming
def readvideo():
    success = False
    count = 1
    while not success:
        try:
            imgResp = urllib.request.urlopen(url.load())
            imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
            img = cv2.imdecode(imgNp, -1)
            success = True
            return img
        except Exception as e:
            print(e, '\nReintentando conectar (%d/6)\n'
                  % count)
            count += 1
            if count > 6:
                return
# -- Función modificada por Cristian 04-06-2019


# -- Reproducir el video capturado.
def playvideo():
    while True:
        img = readvideo()
        if img is None:
            print(' -- (!)No se ha logrado capturar una imagen')
            print('Streaming finalizado')
            break
        cv2.useOptimized()
        detect_and_display(img)
        if cv2.waitKey(1) == ord('q'):
            break
# -- Función modificada por Cristian 04-06-2019.


class video:
    def __init__(self, url_video):
        self.url = url_video

    def load(self):
        return self.url


class cascada:
    def __init__(self, dir_path):
        path_str = r'%s' % dir_path
        path_str = str(path_str).replace('\\', '\\\\')
        self.dir_path = path_str

    def read(self):
        return self.dir_path
# -- Se crearon ambas clases para mantener datos de cascadas y videos


# -- Proceso principal
frontalface_cascades = cascada(normpath(realpath(
    'D:/OpenCV/opencv/sources/data/' +
    'lbpcascades/lbpcascade_frontalface_improved.xml')))
profileface_cascades = cascada(normpath(realpath(
    'D:/OpenCV/opencv/sources/data/' +
    'lbpcascades/lbpcascade_profileface.xml')))
eyes_cascades = cascada(normpath(realpath(
    'D:/OpenCV/opencv/sources/data/' +
    'haarcascades/haarcascade_eye_tree_eyeglasses.xml')))
smile_cascades = cascada(normpath(realpath(
    'D:/OpenCV/opencv/sources/data/' +
    'haarcascades/haarcascade_smile.xml')))
frontalface_cascade = cv2.CascadeClassifier()
profileface_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()
smile_cascade = cv2.CascadeClassifier()
url = video('http://192.168.101.100:8080/shot.jpg')
tryload()
playvideo()
