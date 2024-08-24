import os
import sys
import cv2
import imutils
sys.path.append(os.path.join(os.path.dirname(__file__), 'funciones'))


#Importaciones archivos para el proyecto
import funciones as fn

dataPath =  './frames entrenamiento'
fn.entrenamientoReconocimientoRostro(dataPath)
#fn.reconocimiento(dataPath)