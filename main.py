import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'funciones'))


#Importaciones archivos para el proyecto
import funciones as fn

personName = input("Escribe tu nombre: ")

dataPath =  './video entrenamiento'
#personPath = dataPath + '/' + personName
#os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
#print(personPath)
#fn.validarCarpeta( dataPath)
#fn.crearCarpeta(dataPath, personName)
fn.abrirCamaraGrabar(dataPath, personName)