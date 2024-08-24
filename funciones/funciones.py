import os
import sys
import cv2
import imutils
import threading
import time
import numpy as np
#from moviepy.editor import TextClip, CompositeVideoClip
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

rutaReal = sys.path.append(os.path.join(os.path.dirname(__file__), 'funciones'))

#sys.path.append(os.path.join(os.path.dirname(__file__), 'librerias'))
clasificadorRostros = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#gray = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2GRAY)

# Variables globales
labels = []
faceData = []
label = 0

"""datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2]
)"""

def validarCarpeta(ruta):
    if not os.path.exists(ruta):
        return False
    else:
        return True

def crearCarpeta(ruta, carpeta):
    if not validarCarpeta(ruta + "/" + carpeta):
        os.makedirs(ruta + "/" + carpeta)
        print(f"Carpeta '{carpeta}' creada con exito")
    else:
        print("Si Existe")
    return ruta + "/" + carpeta

def crearArchivo(nombre, extension, contenido = ""):
    
    nombre_completo = f"{nombre}.{extension}"
    
    # Crea y abre el archivo en modo escritura
    with open(nombre_completo, 'w') as archivo:
        if not contenido == '':
            archivo.write(contenido)
        else:
            pass

def crearFrames(ruta, nombre, video):

    frameRutas = crearCarpeta("frames entrenamiento", nombre)

    count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()

        rostros = clasificadorRostros.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in rostros:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            rostro = auxFrame[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(frameRutas, f'{nombre}_{count}.jpg'), rostro)
            count += 1
        
        #cv2.imshow('frame', frame)

        k = cv2.waitKey(1)
        if k == 27 or count >= 300:
            break

def grabarVideo(cap, ruta, nombre):
    global grabar

    full_path = f'{ruta}/{nombre}.mp4'
    
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{full_path}', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    start_time = time.time()
    while grabar:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        if time.time() - start_time > 20:
            break
    
    out.release()
    video = cv2.VideoCapture(full_path)
    crearFrames(ruta, nombre, video)
    print("Grabación detenida")

def abrirCamaraGrabar(ruta, nombre):
    global grabar

    # Crear carpeta para guardar el video
    videoRuta = crearCarpeta(ruta, nombre)

    # Inicia la captura de video desde la cámara predeterminada (cámara 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return
    
    # Iniciar la grabación en un hilo separado
    grabar = True
    grabar_thread = threading.Thread(target=grabarVideo, args=(cap, videoRuta, nombre))
    grabar_thread.start()

    tiempo_inicio = time.time()  # Tiempo de inicio

    while True:
        # Captura frame por frame
        ret, frame = cap.read()

        if not ret:
            print("No se pudo recibir el frame (¿se ha desconectado la cámara?).")
            break

        # Convertir a escala de grises para detección de rostros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostros = clasificadorRostros.detectMultiScale(gray, 1.3, 5)

        # Dibujar un cuadro verde alrededor de cada rostro detectado
        for (x, y, w, h) in rostros:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Muestra el frame en una ventana
        cv2.imshow('Video en vivo', frame)

        # Si se presiona la tecla 'q', se sale del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Si han pasado más de 10 segundos, se sale del bucle
        if time.time() - tiempo_inicio > 20:
            print("Tiempo límite alcanzado. Cerrando cámara.")
            break

    # Detener la grabación
    grabar = False
    grabar_thread.join()

    # Libera el objeto de captura y cierra las ventanas
    cap.release()
    cv2.destroyAllWindows()

def abrirCamaraReconocimiento(ruta, nombre):
    global grabar

    # Crear carpeta para guardar el video
    videoRuta = crearCarpeta(ruta, nombre)

    # Inicia la captura de video desde la cámara predeterminada (cámara 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return
    
    # Iniciar la grabación en un hilo separado
    grabar = True
    grabar_thread = threading.Thread(target=grabarVideo, args=(cap, videoRuta, nombre))
    grabar_thread.start()

    tiempo_inicio = time.time()  # Tiempo de inicio

    while True:
        # Captura frame por frame
        ret, frame = cap.read()

        if not ret:
            print("No se pudo recibir el frame (¿se ha desconectado la cámara?).")
            break

        # Convertir a escala de grises para detección de rostros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostros = clasificadorRostros.detectMultiScale(gray, 1.3, 5)

        # Dibujar un cuadro verde alrededor de cada rostro detectado
        for (x, y, w, h) in rostros:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Muestra el frame en una ventana
        cv2.imshow('Video en vivo', frame)

        # Si se presiona la tecla 'q', se sale del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Si han pasado más de 10 segundos, se sale del bucle
        if time.time() - tiempo_inicio > 20:
            print("Tiempo límite alcanzado. Cerrando cámara.")
            break

    # Detener la grabación
    grabar = False
    grabar_thread.join()

    # Libera el objeto de captura y cierra las ventanas
    cap.release()
    cv2.destroyAllWindows()

###############################################
################ ENTRENAMIENTO ################
###############################################

def entrenamientoReconocimientoRostro(ruta):
    global label
    labels = []
    faceData = []
    listaRostros = os.listdir(ruta)

    # Configuración para augmentación de datos
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2]
    )

    for nombreRostro in listaRostros:
        rostroPath = os.path.join(ruta, nombreRostro)
        for fileName in os.listdir(rostroPath):
            # Leer la imagen en escala de grises
            image = cv2.imread(os.path.join(rostroPath, fileName), 0)

            # Normalizar la imagen
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

            # Redimensionar la imagen para asegurar un tamaño consistente
            target_size = (200, 200)
            image = cv2.resize(image, target_size)

            # Convertir la imagen a formato (height, width, channels)
            image = np.expand_dims(image, axis=-1)  # Añadir un canal de color

            # Añadir la imagen original y su etiqueta
            labels.append(label)
            faceData.append(image)

            # Aplicar augmentación de datos
            image_expanded = np.expand_dims(image, axis=0)  # Añadir dimensión de batch
            for augmented_image in datagen.flow(image_expanded, batch_size=1):
                augmented_image = augmented_image[0].astype('uint8')
                # Asegurarse de que la imagen aumentada tiene la forma correcta
                if augmented_image.ndim == 3:
                    faceData.append(augmented_image)
                    labels.append(label)
                break  # Solo queremos una versión aumentada por cada imagen

            # Mostrar la imagen procesada (opcional)
            cv2.imshow('image', image.squeeze(-1))  # Eliminar el canal de color para mostrar
            cv2.waitKey(10)

        label += 1

    # División en conjuntos de entrenamiento y prueba (opcional)
    X_train, X_test, y_train, y_test = train_test_split(faceData, labels, test_size=0.2, random_state=42)

    # Entrenando el reconocedor de rostros usando LBPHFaceRecognizer
    reconocimientoRostro = cv2.face.LBPHFaceRecognizer_create()
    reconocimientoRostro.train(X_train, np.array(y_train))

    # Evaluación del modelo en el conjunto de prueba (opcional)
    accuracy = 0
    for i in range(len(X_test)):
        pred_label, confidence = reconocimientoRostro.predict(X_test[i])
        if pred_label == y_test[i]:
            accuracy += 1

    print(f"Precisión del modelo: {accuracy / len(X_test) * 100:.2f}%")

    # Almacenando el modelo obtenido
    reconocimientoRostro.write('modeloEigenFace.xml')

    # Destruir todas las ventanas abiertas
    #cv2.destroyAllWindows()
    """label = 0
    listaRostros = os.listdir(ruta)

    for nombreRostro in listaRostros:
        rostroPath = ruta + '/' + nombreRostro
        for fileName in os.listdir(rostroPath):
            labels.append(label)
            faceData.append(cv2.imread(rostroPath + '/' + fileName,0))
            image = cv2.imread(rostroPath + '/' + fileName,0)
            cv2.imshow('image', image)
            cv2.waitKey(10)

        label = label + 1


    # Entrenando el reconocedor de rostros
    reconocimientoRostro = cv2.face.EigenFaceRecognizer_create()
    reconocimientoRostro.train(faceData, np.array(labels))

    # Almacenando el modelo obtenido
    reconocimientoRostro.write('modeloEigenFace.xml')

    #cv2.destroyAllWindows"""


def reconocimiento(ruta):
    listaRostros = os.listdir(ruta)
    reconocimientoRostro = cv2.face.LBPHFaceRecognizer_create()  # Cambiar a LBPH

    # Leyendo el modelo
    if os.path.exists("modeloEigenFace.xml"):
        reconocimientoRostro.read("modeloEigenFace.xml")  # Cargar el modelo LBPH
    else:
        print("Modelo no encontrado.")
        return

    # Abriendo la cámara
    camara = cv2.VideoCapture(0)

    # Verificar si la cámara se ha abierto correctamente
    if not camara.isOpened():
        print("No se pudo abrir la cámara")
        return

    # Cargando el clasificador para detección de rostros
    clasificadorRostro = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        # Capturando frame por frame
        ret, frame = camara.read()
        if not ret:
            print("Error al acceder a la cámara")
            break
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()

        # Detectar rostros en el frame
        rostros = clasificadorRostro.detectMultiScale(gray, 1.3, 5)

        # Procesando cada rostro detectado
        for (x, y, w, h) in rostros:
            rostro = auxFrame[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (200, 200), interpolation=cv2.INTER_CUBIC)  # Redimensionando al tamaño esperado

            # Predicción con el modelo entrenado
            result = reconocimientoRostro.predict(rostro)

            # Mostrar resultados
            #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if result[1] < 56:
                cv2.putText(frame, '{}'.format(listaRostros[result[0]]), (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Desconocido', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                

        # Mostrar el frame
        cv2.imshow("Reconocimiento Facial", frame)

        # Si se presiona la tecla 'q', se sale del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar ventanas
    camara.release()
    cv2.destroyAllWindows()
    """listaRostros = os.listdir(ruta)
    reconocimientoRostro = cv2.face.EigenFaceRecognizer_create()

    # Leyendo el modelo
    reconocimientoRostro.read("modeloEigenFace.xml")

    # Abriendo la cámara
    camara = cv2.VideoCapture(0)

    # Verificar si la cámara se ha abierto correctamente
    if not camara.isOpened():
        print("No se pudo abrir la cámara")
        return

    # Cargando el clasificador para detección de rostros
    clasificadorRostro = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        # Capturando frame por frame
        ret, frame = camara.read()
        if not ret:
            print("Error al acceder a la cámara")
            break
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()

        # Detectar rostros en el frame
        rostros = clasificadorRostro.detectMultiScale(gray, 1.3, 5)

        # Procesando cada rostro detectado
        for (x, y, w, h) in rostros:
            rostro = auxFrame[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)  # Redimensionando al tamaño esperado

            # Predicción con el modelo entrenado
            result = reconocimientoRostro.predict(rostro)

            # Mostrar resultados
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if result[1] < 5700:
                cv2.putText(frame, '{}'.format(listaRostros[result[0]]), (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'Desconocido', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # Mostrar el frame
        cv2.imshow("Reconocimiento Facial", frame)

        # Si se presiona la tecla 'q', se sale del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar ventanas
    #camara.release()
    #cv2.destroyAllWindows()"""