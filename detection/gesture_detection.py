import cv2
import mediapipe as mp
import math
from queue import Queue
from ultralytics import YOLO
import tkinter as tk
from tkinter import Button, Label
import time
from collections import defaultdict
import face_recognition
from gpt.gpt_description import generar_descripcion, hablar_texto

# Inicializar MediaPipe y YOLO
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Modelos
model1 = YOLO('yolov10n.pt')
model2 = YOLO('/Users/dark0/Documents/Visor/assets/models/train13/weights/best.pt')


juan_image = face_recognition.load_image_file("/Users/dark0/Documents/Visor/assets/img/juan.jpeg")
juan_encoding = face_recognition.face_encodings(juan_image)[0]

# Cola para pasar el frame a OpenGL
frame_queue = Queue(maxsize=1)

# Variables globales
paused = False
capturing = False
hand_position = [0.0, 0.0]
finger_distance = 0.0
detected_objects = []
gesture_action = None

# Variable para almacenar el contador de objetos detectados
object_count = defaultdict(int)  # Almacena el conteo de cada objeto
frame_window = 30  # Ventana de tiempo en cuadros
current_frame = 0  # Llevar cuenta de los cuadros procesados


contexto = {
    "objetos_detectados": [],
    "gestos_detectados": []
}

# Función para iniciar la detección
def iniciar_deteccion():
    global capturing
    capturing = True
    detect_gestures_and_objects()

# Función para detener la detección
def detener_deteccion():
    global capturing
    capturing = False

# Función para pausar/reanudar
def toggle_pause():
    global paused
    paused = not paused

def reiniciar_programa():
    # Reiniciar el diccionario de contexto y despausar el programa
    global contexto
    contexto = {
        "objetos_detectados": [],
        "gestos_detectados": []
    }
    toggle_pause()  # Despausar la captura de gestos y objetos para reiniciar

def cerrar_programa():
    root.quit() 


# Función para verificar si los dedos están extendidos
def dedos_extendidos(hand_landmarks):
    """
    Verifica si los dedos están extendidos.
    Devuelve True si todos los dedos están extendidos.
    """
    # Landmarks de los dedos
    dedos_landmarks = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]

    # Verificar si cada dedo está más arriba que el nudillo correspondiente
    for landmark in dedos_landmarks:
        if hand_landmarks.landmark[landmark].y > hand_landmarks.landmark[landmark - 2].y:
            return False  # Si un dedo no está extendido, devuelve False
    return True


# Función para detectar el gesto de saludo
def detectar_gesto_saludo(hand_landmarks):
    """
    Detecta si el gesto de saludo está presente.
    """
    # Detectar si la palma está hacia la cámara
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    # Si la palma está orientada hacia la cámara (los puntos de los nudillos de los dedos índice y medio están debajo de la muñeca)
    palma_orientada = wrist.z < index_mcp.z and wrist.z < middle_mcp.z

    # Verificar si los dedos están extendidos (excepto el pulgar)
    if palma_orientada and dedos_extendidos(hand_landmarks):
        return True  # Gesto de saludo detectado
    return False


def detect_gestures_and_objects():
    global hand_position, finger_distance, detected_objects, gesture_action
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    start_time = time.time()

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    try:
        while capturing:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo capturar el frame.")
                break

            # Detección de rostro
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            juan_en_pantalla = False

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces([juan_encoding], face_encoding)

                if True in matches:
                    juan_en_pantalla = True
                    if "Juan" not in contexto["objetos_detectados"]:
                        contexto["objetos_detectados"].append("Juan está en pantalla")
                        print("Juan está en pantalla")

            if not juan_en_pantalla:
                print("Juan no está en pantalla.")

            if time.time() - start_time > 5:  # Limitar a 5 segundos de captura
                print("Límite de tiempo alcanzado. Procesando datos...")

                 # Seleccionar el objeto detectado más frecuentemente
                objeto_mas_frecuente = max(object_count, key=object_count.get)
                contexto["objetos_detectados"] = [objeto_mas_frecuente]

                descripcion = generar_descripcion(contexto["objetos_detectados"], ", ".join(contexto["gestos_detectados"]))
                print(descripcion)
                hablar_texto(descripcion)

                detener_deteccion()
                break

            # Detección de gestos
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_hands = hands.process(img_rgb)

            if result_hands.multi_hand_landmarks:
                for hand_landmarks in result_hands.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Obtener la posición de la muñeca y los dedos
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    # Controla la posición de la mano
                    hand_position[0] = (wrist.x - 0.5) * 2
                    hand_position[1] = (wrist.y - 0.5) * 2

                    # Calcular la distancia entre el pulgar y el índice
                    dx = thumb_tip.x - index_tip.x
                    dy = thumb_tip.y - index_tip.y
                    finger_distance = math.sqrt(dx ** 2 + dy ** 2)

                     # Asignar una acción en función del gesto
                    if finger_distance < 0.02:
                        gesture_action = "Pinza"
                    else:
                        gesture_action = None

                    # Detectar si el gesto de saludo está presente
                    if detectar_gesto_saludo(hand_landmarks):
                        print("Saludo detectado")
                        # Puedes hacer algo aquí, como agregar el gesto al contexto
                        contexto["gestos_detectados"].append("Saludo")
                

                    if gesture_action and gesture_action not in contexto["gestos_detectados"]:
                        contexto["gestos_detectados"].append(gesture_action)

           

            for model in [model1, model2]:
                # Detección de objetos con YOLO
                yolo_results = model(frame, show=False)
                for result in yolo_results:
                    for obj in result.boxes:
                        x1, y1, x2, y2 = map(int, obj.xyxy[0])
                        class_id = int(obj.cls)
                        confidence = obj.conf
                        label = f'{model.names[class_id]} {float(confidence):.2f}'
                        detected_objects.append(label)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        object_count[label] += 1

            cv2.imshow('Detección de Objetos y Gestos', frame)    

            # Salir si se presiona la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                detener_deteccion()
                break

    except Exception as e:
        print(f"Error al procesar la cámara: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Crear la interfaz gráfica con Tkinter
root = tk.Tk()
root.title("Control de Cámara")

# Botones
btn_iniciar = Button(root, text="Iniciar Detección", command=iniciar_deteccion)
btn_iniciar.pack(pady=10)

# Botón para reiniciar el programa
btn_reiniciar = Button(root, text="Reiniciar", command=reiniciar_programa)
btn_reiniciar.pack(pady=10)

# Botón para cerrar el programa
btn_salir = Button(root, text="Salir", command=cerrar_programa)
btn_salir.pack(pady=10)

# Ejecutar la ventana de Tkinter
root.mainloop()