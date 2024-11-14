import cv2
import mediapipe as mp
import math
from queue import Queue
from ultralytics import YOLO
import time
from collections import defaultdict
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QGridLayout, QInputDialog
from PyQt6.QtCore import Qt
import sys
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
#import face_recognition
from gpt.gpt_description import generar_descripcion, hablar_texto

# Inicializar MediaPipe y YOLO
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Modelos
model= YOLO('yolov10s.pt')

# Variables globales
paused = False
capturing = False
hand_position = [0.0, 0.0]
finger_distance = 0.0
detected_objects = []
gesture_action = None
object_threshold = 5
# Variable para almacenar el contador de objetos detectados
detected_object_counts = defaultdict(int)
frame_window = 30  # Ventana de tiempo en cuadros
total_frames = 0  # Variable para llevar el total de frames procesados
confidence_threshold = 0.5  # Establece el umbral de confianza


contexto = {
    "objetos_finales": [],
    "gestos_detectados": set(),
    "poses_detectadas": set()
}

def solicitar_ubicacion():
    text, ok = QInputDialog.getText(None, 'Ubicación', 'Ingresa la ubicación actual:')
    if ok and text:
        return text
    else:
        return "Ubicación no especificada"
    
# Función para filtrar objetos por confianza
def filtrar_por_confianza(detected_object_counts, confidence_threshold=0.5):
    """
    Filtra los objetos detectados con una confianza mayor al umbral dado.
    """
    filtered_objects = defaultdict(int)
    for obj, count in detected_object_counts.items():
        parts = obj.rsplit(' ', 1)  # Dividir en el último espacio
        if len(parts) == 2:
            label, confidence = parts[0], parts[1]
            try:
                if float(confidence) >= confidence_threshold:
                    filtered_objects[label] += count
            except ValueError:
                print(f"Error al procesar la confianza de {obj}")
        else:
            print(f"Etiqueta no válida: {obj}")
    return filtered_objects

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
        contexto["gestos_detectados"].add("Saludo")
        return True  # Gesto de saludo detectado
    return False

def detectar_parado(pose_landmarks):
    knee_left = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    knee_right = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    hip_left = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    hip_right = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    if (knee_left.visibility > 0.5 and knee_right.visibility > 0.5 and
        knee_left.y > hip_left.y and knee_right.y > hip_right.y):
        return "De pie"
    return "Sentado"

def detectar_manos_levantadas(pose_landmarks):
    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    if (left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y):
        return "Manos levantadas"
    return None

def detectar_manos_cruzadas(pose_landmarks):
    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]

    if (abs(left_wrist.x - right_wrist.x) < 0.05 and
        left_wrist.y < left_elbow.y and right_wrist.y < right_elbow.y):
        return "Manos cruzadas"
    return None

def detectar_x_brazos(pose_landmarks):
    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    if (left_wrist.x < right_shoulder.x and right_wrist.x > left_shoulder.x and
        left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y):
        return "X con los brazos"
    return None

def detect_gestures_and_objects():
    global hand_position, finger_distance, detected_objects, gesture_action, paused, capturing, total_frames
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
            
            total_frames += 1


            if time.time() - start_time > 5:  # Limitar a 5 segundos de captura
                print("Límite de tiempo alcanzado. Procesando datos...")

                 # Seleccionar el objeto detectado más frecuentemente
                min_frames_for_object = total_frames * 0.1
                # Filtrar objetos por confianza
                objetos_filtrados = filtrar_por_confianza(detected_object_counts, confidence_threshold=0.5)

                # Filtrar objetos que aparecieron en más del 10% de los frames
                objetos_finales = [obj for obj, count in objetos_filtrados.items() if count >= min_frames_for_object]

                
                # Mostrar resultados
                print(f"Total de fotogramas procesados: {total_frames}")
                print(f"objetos filtrados: {objetos_filtrados}")
                print(f"Conteo de objetos detectados: {dict(detected_object_counts)}")
                print("Objetos detectados:", objetos_finales)
                print("Gestos detectados:", set(contexto["gestos_detectados"]))
 
                descripcion = generar_descripcion(
                    objetos_finales, 
                    ", ".join(contexto["gestos_detectados"]),
                    ", ".join(contexto["poses_detectadas"]),                    
                    ubicacion,
                )
                print(descripcion)
                hablar_texto(descripcion)
               
                # Detener la detección para evitar más procesamiento
                capturing = False
                break

            # Detección de gestos
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_hands = hands.process(img_rgb)
            result_pose = pose.process(img_rgb)

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

                    # Detectar si el gesto de saludo está presente
                    if detectar_gesto_saludo(hand_landmarks):
                        print("Saludo detectado")
                        # Puedes hacer algo aquí, como agregar el gesto al contexto
                        contexto["gestos_detectados"].add("Saludo")
                    if gesture_action and gesture_action not in contexto["gestos_detectados"]:
                        contexto["gestos_detectados"].append(gesture_action)

            if result_pose.pose_landmarks:
                mp_draw.draw_landmarks(frame, result_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                pose_manos_levantadas = detectar_manos_levantadas(result_pose.pose_landmarks)
                if pose_manos_levantadas:
                    contexto["poses_detectadas"].add(pose_manos_levantadas)
                pose_manos_cruzadas = detectar_manos_cruzadas(result_pose.pose_landmarks)
                if pose_manos_cruzadas:
                    contexto["poses_detectadas"].add(pose_manos_cruzadas)
                pose_x_brazos = detectar_x_brazos(result_pose.pose_landmarks)
                if pose_x_brazos:
                    contexto["poses_detectadas"].add(pose_x_brazos)
                pose_parado_sentado = detectar_parado(result_pose.pose_landmarks)
                contexto["poses_detectadas"].add(pose_parado_sentado)

            yolo_results = model(frame, show=False)

            for result in yolo_results:
                for obj in result.boxes:
                    x1, y1, x2, y2 = map(int, obj.xyxy[0])
                    class_id = int(obj.cls)
                    confidence = obj.conf
                    label = f'{model.names[class_id]} {float(confidence):.2f}'
                      # Evitar el uso de rsplit para evitar el error, simplemente maneja nombre y confianza por separado
                    objeto_confianza = f'{model.names[class_id]} {float(confidence):.2f}'  

                    if confidence >= confidence_threshold:
                        detected_objects.append(f"{model.names[class_id]}")
                        print(f"Detected objects: {detected_objects}")
                        detected_object_counts[objeto_confianza] += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                       
            cv2.imshow('Detección de Objetos y Gestos', frame)    

            # Salir si se presiona la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error al procesar la cámara: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


class FloatingMenu(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(50, 50, 200, 200)

        central_widget = QWidget()
        layout = QVBoxLayout()

        # Botón para iniciar la detección
        button_start = QPushButton("Iniciar Detección")
        button_start.clicked.connect(self.iniciar_deteccion)
        layout.addWidget(button_start)

        # Botón para pausar/reanudar
        button_pause = QPushButton("Pausar/Continuar")
        button_pause.clicked.connect(self.toggle_pause)
        layout.addWidget(button_pause)

        # Botón para reiniciar el programa
        button_restart = QPushButton("Reiniciar")
        button_restart.clicked.connect(self.reiniciar_programa)
        layout.addWidget(button_restart)

        # Botón para salir de la aplicación
        button_exit = QPushButton("Salir")
        button_exit.clicked.connect(self.close_app)
        layout.addWidget(button_exit)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def iniciar_deteccion(self):
        global capturing, ubicacion
        ubicacion = solicitar_ubicacion()
        capturing = True
        detect_gestures_and_objects()

    def toggle_pause():
        global paused
        paused = not paused

    def reiniciar_programa(self):
        global contexto
        contexto = {
            "objetos_detectados": [],
            "gestos_detectados": [],
            "poses_detectadas": []
        }
    toggle_pause()

    def close_app(self):
        sys.exit()

# Ejecutar la aplicación PyQt6
app = QApplication(sys.argv)
window = FloatingMenu()
window.show()
sys.exit(app.exec())