import cv2
import mediapipe as mp
import math
from queue import Queue
from ultralytics import YOLO
from gpt.gpt_description import generar_descripcion, hablar_texto

# Inicializar MediaPipe y YOLO
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
model = YOLO('yolov10n.pt')

# Cola para pasar el frame a OpenGL
frame_queue = Queue(maxsize=1)

# Variables globales
hand_position = [0.0, 0.0]
finger_distance = 0.0
detected_objects = []
gesture_action = None

contexto = {
    "objetos_detectados": [],
    "gestos_detectados": []
}

def detect_gestures_and_objects():
    global hand_position, finger_distance, detected_objects, gesture_action
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo capturar el frame.")
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
                    if finger_distance < 0.05:
                        gesture_action = "Pinza"
                    else:
                        gesture_action = None

                    if gesture_action and gesture_action not in contexto["gestos_detectados"]:
                        contexto["gestos_detectados"].append(gesture_action)

            # Detección de objetos con YOLO
            yolo_results = model(frame, show=False)

            detected_objects = []
            for result in yolo_results:
                for obj in result.boxes:
                    x1, y1, x2, y2 = map(int, obj.xyxy[0])
                    class_id = int(obj.cls)
                    confidence = obj.conf
                    label = f'{model.names[class_id]} {float(confidence):.2f}'
                    detected_objects.append(label)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    # Almacenar el objeto detectado en el diccionario
                    if label not in contexto["objetos_detectados"]:
                        contexto["objetos_detectados"].append(label)

            # Generar narrativa de la escena basada en objetos y gestos cada cierto tiempo
            if contexto["objetos_detectados"] and contexto["gestos_detectados"]:
                descripcion = generar_descripcion(contexto["objetos_detectados"], ", ".join(contexto["gestos_detectados"]))
                print(descripcion)
                hablar_texto(descripcion)  


            # Mandar el frame a OpenGL
            if not frame_queue.full():
                frame_queue.put(frame)
            else:
                frame_queue.get()
                frame_queue.put(frame)

            cv2.imshow('Detección de Objetos y Gestos', frame)    

            # Salir si se presiona la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error al procesar la cámara: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()