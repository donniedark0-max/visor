import sys
import os
import cv2
import threading
import time
import numpy as np
import mediapipe as mp
import math
import random  # Importar el módulo random para respuestas aleatorias
from collections import defaultdict
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QInputDialog, QLabel
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from ultralytics import YOLO
from gpt.gpt_description import generar_descripcion, hablar_texto
import pytesseract
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from functools import partial
import requests
from detection.mongodb import guardar_deteccion,guardar_persona, guardar_log, personas_collection
from word2number import w2n
import face_recognition
import base64
from io import BytesIO
from PIL import Image

# Almacenar datos faciales y nombres
known_face_encodings = []
known_face_names = []
known_face_teams = {}
# Cargar variables de entorno desde .env
load_dotenv()

# Configuración de la API de Azure Speech
azure_speech_key = os.getenv('AZURE_SPEECH_KEY')
azure_region = os.getenv('AZURE_REGION')
barcelona_api =os.getenv('BARCA_KEY')

speech_config = speechsdk.SpeechConfig(subscription=azure_speech_key, region=azure_region)
speech_config.speech_synthesis_voice_name = "es-AR-ElenaNeural"

#pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


# Inicializar MediaPipe y YOLO
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Modelo YOLO
model = YOLO('yolov10x.pt')

# Variables globales
capturing = True
command_detected = None
texto_detectado = ""
ubicacion = ""
confidence_threshold = 0.5  # Umbral de confianza

contexto = {
    "objetos_finales": [],
    "gestos_detectados": set(),
    "poses_detectadas": set(),
    "personas_detectadas": set(),
    "equipos_detectados": {}
}
translations = {
    "person": "persona",
    "cell phone": "teléfono",
    "laptop": "computadora portátil",
    "car": "auto",
    "dog": "perro",
    "cat": "gato",
    "bottle": "botella",
    "chair": "silla",
    "table": "mesa",
    "walllet": "billetera",
    "umbrella": "paraguas",
    "sunglasses": "gafas de sol",
    "book": "libro",
    "backpack": "mochila",
    "shirt": "pantalón",
    # Agrega más traducciones según sea necesario
}
def translate_label(label):
    return translations.get(label, label)  # Si no hay traducción, devuelve la etiqueta original
# Coloca la función después de las importaciones y la declaración de las listas known_face_encodings y known_face_names
def cargar_datos_entrenados():
    """Carga los datos de la base de datos para el reconocimiento facial."""
    known_face_encodings.clear()
    known_face_names.clear()
    known_face_teams.clear()

    personas = personas_collection.find()  # Asegúrate de importar personas_collection desde mongodb.py
    for persona in personas:
        if "imagen" in persona:
            try:
                # Convertir la imagen desde base64 a un array de imagen
                img_data = base64.b64decode(persona["imagen"])
                img_array = np.array(Image.open(BytesIO(img_data)))
                
                # Verificar que la codificación facial es posible
                encodings = face_recognition.face_encodings(img_array)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(persona['nombre'])
                    known_face_teams[persona['nombre']] = persona.get('equipo_favorito', '')

                else:
                    print(f"No se pudo encontrar una cara en la imagen de {persona['nombre']}")
            except Exception as e:
                print(f"Error procesando la imagen de {persona['nombre']}: {e}")
    print("Datos entrenados cargados correctamente.")

# Después, asegúrate de que la función `cargar_datos_entrenados()` se llame en `main()` o antes de `reconocer_personas()`.



def detect_and_identify_object():
    """
    Detecta objetos en un frame capturado y responde con el objeto más probable.
    """
    try:
        # Inicializar la cámara
        cap = cv2.VideoCapture(1)  # Usar CAP_DSHOW para Windows si aplica
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara.")
            hablar_texto("Lo siento, no pude acceder a la cámara.")
            return

        # Configurar la cámara
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Intentar capturar un frame válido
        max_retries = 5
        retries = 0
        frame = None

        while retries < max_retries:
            ret, frame = cap.read()
            if ret and frame is not None:
                break
            retries += 1
            print(f"Intentando capturar el frame nuevamente... Intento {retries}/{max_retries}")
            time.sleep(0.5)

        if frame is None:
            print("Error: No se pudo capturar un frame válido después de varios intentos.")
            hablar_texto("Lo siento, no pude capturar la imagen.")
            return

        print("Frame capturado exitosamente. Iniciando detección de objetos...")

        # Procesar detección de objetos con YOLO
        try:
            yolo_results = model(frame, show=False)  # Procesar el frame con YOLO
            detected_objects = []

            for result in yolo_results:
                for obj in result.boxes:
                    class_id = int(obj.cls)
                    confidence = float(obj.conf)
                    label = model.names[class_id]
                    detected_objects.append((label, confidence))

            if detected_objects:
                # Ordenar por confianza y tomar el objeto más probable
                detected_objects.sort(key=lambda x: x[1], reverse=True)
                objeto, confianza = detected_objects[0]
                objeto_traducido = translate_label(objeto)
                print(f"Objeto detectado: {objeto_traducido} con confianza {confianza:.2f}")
                hablar_texto(f"Creo que es un {objeto_traducido} con una confianza de {int(confianza * 100)}%.")
            else:
                print("No se detectaron objetos en el frame.")
                hablar_texto("Lo siento, no pude identificar ningún objeto.")
        except Exception as e:
            print(f"Error durante la detección de objetos: {e}")
            hablar_texto("Ocurrió un error durante la detección de objetos.")
    except Exception as e:
        print(f"Error general en detect_and_identify_object: {e}")
        hablar_texto("Lo siento, ocurrió un error inesperado durante la ejecución.")
    finally:
        # Liberar la cámara y limpiar recursos
        if 'cap' in locals() and cap.isOpened():
            cap.release()

def convertir_texto_a_equipo(text):
    """
    Convierte un texto en español al nombre de un equipo válido.
    """
    equipos_validos = {
        "barcelona": "FC Barcelona",
        "real madrid": "Real Madrid",
        "atletico de madrid": "Atlético de Madrid",
        "manchester united": "Manchester United",
        "liverpool": "Liverpool",
        "chelsea": "Chelsea",
        "juventus": "Juventus",
        "milan": "AC Milan",
        "boca juniors": "Boca Juniors",
        "river plate": "River Plate",
        "psg": "Paris Saint-Germain"
    }

    # Convertir a minúsculas y eliminar espacios extra
    text = text.lower().strip()

    # Buscar el nombre del equipo en el diccionario
    return equipos_validos.get(text, text)  # Si no está en el diccionario, devolver el texto original

def ask_for_favorite_team():
    while True:
        hablar_texto("Por favor, dime cuál es tu equipo favorito.")
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
        result = speech_recognizer.recognize_once_async().get()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            recognized_text = result.text.strip().lower()
            print(f"Texto reconocido: {recognized_text}")

            # Intentar convertir el texto reconocido al nombre del equipo
            team = convertir_texto_a_equipo(recognized_text)
            print(f"Equipo reconocido: {team}")

            # Confirmar antes de guardar
            if confirm_input("equipo favorito", team):
                return team
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No se pudo reconocer ninguna voz")
            hablar_texto("No he escuchado nada. Por favor, intenta nuevamente.")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Reconocimiento cancelado: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"Detalles del error: {cancellation_details.error_details}")
        else:
            print(f"Resultado desconocido: {result.reason}")

def ask_for_input(prompt_text):
    hablar_texto(prompt_text)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    result = speech_recognizer.recognize_once_async().get()
    
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        recognized_text = result.text.strip()
        print(f"Texto reconocido: {recognized_text}")
        return recognized_text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No se pudo reconocer ninguna voz")
        hablar_texto("No he escuchado nada. Por favor, intenta nuevamente.")
        return None
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Reconocimiento cancelado: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Detalles del error: {cancellation_details.error_details}")
        return None
    else:
        print(f"Resultado desconocido: {result.reason}")
        return None

def confirm_input(info_type, info_value):
    hablar_texto(f"Tu {info_type} es {info_value}, ¿correcto?")
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    result = speech_recognizer.recognize_once_async().get()
    
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        recognized_text = result.text.lower().strip()
        print(f"Confirmación reconocida: {recognized_text}")
        if any(palabra in recognized_text for palabra in ["sí", "si", "correcto","correct", "así es", "see"]):
            return True
        else:
            return False
    else:
        print("No se pudo obtener la confirmación.")
        hablar_texto("No he escuchado tu confirmación. Por favor, intentemos de nuevo.")
        return False

def capture_photo(name):
    hablar_texto("Voy a tomar una foto para tu perfil.")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        hablar_texto("Lo siento, no pude acceder a la cámara.")
        return None
    ret, frame = cap.read()
    cap.release()
    if ret:
        image_path = f"assets/img/{name}.jpg"
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        cv2.imwrite(image_path, frame)
        hablar_texto("Foto tomada correctamente.")
        return image_path
    else:
        print("Error: No se pudo capturar la imagen.")
        hablar_texto("Lo siento, no pude tomar la foto.")
        return None
def capturar_imagen():
    """Abre la cámara, captura una imagen automáticamente y proporciona un feedback verbal."""
    hablar_texto("Voy a tomar una foto en 3 segundos. Por favor, prepárate.")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        hablar_texto("Lo siento, no pude acceder a la cámara.")
        return None

    # Esperar unos segundos antes de capturar la imagen
    time.sleep(3)
    
    ret, frame = cap.read()
    cap.release()
    if ret:
        hablar_texto("Foto tomada correctamente.")
        return frame
    else:
        print("Error: No se pudo capturar la imagen.")
        hablar_texto("Lo siento, no pude tomar la foto.")
        return None


def registrar_persona():
    """Registra una nueva persona en la base de datos."""
    frame = capturar_imagen()
    _, buffer = cv2.imencode('.jpg', frame)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    while True: 
        nombre = ask_for_input("Dime su nombre : ")
        if nombre:
            if confirm_input("nombre", nombre):
                break
            else:
                hablar_texto("Entiendo. Intentemos de nuevo.")
        else:
            hablar_texto("No he podido entender su nombre. Por favor, intentemos de nuevo.")

            # Solicitar el equipo favorito por voz
    while True:
                equipo_favorito = ask_for_input("Ahora dime cuál es su equipo favorito.")
                if equipo_favorito:
                    if confirm_input("equipo favorito", equipo_favorito):
                        break
                    else:
                        hablar_texto("Entendido. Intentemos de nuevo.")
                else:
                    print("No he podido entender tu equipo favorito. Por favor, intentemos de nuevo.")


            # Tomar la foto

        
    guardar_persona(nombre, equipo_favorito, image_base64)
    guardar_log("registro", f"Nueva persona registrada: {nombre}")
    print("Persona registrada correctamente.")
    cargar_datos_entrenados()

        
def saludo_inicial(user_name):
    saludos = [
        f"Hola {user_name}, ¿en qué puedo ayudarte?",
        f"¡Hola {user_name}! ¿Cómo puedo asistirte hoy?",
        f"¡Buenas {user_name}! Dime, ¿cómo puedo ayudarte?",
        f"Hola {user_name}, estoy aquí para ayudarte. ¿Qué necesitas?",
        f"¡Hola {user_name}! Estoy lista para ayudarte. ¿En qué te puedo asistir?",
        f"Hola, {user_name}. ¿En qué puedo ser útil hoy?",
        f"¡Hey {user_name}! ¿Qué necesitas que haga por ti?",
        f"Hola {user_name}, espero que estés bien. ¿Cómo puedo ayudarte?",
        f"¡Buenos días {user_name}! ¿Cómo puedo asistirte en este momento?",
        f"¡Hola de nuevo, {user_name}! Estoy lista para lo que necesites.",
        f"Hola {user_name}, encantada de verte de nuevo. ¿Qué puedo hacer por ti?",
        f"Hola {user_name}, ¿cómo va tu día? ¿Te ayudo con algo?",
        f"¡Hola, {user_name}! ¿Qué tal si empezamos? Dime, ¿en qué puedo ayudarte?",
        f"Hola {user_name}, siempre estoy aquí para ayudarte. ¿Qué necesitas?",
        f"¡Hola {user_name}! Listos para trabajar juntos. ¿Qué necesitas de mí?",
        f"Hola {user_name}, ¿cómo estás hoy? Dime cómo puedo ayudarte.",
        f"¡Hola {user_name}! Estoy preparada para ayudarte. ¿Qué necesitas?",
        f"Hola {user_name}, me alegra escucharte. ¿Qué puedo hacer por ti hoy?",
        f"¡Hola, {user_name}! Dime en qué puedo asistirte, estoy lista.",
        f"Hola {user_name}, ¿qué necesitas que haga hoy? Estoy a tu disposición."
    ]
    texto = random.choice(saludos)
    hablar_texto(texto)

class DetectAndIdentifyObjectWorker(QObject):
    finished = pyqtSignal()

    def run(self):
        try:
            print("DetectAndIdentifyObjectWorker: Iniciando detect_and_identify_object")
            detect_and_identify_object()
        except Exception as e:
            print(f"DetectAndIdentifyObjectWorker: Exception occurred: {e}")
        finally:
            self.finished.emit()

class TimeWorker(QObject):
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()

    def run(self):
        try:
            from datetime import datetime
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            texto = f"La hora actual es {current_time}."
            print(texto)
            hablar_texto(texto)
        except Exception as e:
            print(f"TimeWorker: Exception occurred: {e}")
        finally:
            self.finished.emit()


class ResponderPreguntaWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, pregunta, conversation_history):
        super().__init__()
        self.pregunta = pregunta
        self.conversation_history = conversation_history

    def run(self):
        try:
            print("ResponderPreguntaWorker: Generando respuesta breve")
            from gpt.gpt_description import responder_pregunta, hablar_texto

            # Respuesta breve
            respuesta = responder_pregunta(self.pregunta, conversation_history=self.conversation_history)
            hablar_texto(respuesta)

            # Actualizar el historial de conversación
            self.conversation_history.append({"role": "user", "content": self.pregunta})
            self.conversation_history.append({"role": "assistant", "content": respuesta})

            # Preguntar si desea más información
            hablar_texto("¿Quieres más información sobre este tema? Por favor, di sí o no.")
            respuesta_usuario = self.escuchar_respuesta()

            if "sí" in respuesta_usuario or "si" in respuesta_usuario:
                hablar_texto("Dame un momento, te doy más información.")
                respuesta_detallada = responder_pregunta(self.pregunta, detallada=True, conversation_history=self.conversation_history)
                if respuesta_detallada:
                    hablar_texto(respuesta_detallada)
                    self.conversation_history.append({"role": "assistant", "content": respuesta_detallada})                    
                else:
                    hablar_texto("Lo siento, no puedo proporcionar más información en este momento.")
            elif "no" in respuesta_usuario:
                hablar_texto("De acuerdo. Si necesitas algo más, estoy aquí para ayudarte.")
            else:
                hablar_texto("No entendí tu respuesta. Por favor, intenta nuevamente si necesitas más ayuda.")
        except Exception as e:
            print(f"ResponderPreguntaWorker: Error al procesar: {e}")
            hablar_texto("Lo siento, no puedo responder en este momento.")
        finally:
            self.finished.emit()

    def escuchar_respuesta(self):
        """
        Escucha una respuesta del usuario y devuelve el texto reconocido.
        """
        from gpt.gpt_description import speechsdk
        import os

        # Configuración de Azure Speech
        speech_config = speechsdk.SpeechConfig(
            subscription=os.getenv('AZURE_SPEECH_KEY'),
            region=os.getenv('AZURE_REGION')
        )
        speech_config.speech_recognition_language = "es-ES"
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        print("Esperando respuesta del usuario...")
        result = speech_recognizer.recognize_once_async().get()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            recognized_text = result.text.strip().lower()
            print(f"Texto reconocido: '{recognized_text}'")
            return recognized_text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No se entendió la respuesta. Preguntando de nuevo...")
            return "No se entendió"
        else:
            print("Error en el reconocimiento de voz.")
            return "Error"


class AgregarPersonaWorker(QObject) :
    finished = pyqtSignal()
    

    def __init__(self, frame):
        super().__init__()
        self.frame = frame
    def run(self):
        try:         
            print("AgregarPersonaWorker: Iniciando registrar_persona")
            # Obtener ubicación por voz
            registrar_persona()
        except Exception as e:
            print(f"AgregarPersonaWorker: Exception occurred: {e}")
        finally:
            print("AgregarPersonaWorker: Finished run method")
            self.finished.emit()

class BarcelonaWorker(QObject):
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()

    def run(self):
        try:
            import requests
            from detection.mongodb import guardar_partido, guardar_log



            api_key = barcelona_api
            team_id = 81  # ID del FC Barcelona en la API
            base_url = 'https://api.football-data.org/v4/teams/'
            headers = {'X-Auth-Token': api_key}

            response = requests.get(f"{base_url}{team_id}/matches?status=SCHEDULED", headers=headers)
            data = response.json()

            if 'matches' in data and len(data['matches']) > 0:
                next_match = data['matches'][0]
                home_team = next_match['homeTeam']['name']
                away_team = next_match['awayTeam']['name']
                match_date = next_match['utcDate']
                
                # Convertir la fecha a un formato más legible
                from datetime import datetime
                match_datetime = datetime.strptime(match_date, "%Y-%m-%dT%H:%M:%SZ")
                match_date_str = match_datetime.strftime("%d de %B, %Y a las %H:%M")
                
                # Determinar si el Barcelona es el equipo local o visitante
                if next_match['homeTeam']['id'] == team_id:
                    texto = f"El próximo partido del Barcelona es contra {away_team} el {match_date_str}."
                else:
                    texto = f"El próximo partido del Barcelona es contra {home_team} el {match_date_str}."

                # Crear un diccionario con la información del partido
                partido_info = {
                    "equipo_local": home_team,
                    "equipo_visitante": away_team,
                    "fecha_partido": match_date_str,
                    "descripcion": texto
                }

                guardar_log("info", f"Partido guardado: {texto}")


                # Guardar en la base de datos
                guardar_partido(partido_info)
                print(texto)
                hablar_texto(texto)
            else:
                texto = "No se encontró información sobre el próximo partido del Barcelona."
                guardar_log("warning", texto)

                print(texto)
                hablar_texto(texto)
        except Exception as e:
            error_msg = f"BarcelonaWorker: Exception occurred: {e}"
            guardar_log("error", error_msg)
            print(error_msg)
        finally:
            self.finished.emit()


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

# Funciones para respuestas aleatorias
def respuesta_no_entendida():
    respuestas = [
        "Lo siento, no he entendido. ¿Podrías repetir, por favor?",
        "Disculpa, no capté eso. ¿Puedes decirlo de nuevo?",
        "Hmm, no estoy segura de lo que quieres decir. ¿Me lo repites?",
        "No entendí el comando. ¿Podrías repetirlo?",
        "Perdón, no te he entendido bien. ¿Puedes repetir?",
        "Lo siento, creo que no entendí bien. ¿Me lo explicas otra vez?",
        "No estoy segura de qué quieres decir. ¿Puedes intentarlo de nuevo?",
        "Perdón, parece que no te entendí. ¿Puedes repetirlo más claro?",
        "Hmm, creo que no entendí eso. ¿Puedes decirlo de otra manera?",
        "Disculpa, no logré entender. ¿Lo intentas de nuevo?",
        "Creo que no capté eso correctamente. ¿Puedes volver a intentarlo?",
        "No estoy muy segura de lo que intentas decir. ¿Podrías explicarlo otra vez?",
        "Lo siento, no estoy segura de lo que quisiste decir. ¿Me ayudas repitiéndolo?",
        "Hmm, no logré comprender eso. ¿Podrías repetirlo para mí?",
        "No entendí bien, ¿podrías intentarlo de otra forma?",
        "Creo que no entendí lo que dijiste. ¿Me ayudas repitiendo?",
        "Lo siento, eso no lo entendí bien. ¿Puedes decirlo de nuevo, por favor?",
        "Disculpa, parece que no capté bien lo que dijiste. ¿Puedes explicarlo otra vez?",
        "Creo que no escuché correctamente. ¿Me lo repites, por favor?",
        "Hmm, no entendí eso. ¿Puedes ser un poco más claro?"
    ]
    texto = random.choice(respuestas)
    hablar_texto(texto)

def preguntar_mas_ayuda():
    respuestas = [
        "¿Puedo ayudarte en algo más?",
        "¿Necesitas algo más?",
        "¿Hay algo más en lo que pueda asistirte?",
        "¿Te ayudo con algo más?",
        "¿Hay algo más que pueda hacer por ti?",
        "¿Quieres que haga algo más por ti?",
        "¿Puedo ayudarte con algo adicional?",
        "¿Tienes algo más en mente en lo que pueda ayudarte?",
        "¿Hay otra cosa con la que necesites ayuda?",
        "¿Te puedo apoyar en algo más?",
        "¿Hay algo más que pueda resolver por ti?",
        "¿En qué más puedo serte útil?",
        "¿Te queda alguna duda o necesitas algo más?",
        "¿Hay algo más que te gustaría saber o hacer?",
        "¿Puedo ofrecerte ayuda con algo más?"
    ]
    texto = random.choice(respuestas)
    hablar_texto(texto)

def saludo_inicial(user_name):
    saludos = [
         f"Hola, {user_name}. ¿Cómo te encuentras hoy? ¿En qué puedo asistirte?",
        f"¡Hola, {user_name}! Estoy lista para ayudarte, dime, ¿qué necesitas?",
        f"Hola {user_name}, ¿cómo puedo asistirte el día de hoy?",
        f"¡Hola, {user_name}! Espero que estés bien. ¿En qué puedo ayudarte?",
        f"¡Hola, {user_name}! Estoy aquí para ti. ¿Qué necesitas?",
        f"Buenas, {user_name}. ¿Cómo te puedo ayudar?",
        f"¡Hola, {user_name}! Estoy lista para lo que necesites. ¿Cómo te ayudo?",
        f"Hola, {user_name}. Cuéntame, ¿en qué puedo asistirte?",
        f"¡Hola {user_name}! Estoy disponible para ayudarte. ¿Qué necesitas?",
        f"Hola {user_name}, dime, ¿cómo puedo ayudarte hoy?",
        f"¡Hola! ¿Cómo estás? Estoy lista para ayudarte.",
        f"Hola {user_name}, ¿cómo va tu día? ¿Necesitas algo?",
        f"Hola, {user_name}. Estoy aquí para ayudarte. ¿Qué necesitas de mí?",
        f"¡Hola! ¿Qué puedo hacer por ti hoy?",
        f"Hola {user_name}, espero que todo esté bien. ¿En qué te puedo ayudar?"
    ]
    texto = random.choice(saludos)
    hablar_texto(texto)

def respuesta_afirmativa(user_name):
    respuestas = [
        f"Perfecto, {user_name}. ¿Qué deseas que haga?",
        f"Claro, dime en qué puedo ayudarte.",
        f"Por supuesto, ¿cómo puedo asistirte?",
        f"¡Excelente! ¿Qué necesitas?",
        f"Muy bien, estoy escuchando. ¿En qué te puedo ayudar?",
        "Claro, dime lo que necesitas.",
        "Perfecto, estoy lista para ayudarte. ¿Qué necesitas?",
        "Entendido, dime cómo puedo asistirte.",
        f"Por supuesto, {user_name}, estoy aquí para lo que necesites.",
        "De acuerdo, cuéntame qué necesitas.",
        f"¡Claro que sí, {user_name}! Estoy lista para ayudarte.",
        "Sin problema, ¿cómo te ayudo?",
        "Listo, dime cómo puedo asistirte.",
        "Perfecto, hablemos de lo que necesitas.",
        f"¡Adelante, {user_name}! ¿Qué hago por ti?"
    ]
    texto = random.choice(respuestas)
    hablar_texto(texto)

def respuesta_negativa(user_name):
    despedidas = [
        "De acuerdo, si necesitas algo más, solo dime 'Hola, Aitana'.",
        f"Entendido, estaré aquí si me necesitas. ¡Hasta luego, {user_name}!",
        "Muy bien, que tengas un buen día. Estoy aquí si necesitas ayuda.",
        "Está bien, no dudes en llamarme si me necesitas. ¡Hasta pronto!",
        "Claro, me avisas si puedo ayudarte en algo más. ¡Cuídate!",
        "De acuerdo, recuerda que estoy aquí si necesitas algo.",
        f"Entiendo, {user_name}. Si necesitas algo, no dudes en llamarme.",
        "Perfecto, estaré esperando si me necesitas.",
        f"Está bien, disfruta tu día. ¡Nos vemos luego, {user_name}!",
        "Claro que sí, estaré aquí para cuando necesites ayuda.",
        "Sin problema, ¡cuídate y nos vemos pronto!",
        "De acuerdo, no dudes en decir 'Hola, Aitana' cuando necesites algo.",
        f"¡Nos vemos, {user_name}! Estoy a tu disposición cuando quieras.",
        "Perfecto, cualquier cosa aquí estaré.",
        "Muy bien, que tengas un excelente día."
    ]
    texto = random.choice(despedidas)
    hablar_texto(texto)

# Función para hablar texto usando Azure
def hablar_texto(texto):
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    result = speech_synthesizer.speak_text_async(texto).get()

    # Verificar el resultado de la síntesis
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"Texto sintetizado correctamente: '{texto}'")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Síntesis cancelada: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Detalles del error: {cancellation_details.error_details}")

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    command_finished_signal = pyqtSignal(str)

    def __init__(self, conversation_history):
        super().__init__()
        self._run_flag = True
        self.conversation_history = conversation_history  # Añadido
        self.command_detected = None
        self.cap = cv2.VideoCapture(1)

        ###############################################################################################################
        #video_path = "/Users/dark0/Documents/Visor/A walk in Shibuya, Tokyo.webm"
        #self.cap = cv2.VideoCapture(video_path)

        # Obtener la tasa de fotogramas (FPS) del video
        #self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        #print(f"FPS del video: {self.fps}")

        #self.frame_delay = 1 / self.fps if self.fps > 0 else 0.066
        ###############################################################################################################

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.location = None  # Inicializar la ubicación en None
        self.current_command_name = None
        self.command_thread = None  # Hilo para ejecutar comandos
          # Añadido
        self.current_question = None  # Inicializa la variable pregunta
    

    def run(self):
        while self._run_flag:
            ret, frame = self.cap.read()
            if ret:
                # Emitir el frame para que sea mostrado en la GUI
                self.change_pixmap_signal.emit(frame)

                # Verificar si se ha recibido algún comando y no hay un comando en ejecución
                if self.command_detected and (self.command_thread is None or not self.command_thread.isRunning()):
                    print(f"VideoThread: Comando detectado '{self.command_detected}'")
                    if self.command_detected == "lee_texto":
                        print("VideoThread: Iniciando OCRWorker")
                        self.command_thread = QThread()
                        worker = OCRWorker(frame.copy())
                        worker.moveToThread(self.command_thread)
                        self.command_thread.started.connect(worker.run)
                        worker.finished.connect(self.command_thread.quit)
                        worker.finished.connect(worker.deleteLater)
                        self.command_thread.finished.connect(self.command_thread.deleteLater)
                        # Guardar el nombre del comando
                        self.current_command_name = "lee_texto"
                        # Conectar la señal de finalización del worker
                        worker.finished.connect(self.command_finished)
                        self.command_thread.start()
                        self.command_detected = None
                    elif self.command_detected == "describe_escena":
                        print("VideoThread: Iniciando DescriptionWorker")
                        self.command_thread = QThread()
                        worker = DescriptionWorker()
                        worker.moveToThread(self.command_thread)
                        self.command_thread.started.connect(worker.run)
                        worker.finished.connect(self.command_thread.quit)
                        worker.finished.connect(worker.deleteLater)
                        self.command_thread.finished.connect(self.command_thread.deleteLater)
                        # Guardar el nombre del comando
                        self.current_command_name = "describe_escena"
                        # Conectar la señal de finalización del worker
                        worker.finished.connect(self.command_finished)
                        self.command_thread.start()
                        self.command_detected = None
                    elif self.command_detected == "dime_la_hora":
                        print("VideoThread: Iniciando TimeWorker")
                        self.command_thread = QThread()
                        worker = TimeWorker()
                        worker.moveToThread(self.command_thread)
                        self.command_thread.started.connect(worker.run)
                        worker.finished.connect(self.command_thread.quit)
                        worker.finished.connect(worker.deleteLater)
                        self.command_thread.finished.connect(self.command_thread.deleteLater)
                        self.current_command_name = "dime_la_hora"
                        worker.finished.connect(self.command_finished)
                        self.command_thread.start()
                        self.command_detected = None
                    elif self.command_detected == "juega_barcelona":
                        print("VideoThread: Iniciando BarcelonaWorker")
                        self.command_thread = QThread()
                        worker = BarcelonaWorker()
                        worker.moveToThread(self.command_thread)
                        self.command_thread.started.connect(worker.run)
                        worker.finished.connect(self.command_thread.quit)
                        worker.finished.connect(worker.deleteLater)
                        self.command_thread.finished.connect(self.command_thread.deleteLater)
                        self.current_command_name = "juega_barcelona"
                        worker.finished.connect(self.command_finished)
                        self.command_thread.start()
                        self.command_detected = None
                    elif self.command_detected == "agregar_persona":
                        print("VideoThread: Iniciando AgregarPersonaWorker")
                        self.command_thread = QThread()
                        worker = AgregarPersonaWorker(frame)
                        worker.moveToThread(self.command_thread)
                        self.command_thread.started.connect(worker.run)
                        worker.finished.connect(self.command_thread.quit)
                        worker.finished.connect(worker.deleteLater)
                        self.command_thread.finished.connect(self.command_thread.deleteLater)
                        self.current_command_name = "agregar_persona"
                        worker.finished.connect(self.command_finished)
                        self.command_thread.start()
                        self.command_detected = None
                    elif self.command_detected == "responder_pregunta":
                        print("VideoThread: Iniciando ResponderPreguntaWorker")
                        self.command_thread = QThread()
                        worker = ResponderPreguntaWorker(self.current_question, self.conversation_history)  # Pasar el historial
                        worker.moveToThread(self.command_thread)  # Mueve el Worker al hilo
                        self.command_thread.started.connect(worker.run)  # Conecta el inicio del hilo con el método run del Worker
                        worker.finished.connect(self.command_thread.quit)  # Conecta la señal de finalización del Worker con el cierre del hilo
                        worker.finished.connect(worker.deleteLater)  # Elimina el Worker cuando termine
                        self.command_thread.finished.connect(self.command_thread.deleteLater)  # Elimina el hilo cuando termine
                        self.current_command_name = "responder_pregunta"
                        worker.finished.connect(self.command_finished)  # Notifica que el comando ha terminado
                        self.command_thread.start()
                        self.command_detected = None
                    elif self.command_detected == "detect_and_identify_object":
                        print("VideoThread: Iniciando DetectAndIdentifyObjectWorker")
                        self.command_thread = QThread()
                        worker = DetectAndIdentifyObjectWorker()
                        worker.moveToThread(self.command_thread)
                        self.command_thread.started.connect(worker.run)
                        worker.finished.connect(self.command_thread.quit)
                        worker.finished.connect(worker.deleteLater)
                        self.command_thread.finished.connect(self.command_thread.deleteLater)
                        self.current_command_name = "detect_and_identify_object"
                        worker.finished.connect(self.command_finished)
                        self.command_thread.start()
                        self.command_detected = None

                        
                #BORRAR SI SE USCA CV(1)        
                #time.sleep(self.frame_delay)
        
            else:
                print("Error al leer el frame de la cámara.")

    def command_finished(self):
        print(f"VideoThread: Command '{self.current_command_name}' finished.")
        self.command_finished_signal.emit(self.current_command_name)
        self.current_command_name = None
        self.command_thread = None

    def stop(self):
        self._run_flag = False
        self.wait()

    @pyqtSlot(str)
    def receive_command(self, command):
        print(f"VideoThread: Comando recibido '{command}'")
        self.command_detected = command

class OCRWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, frame):
        super().__init__()
        self.frame = frame

    def run(self):
        try:
            print("OCRWorker: Iniciando perform_ocr_and_read")
            perform_ocr_and_read(self.frame)
        except Exception as e:
            print(f"OCRWorker: Exception occurred: {e}")
        finally:
            self.finished.emit()

class DescriptionWorker(QObject):
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()

    def run(self):
        try:
            print("DescriptionWorker: Iniciando perform_detection_and_description")
            # Obtener ubicación por voz
            location = self.get_location_by_voice()
            print(f"DescriptionWorker: Ubicación obtenida: '{location}'")
            # Ejecutar perform_detection_and_description con la ubicación
            perform_detection_and_description(location)
        except Exception as e:
            print(f"DescriptionWorker: Exception occurred: {e}")
        finally:
            print("DescriptionWorker: Finished run method")
            self.finished.emit()

    def get_location_by_voice(self):
        # Decir "Dime la ubicación" usando síntesis de voz
        hablar_texto("Dime la ubicación")

        # Configurar el reconocimiento de voz
        speech_config_recognition = speechsdk.SpeechConfig(subscription=azure_speech_key, region=azure_region)
        speech_config_recognition.speech_recognition_language = "es-ES"
        # Ajustar los tiempos de espera si es necesario
        speech_config_recognition.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, '10000')  # 10 segundos
        speech_config_recognition.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, '1000')  # 2 segundos

        # Crear el reconocedor de voz
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config_recognition, audio_config=audio_config)

        print("Esperando la ubicación...")
        result = speech_recognizer.recognize_once_async().get()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            recognized_text = result.text.strip()
            print(f"Ubicación reconocida: {recognized_text}")
            return recognized_text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No se pudo reconocer ninguna voz")
            return "Ubicación no especificada"
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Reconocimiento cancelado: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"Detalles del error: {cancellation_details.error_details}")
            return "Ubicación no especificada"
        else:
            print(f"Resultado desconocido: {result.reason}")
            return "Ubicación no especificada"

class CommandListenerThread(QThread):
    command_signal = pyqtSignal(str)
    text_signal = pyqtSignal(str)

    def __init__(self, video_thread, user_name, conversation_history):
        super().__init__()
        self.video_thread = video_thread  # Guarda referencia de VideoThread
        self.user_name = user_name  # Guarda el nombre del usuario
        self.conversation_history = conversation_history
        self._run_flag = True
        self.state = "waiting_wake_word"
        
        
        self._run_flag = True
        self.state = "waiting_wake_word"  # Estados: waiting_wake_word, waiting_command, asking_more_help, waiting_response
        self.speech_config_recognition = speechsdk.SpeechConfig(subscription=azure_speech_key, region=azure_region)
        self.speech_config_recognition.speech_recognition_language = "es-ES"

    def run(self):
        while self._run_flag:
            print(f"Estado actual: {self.state}")
            if self.state == "waiting_wake_word":
                self.wait_for_wake_word()
            elif self.state == "waiting_command":
                self.listen_for_command()
            elif self.state == "command_executing":
                # Esperar a que el comando termine
                time.sleep(0.1)
            elif self.state == "asking_more_help":
                self.ask_more_help()
            elif self.state == "waiting_response":
                self.listen_for_response()
            else:
                time.sleep(0.1)

    def wait_for_wake_word(self):
        print("Esperando el 'wake word'...")
        result = self.recognize_once()

        # Imprimir siempre el texto reconocido, si lo hay
        print(f"Resultado del reconocimiento: {result.reason}")
        print(f"Texto reconocido: '{result.text}'")

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            recognized_text = result.text.lower().strip()
            if recognized_text == "":
                print("No se reconoció ninguna voz o el texto está vacío.")
            else:
                print(f"Texto reconocido: {recognized_text}")
                if any(frase in recognized_text for frase in ["hola aitana", "hola, aitana", "hola aitana", "hola, aitana"]):
                    print("Wake word detectado. Activando escucha de comandos.")
                    saludo_inicial(self.user_name)
                    self.state = "waiting_command"
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No se pudo reconocer ninguna voz")
            print(f"Detalles de NoMatch: {result.no_match_details}")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Reconocimiento cancelado: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"Detalles del error: {cancellation_details.error_details}")
        else:
            print(f"Resultado desconocido: {result.reason}")

    def listen_for_command(self):
        print("Esperando un comando...")
        result = self.recognize_once()

        print(f"Resultado del reconocimiento: {result.reason}")
        print(f"Texto reconocido: '{result.text}'")

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            recognized_text = result.text.lower().strip()
            if recognized_text:
                self.text_signal.emit(recognized_text)  # Emitir el texto reconocido
                if any(frase in recognized_text for frase in ["lee lo de la cámara", "lee lo de la camara", "lee esto", "¿Qué dice acá?", "¿Qué dice aquí?", "lee lo que hay aquí",  "lee lo que hay aca", "lee lo que hay en la cámara", "lee lo que hay en la camara", "¿Qué dice en la pantalla?","léelo", "¿Qué puedo leer aquí?", "¿Qué dice eso?", "¿Qué pone acá?", "¿Qué pone aquí?", "léeme esto", "léeme lo que aparece", "lee la pantalla", "léeme lo que ves", "¿Qué hay escrito aquí?", "¿Qué hay escrito acá?", "¿Puedes leer esto?", "¿Qué texto hay aquí?", "¿Puedes decirme qué dice?", "léeme el texto de la cámara", "¿Qué está escrito aquí?","¿Qué dice esto?", "lee, texto", "dime lo que pone aquí", "lee esto de la cámara", "léelo de la pantalla", "¿Qué palabras hay acá?","¿Qué texto aparece aquí?", "lee esta parte", "léeme lo que detectas", "¿Qué texto detectas?"]): 
                    print("Comando reconocido: 'lee_texto'")
                    self.command_signal.emit("lee_texto")
                    self.state = "command_executing"
                elif any(frase in recognized_text for frase in ["descríbeme la escena", "describeme la escena", "dime la escena", "describe la escena", "detalla la escena", "¿Qué esta pasando aquí?", "¿Qué hay en la cámara?", "¿Qué hay en la camara?", "¿Qué veo aquí?", "¿Qué veo aca?", "Escribe la escena"]):
                    print("Comando reconocido: 'describe_escena'")
                    self.command_signal.emit("describe_escena")
                    self.state = "command_executing"
                elif "dime la hora" in recognized_text or "Qué hora es? " in recognized_text:
                    print("Comando reconocido: 'dime_la_hora'")
                    self.command_signal.emit("dime_la_hora")
                    self.state = "command_executing"
                elif "barcelona" in recognized_text:
                    print("Comando reconocido: 'juega_barcelona'")
                    self.command_signal.emit("juega_barcelona")
                    self.state = "command_executing"
                elif any(agre in recognized_text for agre in ["agregar a persona", "agregar nuevo", "Agregar a persona", "agregar amigo", "agregar", "agregar persona", "agregar a una persona"]): 
                    print("Comando reconocido: 'agregar_persona'")
                    self.command_signal.emit("agregar_persona")
                    self.state = "command_executing"
                elif any(frase in recognized_text for frase in ["aitana", "dime", "explícame", "pregunta"]):
                    print("Comando reconocido: 'responder_pregunta'")
                    self.video_thread.current_question = recognized_text # Guarda la pregunta en VideoThread
                    self.command_signal.emit("responder_pregunta")
                    self.state = "command_executing"
                elif any(frase in recognized_text for frase in ["qué es esto", "que es esto", "qué tengo en la mano", "que tengo en la mano", "¿Qué tengo en la mano?"]):
                    print("Comando reconocido: 'detect_and_identify_object'")
                    self.video_thread.current_question = None  # No es una pregunta específica
                    self.command_signal.emit("detect_and_identify_object")
                    self.state = "command_executing"
                

                else:
                    respuesta_no_entendida()
            else:
                print("No se reconoció ninguna voz o el texto está vacío.")
                respuesta_no_entendida()
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No se pudo reconocer ninguna voz")
            hablar_texto("No he escuchado nada. ¿Puedes repetirlo?")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Reconocimiento cancelado: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"Detalles del error: {cancellation_details.error_details}")
        else:
            print(f"Resultado desconocido: {result.reason}")

    def ask_more_help(self):
        preguntar_mas_ayuda()
        self.state = "waiting_response"

    def listen_for_response(self):
        print("Esperando respuesta a '¿Te puedo ayudar en algo más, o No ?'...")
        result = self.recognize_once()

        print(f"Resultado del reconocimiento: {result.reason}")
        print(f"Texto reconocido: '{result.text}'")
        
        self.state = "waiting_command"
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            recognized_text = result.text.lower()
            if "no" in recognized_text:
                print("El usuario no desea más ayuda.")
                respuesta_negativa(self.user_name)
                self.state = "waiting_wake_word"
            else:
                print("El usuario desea continuar.")
                self.state = "waiting_command"
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No se pudo reconocer ninguna voz")
            hablar_texto("No he escuchado nada. Por favor, mencione en que lo puedo ayudar o para finalizar diga 'no'.")
            self.state = "waiting_command"  # Sigue esperando comandos
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Reconocimiento cancelado: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"Detalles del error: {cancellation_details.error_details}")
            self.state = "waiting_command"  # Sigue esperando comandos
        else:
            print(f"Resultado desconocido: {result.reason}")
            self.state = "waiting_command" # Sigue esperando comandos

    def recognize_once(self):
        # Configurar el reconocimiento de voz
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config_recognition)
        # Ajustar los tiempos de espera si es necesario
        self.speech_config_recognition.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, '10000')  # 10 segundos

        result = speech_recognizer.recognize_once_async().get()
        return result

    def on_command_finished(self, command_name):
        print(f"CommandListenerThread: Comando '{command_name}' ha finalizado.")
        self.state = "asking_more_help"
        
    def stop(self):
        self._run_flag = False
        self.wait()

# Función para realizar OCR y leer el texto detectado
def perform_ocr_and_read(frame):
    print("perform_ocr_and_read: Iniciando OCR")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    texto_detectado = pytesseract.image_to_string(gray, lang="spa")  # Usa 'spa' para español
    texto_detectado = texto_detectado.strip()
    print(f"Texto detectado: '{texto_detectado}'")
    if texto_detectado:
        texto_a_leer = f"Se detectó el siguiente texto: {texto_detectado}"
        print(texto_a_leer)
        hablar_texto(texto_a_leer)
    else:
        print("No se detectó ningún texto.")
        hablar_texto("No se detectó ningún texto.")

# Función para realizar detección de objetos y generar descripción
def perform_detection_and_description(ubicacion):
    try:

        cargar_datos_entrenados()

        global contexto
        contexto = {
            "objetos_finales": [],
            "gestos_detectados": set(),
            "poses_detectadas": set(),
            "personas_detectadas": set(),
            "equipos_detectados": {}
        }
        cap = cv2.VideoCapture(1)

        ###############################################################################################################
        #video_path = "/Users/dark0/Documents/Visor/A walk in Shibuya, Tokyo.webm"
        #cap = cv2.VideoCapture(video_path)
        ###############################################################################################################

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        total_frames = 0
        start_time = time.time()
        detected_object_counts = defaultdict(int)
        objects_per_frame = []  # Lista para almacenar conteos de objetos por fotograma

        while time.time() - start_time < 5:  # Capturar durante 5 segundos
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo capturar el frame.")
                break

            total_frames += 1

            frame_objects_counts = defaultdict(int)

            # Procesar detección de objetos
            yolo_results = model(frame, show=False)
            for result in yolo_results:
                for obj in result.boxes:
                    x1, y1, x2, y2 = map(int, obj.xyxy[0])
                    class_id = int(obj.cls)
                    confidence = obj.conf
                    label = f'{model.names[class_id]} {float(confidence):.2f}'
                    objeto_confianza = f'{model.names[class_id]} {float(confidence):.2f}'

                    if confidence >= confidence_threshold:
                        if model.names[class_id] == "person":
                            face_names_in_frame = detect_and_recognize_faces(frame)
                            if face_names_in_frame:
                                for name in face_names_in_frame:
                                    detected_object_counts[f"{name} {float(confidence):.2f}"] += 1                                    
                                    frame_objects_counts[name] += 1
                            else:
                                detected_object_counts[objeto_confianza] += 1
                                frame_objects_counts[model.names[class_id]] += 1
                        else:                    
                            detected_object_counts[objeto_confianza] += 1
                            frame_objects_counts[model.names[class_id]] += 1
                    else:
                        pass        

            objects_per_frame.append(frame_objects_counts)

            # Detección de gestos y poses
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_hands = hands.process(img_rgb)
            result_pose = pose.process(img_rgb)

            if result_hands.multi_hand_landmarks:
                for hand_landmarks in result_hands.multi_hand_landmarks:
                    if detectar_gesto_saludo(hand_landmarks):
                        print("Saludo detectado")
                        contexto["gestos_detectados"].add("Saludo")

            if result_pose.pose_landmarks:
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

        cap.release()
        
        # Mantener tu filtrado existente
        min_frames_for_object = total_frames * 0.1
        objetos_filtrados = filtrar_por_confianza(detected_object_counts, confidence_threshold=0.5)
        objetos_filtrados = {obj: count for obj, count in objetos_filtrados.items() if count >= min_frames_for_object}

        # Ahora, calcular el promedio de objetos detectados solo para los objetos filtrados
        average_object_counts = {}
        for obj_label in objetos_filtrados.keys():
            # Obtenemos el nombre del objeto sin la confianza
            obj_name = obj_label
            counts = [frame_counts.get(obj_name, 0) for frame_counts in objects_per_frame]
            if counts:
                average_count = sum(counts) / len(counts)
                average_object_counts[obj_name] = average_count

        # Preparar objetos_finales con los promedios y nombres en inglés
        objetos_finales = []
        for obj_label, avg_count in average_object_counts.items():
            avg_count_rounded = int(round(avg_count))
            if avg_count_rounded > 0:
                objetos_finales.append(f"{avg_count_rounded} {obj_label}")

        # Mostrar resultados
        print(f"Total de fotogramas procesados: {total_frames}")
        print(f"Objetos filtrados: {objetos_filtrados}")
        print(f"Conteo de objetos detectados: {dict(detected_object_counts)}")
        print("Objetos detectados:", objetos_finales)
        print("Gestos detectados:", set(contexto["gestos_detectados"]))

        descripcion = generar_descripcion(
            objetos_finales,
            ", ".join(contexto["gestos_detectados"]),
            ", ".join(contexto["poses_detectadas"]),
            ubicacion,
            contexto["personas_detectadas"],
            contexto["equipos_detectados"]
        )

        guardar_deteccion(
            objetos_detectados=list(detected_object_counts.keys()),
            gestos_detectados=contexto['gestos_detectados'],
            poses_detectadas=contexto['poses_detectadas'],
            ubicacion=ubicacion,
            descripcion=descripcion
        )
        print("Detección guardada correctamente.")

        guardar_log("info", "Detección y guardado completados")


        print(descripcion)
        hablar_texto(descripcion)
    except Exception as e:
        guardar_log("error", f"Error en la detección: {e}")
    
def detect_and_recognize_faces(frame):
    face_names_in_frame = []
    # Convertir el frame de BGR a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Encontrar todas las caras y codificaciones en el frame actual
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings_in_frame = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings_in_frame:
        # Comparar la cara con las caras conocidas
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconocido"

        # Usar la cara conocida con la menor distancia al nuevo encoding
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            # Agregar el nombre al contexto
            contexto["personas_detectadas"].add(name)
            # A veces, agregar el equipo favorito
            if random.choice([True, False]):
                team = known_face_teams.get(name, "")
                if team:
                    contexto["equipos_detectados"][name] = team
        face_names_in_frame.append(name)
    return face_names_in_frame            

class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        
        # Importar las funciones de mongodb.py
        from detection.mongodb import check_user_data, save_user_data, get_user_data

        # Variable para almacenar el nombre del usuario
        self.user_name = ""
        self.favorite_team = ""
        self.conversation_history = []

        if not check_user_data():
            # Solicitar el nombre por voz
            while True:
                name = ask_for_input("Por favor, dime tu nombre.")
                if name:
                    if confirm_input("nombre", name):
                        break
                    else:
                        hablar_texto("Entiendo. Intentemos de nuevo.")
                else:
                    print("No he podido entender tu nombre. Por favor, intentemos de nuevo.")

            # Solicitar el equipo favorito por voz
            while True:
                favorite_team = ask_for_input("Ahora dime cuál es tu equipo favorito.")
                if favorite_team:
                    if confirm_input("equipo favorito", favorite_team):
                        break
                    else:
                        hablar_texto("Entendido. Intentemos de nuevo.")
                else:
                    hablar_texto("No he podido entender tu equipo favorito. Por favor, intentemos de nuevo.")


            # Tomar la foto
            image_path = capture_photo(name)
            if image_path:
                # Guardar los datos en la base de datos
                save_user_data(name, favorite_team, image_path)
                self.user_name = name
                self.favorite_team = favorite_team
            else:
                print("Error al capturar la foto. Saliendo de la aplicación.")
                sys.exit(1)
        else:
            # Obtener los datos del usuario y saludarlo
            user_data = get_user_data()
            self.user_name = user_data['nombre']
            self.favorite_team = user_data['equipo_favorito']
            hablar_texto(f"Hola, {self.user_name}. Tu equipo favorito es {self.favorite_team}.")

        self.setWindowTitle("Detección de Gestos y Objetos")
        self.setGeometry(100, 100, 800, 600)

        # Widget central
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Layout
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Etiqueta para mostrar el video
        self.label = QLabel(self)
        self.layout.addWidget(self.label)

        # Etiqueta para mostrar el texto reconocido
        self.text_label = QLabel(self)
        self.layout.addWidget(self.text_label)

         # Pasar el historial a VideoThread
        self.thread = VideoThread(self.conversation_history)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        # Inicializar el hilo de comandos
        self.command_thread = CommandListenerThread(self.thread, self.user_name, self.conversation_history)
        self.command_thread.command_signal.connect(self.thread.receive_command)
        self.command_thread.text_signal.connect(self.update_recognized_text)
        self.thread.command_finished_signal.connect(self.command_thread.on_command_finished)
        self.command_thread.start()

        # Botón para salir de la aplicación
        self.button_exit = QPushButton("Salir", self)
        self.button_exit.clicked.connect(self.close_app)
        self.layout.addWidget(self.button_exit)

    def closeEvent(self, event):
        self.thread.stop()
        self.command_thread.stop()
        event.accept()

    def update_image(self, cv_img):
        """ Actualizar la imagen mostrada """
        qt_img = self.convert_cv_qt(cv_img)
        self.label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """ Convertir de OpenCV a QPixmap """
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        qt_image = qt_image.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(qt_image)

    def update_recognized_text(self, text):
        """ Actualizar la etiqueta con el texto reconocido """
        self.text_label.setText(f"Texto reconocido: {text}")

    def close_app(self):
        self.close()

class GestureDetectionApp(QApplication):
    def __init__(self, sys_argv):
        super().__init__(sys_argv)
        self.main_window = MainWindow()
        self.main_window.show()

if __name__ == '__main__':
    app = GestureDetectionApp(sys.argv)
    sys.exit(app.exec())