import sys
import os
import cv2
import threading
import time
import numpy as np
import mediapipe as mp
import math
from collections import defaultdict
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QInputDialog, QLabel
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QImage, QPixmap
from ultralytics import YOLO
from gpt.gpt_description import generar_descripcion, hablar_texto
import pytesseract
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from functools import partial

# Cargar variables de entorno desde .env
load_dotenv()

# Configuración de la API de Azure Speech
azure_speech_key = os.getenv('AZURE_SPEECH_KEY')
azure_region = os.getenv('AZURE_REGION')

speech_config = speechsdk.SpeechConfig(subscription=azure_speech_key, region=azure_region)
speech_config.speech_synthesis_voice_name = "es-AR-ElenaNeural"

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# Inicializar MediaPipe y YOLO
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Modelo YOLO
model = YOLO('yolov10s.pt')

# Variables globales
capturing = True
command_detected = None
texto_detectado = ""
ubicacion = ""

confidence_threshold = 0.5  # Umbral de confianza

contexto = {
    "objetos_finales": [],
    "gestos_detectados": set(),
    "poses_detectadas": set()
}


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
    command_signal = pyqtSignal(str)
    command_finished_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.command_detected = None
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.location = None  # Inicializar la ubicación en None
        self.command_thread = None  # Hilo para ejecutar comandos

    def run(self):
        while self._run_flag:
            ret, frame = self.cap.read()
            if ret:
                # Emitir el frame para que sea mostrado en la GUI
                self.change_pixmap_signal.emit(frame)

                # Verificar si se ha recibido algún comando y no hay un comando en ejecución
                if self.command_detected and (self.command_thread is None or not self.command_thread.isRunning()):
                    if self.command_detected == "lee_texto":
                        self.command_thread = QThread()
                        worker = OCRWorker(frame)
                        worker.moveToThread(self.command_thread)
                        self.command_thread.started.connect(worker.run)
                        worker.finished.connect(self.command_thread.quit)
                        worker.finished.connect(worker.deleteLater)
                        self.command_thread.finished.connect(self.command_thread.deleteLater)
                        # Conectar la señal de finalización del worker
                        worker.finished.connect(partial(self.command_finished, "lee_texto"))
                        self.command_thread.start()
                        self.command_detected = None
                    elif self.command_detected == "describe_escena":
                        self.command_thread = QThread()
                        worker = DescriptionWorker()
                        worker.moveToThread(self.command_thread)
                        self.command_thread.started.connect(worker.run)
                        worker.finished.connect(self.command_thread.quit)
                        worker.finished.connect(worker.deleteLater)
                        self.command_thread.finished.connect(self.command_thread.deleteLater)
                        # Conectar la señal de finalización del worker
                        worker.finished.connect(partial(self.command_finished, "describe_escena"))
                        self.command_thread.start()
                        self.command_detected = None

        # Liberar la cámara al finalizar
        self.cap.release()

    def command_finished(self, command_name):
        print(f"VideoThread: Command '{command_name}' finished.")
        self.command_finished_signal.emit(command_name)
    def stop(self):
        self._run_flag = False
        self.wait()

    # Agregar este método a la clase VideoThread
    def set_command(self, command):
        self.command_detected = command

class OCRWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, frame):
        super().__init__()
        self.frame = frame

    def run(self):
        perform_ocr_and_read(self.frame)
        self.finished.emit()

class DescriptionWorker(QObject):
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()

    def run(self):
        # Obtener ubicación por voz
        location = self.get_location_by_voice()
        # Ejecutar perform_detection_and_description con la ubicación
        perform_detection_and_description(location)
        self.finished.emit()

    def get_location_by_voice(self):
        # Decir "Dime la ubicación" usando síntesis de voz
        hablar_texto("Dime la ubicación")

        # Configurar el reconocimiento de voz
        speech_config_recognition = speechsdk.SpeechConfig(subscription=azure_speech_key, region=azure_region)
        speech_config_recognition.speech_recognition_language = "es-ES"
        # Ajustar los tiempos de espera si es necesario
        speech_config_recognition.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, '10000')  # 10 segundos
        speech_config_recognition.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, '2000')  # 2 segundos

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

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.state = "waiting_wake_word"  # Estados: waiting_wake_word, waiting_command, asking_more_help, waiting_response
        self.speech_config_recognition = speechsdk.SpeechConfig(subscription=azure_speech_key, region=azure_region)
        self.speech_config_recognition.speech_recognition_language = "es-ES"

    def run(self):
       while self._run_flag:
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
            recognized_text = result.text.lower()
            if recognized_text.strip() == "":
                print("No se reconoció ninguna voz o el texto está vacío.")
            else:
                print(f"Texto reconocido: {recognized_text}")
                if "hola elara" in recognized_text or "hola, helara" in recognized_text or "hola lara" in recognized_text  or "hola helara" in recognized_text or "hola, elara" in recognized_text:
                    print("Wake word detectado. Activando escucha de comandos.")
                    hablar_texto("Hola, en qué puedo ayudarte?")
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
            recognized_text = result.text.lower()
            if recognized_text.strip() == "":
                print("No se reconoció ninguna voz o el texto está vacío.")
            else:
                self.text_signal.emit(recognized_text)  # Emitir el texto reconocido
                if "lee lo de la cámara" in recognized_text or "lee lo de la camara" in recognized_text:
                    self.command_signal.emit("lee_texto")
                    self.state = "command_executing"
                elif ("descríbeme la escena" in recognized_text or "describeme la escena" in recognized_text or
                    "dime la escena" in recognized_text or "escribe la escena" in recognized_text or
                    "ponme la escena" in recognized_text or "llévame la escena" in recognized_text):
                    self.command_signal.emit("describe_escena")
                    self.state = "command_executing"
                else:
                    hablar_texto("No he entendido el comando. ¿Puedes repetirlo?")
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
        hablar_texto("¿Te puedo ayudar en algo más?")
        self.state = "waiting_response"

    def listen_for_response(self):
        print("Esperando respuesta a '¿Te puedo ayudar en algo más?'...")
        result = self.recognize_once()

        print(f"Resultado del reconocimiento: {result.reason}")
        print(f"Texto reconocido: '{result.text}'")

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            recognized_text = result.text.lower()
            if "sí" in recognized_text or "si" in recognized_text:
                print("El usuario desea más ayuda.")
                hablar_texto("¿Qué deseas que haga?")
                self.state = "waiting_command"
            elif "no" in recognized_text:
                print("El usuario no desea más ayuda.")
                hablar_texto("De acuerdo. Si necesitas algo más, dime 'Hola, Elara'.")
                self.state = "waiting_wake_word"
            else:
                hablar_texto("No he entendido tu respuesta. Por favor, di 'sí' o 'no'.")
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No se pudo reconocer ninguna voz")
            hablar_texto("No he escuchado nada. Por favor, di 'sí' o 'no'.")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Reconocimiento cancelado: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"Detalles del error: {cancellation_details.error_details}")
        else:
            print(f"Resultado desconocido: {result.reason}")

    def recognize_once(self):
        # Configurar el reconocimiento de voz
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config_recognition)
        # Ajustar los tiempos de espera si es necesario
        self.speech_config_recognition.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, '10000')  # 10 segundos

        result = speech_recognizer.recognize_once_async().get()
        return result
    
    def on_command_finished(self, command_name):
        print(f"Comando '{command_name}' ha finalizado.")
        self.state = "asking_more_help"

    def stop(self):
        self._run_flag = False
        self.wait()

# Función para realizar OCR y leer el texto detectado
def perform_ocr_and_read(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    texto_detectado = pytesseract.image_to_string(gray, lang="spa")  # Usa 'spa' para español
    texto_detectado = texto_detectado.strip()
    if texto_detectado:
        texto_a_leer = f"Se detectó el siguiente texto: {texto_detectado}"
        print(texto_a_leer)
        hablar_texto(texto_a_leer)
    else:
        print("No se detectó ningún texto.")
        hablar_texto("No se detectó ningún texto.")

# Función para realizar detección de objetos y generar descripción
def perform_detection_and_description(ubicacion):
    global contexto
    contexto = {
        "objetos_finales": [],
        "gestos_detectados": set(),
        "poses_detectadas": set()
    }
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    total_frames = 0
    start_time = time.time()
    detected_object_counts = defaultdict(int)

    while time.time() - start_time < 5:  # Capturar durante 5 segundos
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el frame.")
            break

        total_frames += 1

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
                    detected_object_counts[objeto_confianza] += 1
                    # Opcional: dibujar bounding boxes
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Detección de gestos y poses
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_hands = hands.process(img_rgb)
        result_pose = pose.process(img_rgb)

        if result_hands.multi_hand_landmarks:
            for hand_landmarks in result_hands.multi_hand_landmarks:
                # mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if detectar_gesto_saludo(hand_landmarks):
                    print("Saludo detectado")
                    contexto["gestos_detectados"].add("Saludo")

        if result_pose.pose_landmarks:
            # mp_draw.draw_landmarks(frame, result_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
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

        # Mostrar el frame (opcional)
        # cv2.imshow('Detección de Objetos y Gestos', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    # cv2.destroyAllWindows()

    # Procesar y generar la descripción
    min_frames_for_object = total_frames * 0.1
    objetos_filtrados = filtrar_por_confianza(detected_object_counts, confidence_threshold=0.5)
    objetos_finales = [obj for obj, count in objetos_filtrados.items() if count >= min_frames_for_object]

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
        ubicacion
    )
    print(descripcion)
    hablar_texto(descripcion)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

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

        # Inicializar el hilo de video
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        # Inicializar el hilo de comandos
        self.command_thread = CommandListenerThread()
        self.command_thread.command_signal.connect(self.handle_command)
        self.command_thread.text_signal.connect(self.update_recognized_text)
        self.command_thread.start()

        self.thread.command_finished_signal.connect(self.command_thread.on_command_finished)
       
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

    def handle_command(self, command):
        self.thread.set_command(command)

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

