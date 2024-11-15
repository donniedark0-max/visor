import cv2
import pytesseract
import azure.cognitiveservices.speech as speechsdk

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
# Asegúrate de configurar la ruta de Tesseract en caso de que sea necesario.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configuración del servicio de Azure Speech
def sintetizar_voz(texto):
    # Reemplaza 'YourSubscriptionKey' y 'YourServiceRegion' con tus datos de Azure
    speech_key = ""
    service_region = ""
    
    # Configuración del cliente de Azure Speech
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    
    # Convierte texto a voz
    result = speech_synthesizer.speak_text_async(texto).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Texto sintetizado correctamente.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Error en la síntesis de voz: {}".format(cancellation_details.reason))

# Función para detectar texto en un fotograma específico
def detectar_texto_en_frame(frame):
    # Convertir a escala de grises y realizar OCR
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    texto_detectado = pytesseract.image_to_string(gray, lang="spa")
    print("Texto detectado:", texto_detectado)

    # Reproduce el texto detectado en voz
    if texto_detectado.strip():  # Solo llama a la función si hay texto
        sintetizar_voz(texto_detectado)

# Función principal para iniciar la cámara y esperar la captura
def iniciar_captura_con_ocr():
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

            # Mostrar el feed de video en la ventana
            cv2.imshow('Presiona espacio para capturar texto', frame)

            # Detectar cuando se presiona la barra espaciadora
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # La tecla de espacio
                detectar_texto_en_frame(frame)  # Captura y convierte el texto del frame actual

            # Salir con 'q'
            if key == ord('q'):
                break

    except Exception as e:
        print(f"Error en OCR: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Llamada a la función principal
if __name__ == "__main__":
    print("Iniciando captura de texto al presionar espacio...")
    iniciar_captura_con_ocr()