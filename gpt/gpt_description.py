import os
import requests
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import google.generativeai as genai

# Cargar variables de entorno desde .env
load_dotenv()

# Configuración de la API de Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Configuración de la API de Azure Speech
azure_speech_key = os.getenv('AZURE_SPEECH_KEY')
azure_region = os.getenv('AZURE_REGION')

# Configuración de voz y servicio de Azure Speech
speech_config = speechsdk.SpeechConfig(subscription=azure_speech_key, region=azure_region)
speech_config.speech_synthesis_voice_name = "es-AR-ElenaNeural"  # Voz específica

# Función para obtener texto desde Gemini API
def obtener_texto_desde_gemini(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error al obtener texto de Gemini: {str(e)}")
        return None

# Función para convertir texto a voz usando Azure y reproducirlo en tiempo real
def hablar_texto(texto):
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
    result = speech_synthesizer.speak_text_async(texto).get()

    # Verificar el resultado de la síntesis
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"Texto sintetizado correctamente: '{texto}'")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Síntesis cancelada: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Detalles del error: {cancellation_details.error_details}")

# Ejemplo de uso: Escribir un prompt, obtener respuesta de Gemini y convertir a voz
prompt = "Describe una escena en un parque soleado."
texto_generado = obtener_texto_desde_gemini(prompt)

if texto_generado:
    print(f"Texto generado por Gemini: {texto_generado}")
    hablar_texto(texto_generado)
else:
    print("No se pudo generar texto desde Gemini.")