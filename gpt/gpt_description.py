import os
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import google.generativeai as genai
import random

# Cargar variables de entorno desde .env
load_dotenv()

# Configuración de la API de Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Configuración de la API de Azure Speech
azure_speech_key = os.getenv('AZURE_SPEECH_KEY')
azure_region = os.getenv('AZURE_REGION')

# Configuración de voz y servicio de Azure Speech
speech_config = speechsdk.SpeechConfig(subscription=azure_speech_key, region=azure_region)
speech_config.speech_synthesis_voice_name = "es-AR-ElenaNeural" 

# Función para generar descripción con Gemini
def generar_descripcion(objetos_finales, gestos_detectados):
    # Construcción de la descripción simple basada en la información disponible
    print(f"GEMINI CAPTO :{objetos_finales}")
    if objetos_finales and not gestos_detectados:
        prompt = f"En la escena se captaron {', '.join(objetos_finales)}. Describe lo que está ocurriendo con esos objetos."
    elif gestos_detectados and not objetos_finales:
        prompt = f"Se captaron los gestos: {gestos_detectados}. Describe la acción que podría estar ocurriendo."
    elif objetos_finales and gestos_detectados:
        prompt = f"Se captaron {', '.join(objetos_finales)} y los gestos: {gestos_detectados}. Describe la escena y lo que podría estar pasando."
    else:
        prompt = "No se captaron objetos ni gestos."
    print(f"Prompt enviado a Gemini: {prompt}")
    try:
        # Generación de contenido con Gemini
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=10,  
                temperature=0.2  
            ),
        )
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
