import os
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import google.generativeai as genai
import json

# Cargar variables de entorno desde .env
load_dotenv()

# Configuración de la API de Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Configuración de la API de Azure Speech
azure_speech_key = os.getenv('AZURE_SPEECH_KEY')
azure_region = os.getenv('AZURE_REGION')

speech_config = speechsdk.SpeechConfig(subscription=azure_speech_key, region=azure_region)
speech_config.speech_synthesis_voice_name = "es-MX-JorgeNeural" 

<<<<<<< HEAD
    # Configuración de voz y servicio de Azure Speech
    speech_config = speechsdk.SpeechConfig(subscription=azure_speech_key, region=azure_region)
    speech_config.speech_synthesis_voice_name = "es-AR-ElenaNeural" 

    AUDIO_FORMATS = {
        # Máxima calidad - mejor para uso local
        "HIGH_QUALITY": speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm,
        
        # Calidad media - buen balance entre calidad y tamaño
        "BALANCED": speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3,
       
    }
    
    # Configurar el formato de audio seleccionado
    audio_format = AUDIO_FORMATS.get(formato_audio, AUDIO_FORMATS["HIGH_QUALITY"])
    speech_config.set_speech_synthesis_output_format(audio_format)
    
    return speech_config
=======
>>>>>>> 91ddb9da344142e6ef7addc36e0cf0891325469d

# Función para generar descripción con Gemini
def generar_descripcion(objetos_finales, gestos_detectados, ubicacion):
    # Construcción de la descripción simple basada en la información disponible
    print(f"GEMINI CAPTO:\n - Objetos: {objetos_finales}\n  - Gestos: {gestos_detectados}\n - Ubicación: {ubicacion}")
     # Estructura de objetos en JSON

    
    # Crear el JSON con objetos, gestos y ubicación
    prompt = {
        "scene": {
            "objects": objetos_finales, 
            "gestures": list(gestos_detectados) if gestos_detectados else [],
            "location": ubicacion
        },
        "instructions": {
            "role": "Eres un asistente especializado en describir escenas para personas ciegas, actuando como sus ojos. Debes crear descripciones naturales y fluidas que ayuden a visualizar la escena en tiempo real.",
            "style_guidelines": [
                "Usa lenguaje natural y conversacional",
                "Describe las posibles interacciones entre objetos y personas",
                "Menciona la disposición espacial de los elementos",
                "Incluye detalles relevantes que ayuden a construir una imagen mental sin especular demasiado",
                "Evita frases como 'veo a', 'hay', 'se encuentra' o 'está presente'",
                "Prioriza verbos de acción y descripciones dinámicas",
                "Prioriza descripciones útiles para una persona ciega",

            ],
            "examples": {
                "good": [
                    "Una persona sostiene una botella en la cocina, aparentemente sirviendo alguna bebida.",
                    "Dos personas están sentadas en el sofá de la sala, cada una con un libro en sus manos, muy concentradas en su lectura."
                ],
                "bad": [
                    "Se ha detectado una persona y una botella en la escena.",
                    "Hay dos personas y dos libros presentes en la ubicación."
                ]
            }
        },
        "output_format": {
            "style": "Genera una descripción fluida y natural en español, usando 1-2 oraciones conectadas lógicamente.",
            "tone": "Amigable y descriptivo, como si estuvieras contándole a un amigo lo que ves.",
            "length": "Conciso pero informativo"

        }
    }
    # Convertir a JSON
    prompt_json = json.dumps(prompt)
    print(f"JSON Prompt enviado a Gemini: {prompt_json}")
    
    try:
        # Generación de contenido con Gemini
        model = genai.GenerativeModel("gemini-1.5-flash-002")
        response = model.generate_content(
            prompt_json,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=100,  # Aumenta el número de tokens si deseas una descripción más detallada
                temperature=0.5  # Ajusta la temperatura para controlar la creatividad
            ),
        )
        descripcion = response.text
        descripcion = descripcion.replace("Se ha detectado", "").replace("Está presente", "")
        return descripcion

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

    
    