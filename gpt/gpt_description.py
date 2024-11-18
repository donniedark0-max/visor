import os
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import google.generativeai as genai
import json
import random

# Cargar variables de entorno desde .env
load_dotenv()

# Configuración de la API de Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Configuración de la API de Azure Speech
azure_speech_key = os.getenv('AZURE_SPEECH_KEY')
azure_region = os.getenv('AZURE_REGION')


speech_config = speechsdk.SpeechConfig(subscription=azure_speech_key, region=azure_region)
speech_config.speech_synthesis_voice_name = "es-AR-ElenaNeural"
 

# Función para generar descripción con Gemini
def generar_descripcion(objetos_finales, gestos_detectados, poses_detectadas, ubicacion, personas_detectadas=None, equipos_detectados=None):
    # Construcción de la descripción simple basada en la información disponible
    print(f"GEMINI CAPTO:\n - Objetos: {objetos_finales}\n  - Gestos: {gestos_detectados}\n - Poses: {poses_detectadas}\n - Ubicación: {ubicacion}")
     # Estructura de objetos en JSON

    # Procesar personas detectadas y equipos
    person_descriptions = []
    if personas_detectadas:
        for nombre in personas_detectadas:
            descripcion_persona = f"{nombre} está en la escena."
            # Decide aleatoriamente si incluir el equipo favorito
            if equipos_detectados and nombre in equipos_detectados and random.choice([True, False]):
                equipo = equipos_detectados[nombre]
                frase_equipo = obtener_frase_equipo(equipo)
                descripcion_persona += f" Su equipo favorito es {equipo}. {frase_equipo}"
            person_descriptions.append(descripcion_persona)
    
    # Crear el JSON con objetos, gestos y ubicación
    prompt = {
        "scene": {
            "objects": objetos_finales, 
            "people": person_descriptions,
            "gestures": list(gestos_detectados) if gestos_detectados else [],
            "location": ubicacion,
            "poses": poses_detectadas
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
                    "Ian y otra persona están en la sala, ambos parecen conversar animadamente. El equipo favorito de Ian es FC Barcelona. ¡Visca el Barça!",
                    "María está en la cocina preparando café. Su equipo favorito es Real Madrid. ¡Hala Madrid!"
                    "Una persona sostiene una botella en la cocina, aparentemente sirviendo alguna bebida.",
                    "Dos personas están sentadas en el sofá de la sala, cada una con un libro en sus manos, muy concentradas en su lectura."
                ],
                "bad": [
                    "Se ha detectado una persona llamada Ian y una botella en la escena.",
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
                max_output_tokens=60,  # Aumenta el número de tokens si deseas una descripción más detallada
                temperature=0.5  # Ajusta la temperatura para controlar la creatividad
            ),
        )
        descripcion = response.text
        descripcion = descripcion.replace("Se ha detectado", "").replace("Está presente", "")
        return descripcion

    except Exception as e:
        print(f"Error al obtener texto de Gemini: {str(e)}")
        return None

def obtener_frase_equipo(equipo):
    frases_equipos = {
        "FC Barcelona": "¡Visca el Barça!",
        "Real Madrid": "¡Hala Madrid!",
        "Atlético de Madrid": "¡Aúpa Atleti!",
        "Manchester United": "¡Glory Glory Man United!",
        "Liverpool": "¡You'll Never Walk Alone!",
        "Chelsea": "¡Come on you Blues!",
        "Juventus": "¡Fino Alla Fine!",
        "AC Milan": "¡Forza Milan!",
        "Boca Juniors": "¡Dale Boca!",
        "River Plate": "¡Vamos Millonario!",
        "Paris Saint-Germain": "¡Allez Paris!",
        "Alianza Lima": "¡Arriba Alianza! y, abajo la U",
    }
    return frases_equipos.get(equipo, "")

# Función para convertir texto a voz usando Azure y reproducirlo en tiempo real
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
def responder_pregunta(pregunta, detallada=False):
    """
    Genera una respuesta breve o detallada según el parámetro `detallada`.
    """
    print(f"Pregunta recibida: {pregunta}")
    tipo_respuesta = "breve" if not detallada else "detallada"

    prompt = {
        "instructions": {
            "role": "Eres un asistente virtual que responde preguntas en español, el lugar desde donde se te pregunta todo es Perú, no lo menciones implicitamente ",
            "examples": {
                "good": [
                    "Pregunta: ¿Cuál es la capital de Francia?\nRespuesta: La capital de Francia es París.",
                    "Pregunta: ¿Por qué el cielo es azul?\nRespuesta: Por la dispersión de la luz azul en la atmósfera.",
                    "Pregunta: ¿Desde cuando no vamos al mundial?\nRespuesta: Perú no va al mundial desde el mundial en Rusia 2018."
                ]
            }
        },
        "question": pregunta,
        "output_format": {
            "style": "Responde con detalles extensos y profundos." if detallada else "Responde en 1-2 frases como máximo.",
            "tone": "Amigable y profesional",
            "length": "Extensa" if detallada else "Breve y concisa"
        }
    }

    prompt_json = json.dumps(prompt)
    print(f"JSON Prompt enviado a Gemini ({tipo_respuesta}): {prompt_json}")

    try:
        model = genai.GenerativeModel("gemini-1.5-flash-002")
        response = model.generate_content(
            prompt_json,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=250 if detallada else 50,  # Tokens para respuestas largas o cortas
                temperature=0.7 if detallada else 0.5  # Mayor creatividad para respuestas detalladas
            )
        )
        respuesta = response.text.strip()
        print(f"Respuesta generada por Gemini ({tipo_respuesta}): {respuesta}")
        return respuesta

    except Exception as e:
        print(f"Error al generar respuesta con Gemini: {str(e)}")
        return "Lo siento, no puedo responder en este momento."


