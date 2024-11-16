import os
import time
from pymongo import MongoClient
from dotenv import load_dotenv
import certifi

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Conectar a MongoDB usando el URI de tu .env
MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())

# Crear o seleccionar la base de datos
db = client['visor_database']

# Crear o seleccionar las colecciones
partidos_collection = db['partidos']
detecciones_collection = db['detecciones']
logs_collection = db['logs']

print("Conectado a MongoDB correctamente.")

def guardar_partido(partido):
    partido['timestamp'] = time.time()
    result = partidos_collection.insert_one(partido)
    # Aquí usamos la clave 'descripcion' en lugar de 'partido'
    guardar_log("barcelona", f"Partido guardado: {partido['descripcion']}")
    print(f"Partido guardado con ID: {result.inserted_id}")

def guardar_deteccion(objetos_detectados, gestos_detectados, poses_detectadas, ubicacion, descripcion):
    deteccion = {
        "objetos": objetos_detectados,
        "gestos": list(gestos_detectados),
        "poses": list(poses_detectadas),
        "ubicacion": ubicacion,
        "timestamp": time.time(),
        "descripcion": descripcion
    }
    result = detecciones_collection.insert_one(deteccion)
    guardar_log("gestos", f"Detección guardada: {deteccion}")
    print(f"Detección guardada con ID: {result.inserted_id}")    

def guardar_log(tipo, mensaje):
    log = {
        "tipo": tipo,
        "mensaje": mensaje,
        "timestamp": time.time()
    }
    result = logs_collection.insert_one(log)
    print(f"Log guardado con ID: {result.inserted_id}")    