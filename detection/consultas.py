import os
import time
from pymongo import MongoClient
from dotenv import load_dotenv
import certifi

# Cargar variables de entorno desde .env
load_dotenv()

# Ruta absoluta para la carpeta "consultas"
BASE_DIR = "/Users/dark0/Documents/Visor"
CONSULTAS_DIR = os.path.join(BASE_DIR, "consultas")

# Crear carpeta "consultas" si no existe
if not os.path.exists(CONSULTAS_DIR):
    os.makedirs(CONSULTAS_DIR)
    print(f"Carpeta 'consultas' creada en: {CONSULTAS_DIR}")
else:
    print(f"Carpeta 'consultas' ya existe en: {CONSULTAS_DIR}")

MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
db = client['visor_database']
partidos_collection = db['partidos']
detecciones_collection = db['detecciones']

def guardar_consulta_en_archivo(consulta_id, tipo, contenido):
    """Guardar una consulta en un archivo si no existe ya."""
    # Ruta del archivo donde se guardará la consulta
    filename = os.path.join(CONSULTAS_DIR, f"{consulta_id}_{tipo}.txt")

    # Verificar si el archivo ya existe antes de crear uno nuevo
    if os.path.exists(filename):
        print(f"Archivo ya existe para la consulta con ID: {consulta_id}, ignorando.")
        return

    print(f"Creando archivo para consulta con ID: {consulta_id}")

    try:
        with open(filename, "w") as file:
            file.write(f"ID: {consulta_id}\nTipo: {tipo}\nContenido:\n{contenido}\n")
        print(f"Consulta guardada en archivo: {filename}")
    except Exception as e:
        print(f"Error al crear el archivo para la consulta: {e}")

def obtener_partidos():
    """Obtener y guardar los partidos en archivos si no existen."""
    partidos = partidos_collection.find().sort("timestamp", -1)
    for partido in partidos:
        partido_id = partido["_id"]
        contenido = partido.get("partido", "No hay información")
        timestamp = time.ctime(partido["timestamp"])
        guardar_consulta_en_archivo(partido_id, "partido", f"{timestamp} - {contenido}")

def obtener_detecciones():
    """Obtener y guardar las detecciones en archivos si no existen."""
    detecciones = detecciones_collection.find().sort("timestamp", -1)
    for deteccion in detecciones:
        deteccion_id = deteccion["_id"]
        objetos = deteccion.get("objetos", [])
        gestos = deteccion.get("gestos", [])
        poses = deteccion.get("poses", [])
        ubicacion = deteccion.get("ubicacion", "No especificada")
        timestamp = time.ctime(deteccion["timestamp"])

        contenido = (
            f"Objetos: {', '.join(objetos)}\n"
            f"Gestos: {', '.join(gestos)}\n"
            f"Poses: {', '.join(poses)}\n"
            f"Ubicación: {ubicacion}\n"
            f"Timestamp: {timestamp}"
        )
        guardar_consulta_en_archivo(deteccion_id, "deteccion", contenido)

if __name__ == '__main__':
    print("Elige una opción:")
    print("1. Obtener todos los partidos")
    print("2. Obtener todas las detecciones")

    opcion = input("Selecciona una opción: ")
    if opcion == "1":
        obtener_partidos()
    elif opcion == "2":
        obtener_detecciones()
    else:
        print("Opción no válida")