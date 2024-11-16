import os
import time
from pymongo import MongoClient
from dotenv import load_dotenv
import certifi

# Cargar variables de entorno desde .env
load_dotenv()

# Ruta absoluta para la carpeta "logs"
BASE_DIR = "/Users/dark0/Documents/Visor"
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Crear carpeta "logs" si no existe
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
    print(f"Carpeta 'logs' creada en: {LOGS_DIR}")
else:
    print(f"Carpeta 'logs' ya existe en: {LOGS_DIR}")

MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
db = client['visor_database']
logs_collection = db['logs']

def guardar_log(tipo, mensaje):
    """Guardar un log en la base de datos y en un archivo."""
    log = {
        "tipo": tipo,
        "mensaje": mensaje,
        "timestamp": time.time()
    }
    result = logs_collection.insert_one(log)
    log_id = result.inserted_id

    # Ruta del archivo donde se guardará el log
    filename = os.path.join(LOGS_DIR, f"{log_id}_{tipo}.txt")

    print(f"Intentando crear archivo: {filename}")

    try:
        # Intentar escribir en el archivo
        with open(filename, "w") as file:
            file.write(f"ID: {log_id}\nTipo: {tipo}\nMensaje: {mensaje}\nTimestamp: {time.ctime(log['timestamp'])}\n")
        print(f"Log guardado en archivo: {filename}")
    except Exception as e:
        print(f"Error al crear el archivo: {e}")

def guardar_log_en_archivo(log):
    """Guardar un log leído desde la base de datos en un archivo si no existe ya."""
    log_id = log["_id"]
    tipo = log["tipo"]
    mensaje = log["mensaje"]
    timestamp = log["timestamp"]

    # Ruta del archivo donde se guardará el log
    filename = os.path.join(LOGS_DIR, f"{log_id}_{tipo}.txt")

    # Verificar si el archivo ya existe antes de crear uno nuevo
    if os.path.exists(filename):
        print(f"Archivo ya existe para el log con ID: {log_id}, ignorando.")
        return

    print(f"Creando archivo para log con ID: {log_id}")

    try:
        with open(filename, "w") as file:
            file.write(
                f"ID: {log_id}\nTipo: {tipo}\nMensaje: {mensaje}\nTimestamp: {time.ctime(timestamp)}\n"
            )
        print(f"Archivo creado: {filename}")
    except Exception as e:
        print(f"Error al crear el archivo para el log: {e}")

if __name__ == '__main__':
    print("Elige una opción:")
    print("1. Mostrar todos los logs")
    print("2. Mostrar logs por tipo")

    opcion = input("Selecciona una opción: ")
    if opcion == "1":
        logs = logs_collection.find().sort("timestamp", -1)
        for log in logs:
            print(f"{time.ctime(log['timestamp'])} - {log['tipo']}: {log['mensaje']}")
            guardar_log_en_archivo(log)
    elif opcion == "2":
        tipo = input("Introduce el tipo de log (barcelona, gestos, etc.): ")
        logs = logs_collection.find({"tipo": tipo}).sort("timestamp", -1)
        for log in logs:
            print(f"{time.ctime(log['timestamp'])} - {log['tipo']}: {log['mensaje']}")
            guardar_log_en_archivo(log)
    else:
        print("Opción no válida")