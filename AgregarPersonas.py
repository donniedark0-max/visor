import cv2
import face_recognition
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from detection.mongodb import guardar_persona, guardar_log, personas_collection

# Almacenar datos faciales y nombres
known_face_encodings = []
known_face_names = []


# Coloca la función después de las importaciones y la declaración de las listas known_face_encodings y known_face_names
def cargar_datos_entrenados():
    """Carga los datos de la base de datos para el reconocimiento facial."""
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
                else:
                    print(f"No se pudo encontrar una cara en la imagen de {persona['nombre']}")
            except Exception as e:
                print(f"Error procesando la imagen de {persona['nombre']}: {e}")
    print("Datos entrenados cargados correctamente.")

# Después, asegúrate de que la función `cargar_datos_entrenados()` se llame en `main()` o antes de `reconocer_personas()`.

def capturar_imagen():
    """Abre la cámara y captura una imagen al presionar 'c'."""
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo acceder a la cámara.")
            break

        cv2.imshow("Captura de imagen - Presiona 'c' para capturar", frame)
        key = cv2.waitKey(1)
        if key == ord('c'):  # Captura con 'c'
            cap.release()
            cv2.destroyAllWindows()
            return frame
        elif key == ord('q'):  # Salir con 'q'
            break
    cap.release()
    cv2.destroyAllWindows()
    return None

def registrar_persona():
    """Registra una nueva persona en la base de datos."""
    frame = capturar_imagen()
    if frame is None:
        print("No se capturó ninguna imagen.")
        return

    nombre = input("Ingrese su nombre: ")
    equipo_favorito = input("Ingrese su equipo favorito: ")

    # Convertir la imagen a base64
    _, buffer = cv2.imencode('.jpg', frame)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    # Guardar en MongoDB
    guardar_persona(nombre, equipo_favorito, image_base64)
    guardar_log("registro", f"Nueva persona registrada: {nombre}")
    print("Persona registrada correctamente.")
    cargar_datos_entrenados()  # Actualizar el modelo con los nuevos datos

def reconocer_personas():
    """Reconoce personas en tiempo real usando la cámara."""
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo acceder a la cámara.")
            break

        # Procesar el marco para reconocimiento facial
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Desconocido"

            if True in matches:
                match_index = matches.index(True)
                name = known_face_names[match_index]
                persona = personas_collection.find_one({"nombre": name})
                if persona:
                    equipo = persona["equipo_favorito"]
                    datos = f"{name} - {equipo}"
                else:
                    datos = name
            else:
                datos = name

            # Dibujar un rectángulo alrededor del rostro y mostrar datos
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, datos, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Reconocimiento de Personas", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):  # Salir con 'q'
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    cargar_datos_entrenados()
    while True:
        print("\nOpciones:")
        print("1. Registrar nueva persona")
        print("2. Iniciar reconocimiento en tiempo real")
        print("3. Salir")
        opcion = input("Selecciona una opción: ")

        if opcion == "1":
            registrar_persona()
        elif opcion == "2":
            reconocer_personas()
        elif opcion == "3":
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Intente de nuevo.")

if __name__ == "__main__":
    main()
