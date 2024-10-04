import cv2
from ultralytics import YOLO

# Cargar los dos modelos guardados
model1_path = "/Users/dark0/Documents/Visor/assets/models/train3/weights/best.pt"  # Cambia esto a la ruta de tu primer modelo
model2_path = "/Users/dark0/Documents/Visor/assets/models/train13/weights/best.pt"  # Cambia esto a la ruta de tu segundo modelo

model1 = YOLO(model1_path)
model2 = YOLO(model2_path)

# Inicializar la cámara
cap = cv2.VideoCapture(1)  # Usar la cámara por defecto

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

# Loop para procesar el video en tiempo real
while True:
    # Leer un frame de la cámara
    ret, frame = cap.read()

    if not ret:
        print("No se pudo leer el frame de la cámara")
        break

    # Realizar la detección con el primer modelo
    results1 = model1.predict(frame, conf=0.25)
    # Realizar la detección con el segundo modelo
    results2 = model2.predict(frame, conf=0.25)

    # Obtener los frames anotados de ambos modelos
    annotated_frame1 = results1[0].plot()
    annotated_frame2 = results2[0].plot()

    # Mostrar los resultados en dos ventanas diferentes
    cv2.imshow('Detecciones Modelo 1', annotated_frame1)
    cv2.imshow('Detecciones Modelo 2', annotated_frame2)

    # Presionar 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()