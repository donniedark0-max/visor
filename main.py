import threading
from detection.gesture_detection import detect_gestures_and_objects

def run_gesture_detection():
    detect_gestures_and_objects()

if __name__ == "__main__":
    # Crear e iniciar un hilo para la detección de gestos y objetos
    gesture_thread = threading.Thread(target=run_gesture_detection)
    gesture_thread.start()

    # Esperar a que el hilo de gestos termine
    gesture_thread.join()