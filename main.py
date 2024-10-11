import threading
from detection.gesture_detection import detect_gestures_and_objects, FloatingMenu
from PyQt6.QtWidgets import QApplication
import sys

def run_gesture_detection():
    detect_gestures_and_objects()

# Función para ejecutar la interfaz gráfica
def run_gui():
    app = QApplication(sys.argv)
    window = FloatingMenu()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
     # Crear e iniciar un hilo para la interfaz gráfica
    gui_thread = threading.Thread(target=run_gui)
    gui_thread.start()

    # Crear e iniciar un hilo para la detección de gestos y objetos
    gesture_thread = threading.Thread(target=run_gesture_detection)
    gesture_thread.start()

    # Esperar a que el hilo de gestos termine
    gui_thread.join()    
    gesture_thread.join()