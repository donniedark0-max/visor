import sys
from detection.gesture_detection import GestureDetectionApp

if __name__ == "__main__":
    # Ejecutar la aplicación de PyQt en el hilo principal
    app = GestureDetectionApp(sys.argv)
    sys.exit(app.exec())