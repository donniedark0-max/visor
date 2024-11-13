import threading
import sys
from detection.gesture_detection import detect_gestures_and_objects, FloatingMenu
from detection.pose_detection import detectar_poses_y_almacenar
from PyQt6.QtWidgets import QApplication
import signal

def run_gesture_detection(stop_event):
    try:
        detect_gestures_and_objects()
    except Exception as e:
        print(f"Error en el hilo de detección de gestos: {e}")
        stop_event.set()

def run_pose_detection(stop_event):
    try:
        detectar_poses_y_almacenar()
    except Exception as e:
        print(f"Error en el hilo de detección de poses: {e}")
        stop_event.set()

def run_gui(stop_event):
    app = QApplication(sys.argv)
    window = FloatingMenu()
    window.show()

    stop_event.set()
    app.exec()

def main():
    stop_event = threading.Event()

    gui_thread = threading.Thread(target=run_gui, args=(stop_event,))
    gui_thread.start()

    gesture_thread = threading.Thread(target=run_gesture_detection, args=(stop_event,))
    gesture_thread.start()

    pose_thread = threading.Thread(target=run_pose_detection, args=(stop_event,))
    pose_thread.start()

    def signal_handler(sig, frame):
        print("Terminando la aplicación...")
        stop_event.set()
        gui_thread.join()
        gesture_thread.join()
        pose_thread.join()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    gui_thread.join()
    gesture_thread.join()
    pose_thread.join()

if __name__ == "__main__":
    main()
