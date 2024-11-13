import cv2
import mediapipe as mp
from collections import defaultdict

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Diccionario de contexto para almacenar las poses detectadas
contexto_poses = {
    "poses_detectadas": set(),
}

def detectar_manos_levantadas(pose_landmarks):
    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    if (left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y):
        return "Manos levantadas"
    return None

def detectar_manos_cruzadas(pose_landmarks):
    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]

    if (abs(left_wrist.x - right_wrist.x) < 0.05 and
        left_wrist.y < left_elbow.y and right_wrist.y < right_elbow.y):
        return "Manos cruzadas"
    return None

def detectar_x_brazos(pose_landmarks):
    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    if (left_wrist.x < right_shoulder.x and right_wrist.x > left_shoulder.x and
        left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y):
        return "X con los brazos"
    return None

def detectar_parado(pose_landmarks):
    hip_left = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    hip_right = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    knee_left = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    knee_right = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

    if (knee_left.visibility > 0.5 and knee_right.visibility > 0.5 and
        knee_left.y > hip_left.y and knee_right.y > hip_right.y):
        return "De pie"
    return "Sentado"

def detectar_poses_y_almacenar():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo capturar el frame.")
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            contexto_poses["poses_detectadas"].clear()

            if results.pose_landmarks:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                pose_manos_levantadas = detectar_manos_levantadas(results.pose_landmarks)
                if pose_manos_levantadas:
                    contexto_poses["poses_detectadas"].add(pose_manos_levantadas)

                pose_manos_cruzadas = detectar_manos_cruzadas(results.pose_landmarks)
                if pose_manos_cruzadas:
                    contexto_poses["poses_detectadas"].add(pose_manos_cruzadas)

                pose_x_brazos = detectar_x_brazos(results.pose_landmarks)
                if pose_x_brazos:
                    contexto_poses["poses_detectadas"].add(pose_x_brazos)

                pose_parado_sentado = detectar_parado(results.pose_landmarks)
                contexto_poses["poses_detectadas"].add(pose_parado_sentado)

            # Mostrar en pantalla las poses detectadas en el frame
            for idx, pose_detectada in enumerate(contexto_poses["poses_detectadas"]):
                cv2.putText(frame, pose_detectada, (10, 30 + idx * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Detección de Poses", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error durante la detección de poses: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
