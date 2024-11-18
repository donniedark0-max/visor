import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame. Verifica la cámara.")
            break
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
