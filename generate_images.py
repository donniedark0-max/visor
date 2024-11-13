import cv2
import os

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")

outputp_dir ='output_images'
os.makedirs(outputp_dir, exist_ok=True)

img_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el frame.")
        break

    # Convertir la imagen de BGR a RGB para mostrarla correctamente

    cv2.imshow('Captura de Imágenes', frame)

    k = cv2.waitKey(1)


    if k%256 == 27:
        print("saliendo..")
        break
    elif k%256 == ord('s'):
        # Guardar la imagen en la carpeta output_images
        img_name = os.path.join(outputp_dir, "opencv_frame_{}.png".format(img_counter))
        cv2.imwrite(img_name, frame)
        print("{} guardado!".format(img_name))
        img_counter += 1

cap.release()
cv2.destroyAllWindows()