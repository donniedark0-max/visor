import cv2
import pytesseract

# Cargar imagen
img = cv2.imread('D:/Roboticoss/Proyectos/Vision por computadora/Reconocimiento de texto mediante OpenCV y Tesseract (OCR)/Texto1.png')

# Convertir imagen a escalas de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar umbral para convertir a imagen binaria
threshold_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#Cambiar directorio donde esta instalado tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Pasar la imagen a través de pytesseract
text = pytesseract.image_to_string(threshold_img)

cv2.imshow("gray",gray)
cv2.imshow("threshold", threshold_img)
# Print the extracted text
print(text)
