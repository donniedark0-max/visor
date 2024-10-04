from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import cv2
#from detection.gesture_detection import frame_queue, hand_position, finger_distance
# import math
# Variables globales
camera_texture_id = None
cube_scale = 1.0
print(f"Distancia entre dedos: {finger_distance}")

def init_camera_texture():
    global camera_texture_id
    camera_texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, camera_texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

def update_camera_texture(frame):
    global camera_texture_id
    frame = cv2.flip(frame, 0)  # Invertir la imagen para OpenGL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = frame_rgb.shape

    glBindTexture(GL_TEXTURE_2D, camera_texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)

def draw_camera_background():
    global camera_texture_id
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, camera_texture_id)
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(-10.0, -7.5, -20.0)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(10.0, -7.5, -20.0)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(10.0, 7.5, -20.0)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(-10.0, 7.5, -20.0)
    glEnd()

def draw_cube():
    vertices = [
    (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5),
    (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, -0.5, -0.5), (-0.5, -0.5, -0.5),
    (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
    (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5), (-0.5, -0.5, -0.5),
    (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, -0.5),
    (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5)
    ]

# Triángulos para conectar los vértices del cubo
    triangles = [
        0, 2, 3, 0, 3, 1,   # Frente
        8, 4, 5, 8, 5, 9,   # Arriba
        10, 6, 7, 10, 7, 11,  # Atrás
        12, 13, 14, 12, 14, 15,  # Abajo
        16, 17, 18, 16, 18, 19,  # Lado izquierdo
        20, 21, 22, 20, 22, 23   # Lado derecho
    ]

    glColor3f(1.0, 1.0, 1.0)
    glLineWidth(2.0)

# Función para dibujar el cubo con estructura wireframe usando GL_LINES
   
    glBegin(GL_LINES)
    for t in range(0, len(triangles), 3):
            glVertex3fv(vertices[triangles[t]])
            glVertex3fv(vertices[triangles[t + 1]])
            glVertex3fv(vertices[triangles[t + 2]])
    glEnd()

def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)  # Fondo negro
    glEnable(GL_DEPTH_TEST)

    # Inicializa la textura de la cámara
    init_camera_texture()

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1.0, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)

def draw_text(x, y, text, color=(1.0, 1.0, 1.0)):
    """Función para dibujar texto en la pantalla."""
    glColor3f(color[0], color[1], color[2])  # Definir color del texto
    glWindowPos2f(x, y)  # Posicionar el texto
    for ch in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ctypes.c_int(ord(ch)))  # Dibujar cada carácter

#def draw_circle(x, y, radius):
   # glBegin(GL_LINE_LOOP)
   # for i in range(100):
    #    angle = 2 * math.pi * i / 100
   #     glVertex2f(x + math.cos(angle) * radius, y + math.sin(angle) * radius)
   # glEnd()

def draw_scene():
    global cube_scale
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()


    # Tomar el frame de la cámara desde la cola
    if not frame_queue.empty():
        frame = frame_queue.get()  # Obtener el último frame disponible
        update_camera_texture(frame)
    draw_camera_background()  # Dibuja el último frame aunque no sea nuevo

   # Mover el cubo basado en la posición de la mano
    glTranslatef(hand_position[0] * 5, hand_position[1] * 5, -10.0)
    glRotatef(hand_position[0] * 180, 0.0, 1.0, 0.0)
    

    # Escalar el cubo basado en la distancia entre los dedos
    cube_scale = max(0.9, finger_distance * 20)  # Escalar con un límite mínimo
    glScalef(cube_scale, cube_scale, cube_scale)
    print(f"Escalando cubo: {cube_scale}")

    # Dibujar el cubo
    draw_cube()
    draw_text(60, 60, "Control basado en gestos", color=(1.0, 1.0, 1.0))

    glutSwapBuffers()

def update_scene():
    glutPostRedisplay()

def keyboard_callback(key, x, y):
    if key == b'\x1b':  # Esc key in ASCII
        print("Saliendo del programa...")
        glutLeaveMainLoop()  # Termina el loop de OpenGL y cierra la ventana

def start_visualization():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(800, 600)  # Cambia el tamaño inicial de la ventana
    glutCreateWindow(b"3D Cube with Gesture Control")
    glutDisplayFunc(draw_scene)  # Dibujar la escena
    glutIdleFunc(update_scene)
    
    # Poner en pantalla completa
    glutFullScreen()
    glutKeyboardFunc(keyboard_callback)

    init()
    glutMainLoop()