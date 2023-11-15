import streamlit as st
import cv2
import numpy as np
from datetime import datetime

def display_clock():
    # Mostrar la hora actual
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    st.sidebar.markdown(f'#### ⏰ Hora Actual: {current_time}')

def is_color_in_range(image, lower_color, upper_color):
    # Convertir la imagen a espacio de color HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Crear una máscara que identifique los píxeles en el rango de color
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # Calcular el porcentaje de píxeles en el rango
    coverage = np.sum(mask > 0) / mask.size
    return coverage > 0.2  # Ajustar este umbral según sea necesario

# Cargar el clasificador preentrenado de rostros de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("Sistema de Control de Acceso")
display_clock()  # Mostrar el reloj

uploaded_file = st.sidebar.file_uploader("Carga una imagen", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Detectar rostros
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Definir rangos de color para camiseta negra en HSV
    black_lower = np.array([0, 0, 0], np.uint8)
    black_upper = np.array([180, 255, 30], np.uint8)

    num_faces = len(faces)
    known_person_detected = False

    if num_faces > 0:
        for (x, y, w, h) in faces:
            # Análisis de color de camiseta
            shirt_region = img[y+h:y+h+h//2, x:x+w]
            shirt_black = is_color_in_range(shirt_region, black_lower, black_upper)

            # Dibujar rectángulo alrededor del rostro
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Comprobar si la persona con camiseta negra es conocida
            if shirt_black:
                known_person_detected = True
                st.subheader("¡Puedes entrar a la casa!")
                st.markdown("""
                    **Nombre:** Andrés Julián Múnera Uribe  
                    **Edad:** 23 años  
                    **Cédula:** 1001011725  
                    **Profesión:** Estudiante de Diseño Interactivo
                    """, unsafe_allow_html=True)
                break  # Salir del bucle ya que se encontró a la persona conocida

        if not known_person_detected:
            if num_faces == 1:
                st.subheader("¡Puedes entrar a la casa!")
                st.write("Eres una persona")
            else:
                st.subheader("¡Pueden entrar a la casa!")
                st.write(f"Se detectaron {num_faces} rostros")

            st.image(img, channels="BGR", use_column_width=True)

            # Crear formularios para cada rostro detectado
            for i in range(num_faces):
                with st.form(key=f'Form{i}'):
                    st.subheader(f'Información del Invitado {i+1}')
                    st.text_input("Nombre", value="", key=f'Nombre{i}')
                    st.text_input("Edad", value="", key=f'Edad{i}')
                    st.text_input("Cédula", value="", key=f'Cedula{i}')
                    st.text_input("Profesión", value="", key=f'Profesion{i}')
                    submitted = st.form_submit_button('Listo')
                    if submitted:
                        st.success('Invitados registrados')
        else:
            st.image(img, channels="BGR", use_column_width=True)

    else:
        st.subheader("Acceso bloqueado")


