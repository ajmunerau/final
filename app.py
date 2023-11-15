import streamlit as st
import cv2
import numpy as np
from datetime import datetime

def display_clock():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    st.sidebar.markdown(f'#### ⏰ Hora Actual: {current_time}')

def is_color_in_range(image, lower_color, upper_color):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    coverage = np.sum(mask > 0) / mask.size
    return coverage > 0.2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("Sistema de Control de Acceso")
display_clock()

uploaded_file = st.sidebar.file_uploader("Carga una imagen", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    black_lower = np.array([0, 0, 0], np.uint8)
    black_upper = np.array([180, 255, 30], np.uint8)

    num_faces = len(faces)
    if num_faces > 0:
        shirt_black_detected = False
        for (x, y, w, h) in faces:
            shirt_region = img[y+h:y+h+h//2, x:x+w]
            shirt_black = is_color_in_range(shirt_region, black_lower, black_upper)
            if shirt_black:
                shirt_black_detected = True
                st.subheader("¡Puedes entrar a la casa!")
                st.markdown("""
                    **Nombre:** Andrés Julián Múnera Uribe  
                    **Edad:** 23 años  
                    **Cédula:** 1001011725  
                    **Profesión:** Estudiante de Diseño Interactivo
                    """, unsafe_allow_html=True)
                break  # Detener el bucle ya que hemos detectado a la persona con camiseta negra

        if not shirt_black_detected:
            st.image(img, channels="BGR", use_column_width=True)
            for i in range(num_faces):
                with st.form(key=f'Form{i}'):
                    st.subheader(f'Información del Invitado {i+1}')
                    st.text_input("Nombre", key=f'Nombre{i}')
                    st.text_input("Edad", key=f'Edad{i}')
                    st.text_input("Cédula", key=f'Cedula{i}')
                    st.text_input("Profesión", key=f'Profesion{i}')
                    submitted = st.form_submit_button('Listo')
                    if submitted:
                        st.success('Invitados registrados')
    else:
        # Si no se detectan rostros, se muestra la imagen de acceso denegado
        denied_image_path = "LB.jpeg"  # Asegúrate de poner la ruta correcta aquí
        st.subheader("Acceso bloqueado")
        st.write("No se detectaron rostros")
        st.image(denied_image_path, use_column_width=True)
