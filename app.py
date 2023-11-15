import streamlit as st
import cv2
import numpy as np

# Cargar el clasificador preentrenado de rostros de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("Sistema de Control de Acceso")

uploaded_file = st.file_uploader("Carga una imagen", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Detectar rostros
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dibujar rectángulos alrededor de los rostros y mostrar mensajes según la detección
    if len(faces) > 0:
        st.subheader("¡Puedes entrar a la casa!")
        st.write("Eres una persona")
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    else:
        st.subheader("Acceso bloqueado")
        st.write("No se detectaron rostros")

    # Mostrar el resultado
    st.image(img, channels="BGR", use_column_width=True)



