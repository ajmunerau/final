import streamlit as st
import cv2
import numpy as np

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

uploaded_file = st.file_uploader("Carga una imagen", type=["jpg", "jpeg", "png"])
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
    if num_faces > 0:
        for (x, y, w, h) in faces:
            # Análisis de color de camiseta
            shirt_region = img[y+h:y+h+h//2, x:x+w]
            shirt_black = is_color_in_range(shirt_region, black_lower, black_upper)

            # Dibujar rectángulo alrededor del rostro
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Mostrar información si se detecta camiseta negra
            if shirt_black:
                cv2.putText(img, "Camiseta Negra Detectada", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if num_faces == 1:
            st.subheader("¡Puedes entrar a la casa!")
            st.write("Eres una persona")
            if shirt_black:
                st.write("Andrés Julián Múnera Uribe\n23 años\nCédula 1001011725\nEstudiante de Diseño Interactivo")
        else:
            st.subheader("¡Pueden entrar a la casa!")
            st.write(f"Se detectaron {num_faces} rostros")
    else:
        st.subheader("Acceso bloqueado")
        st.write("No se detectaron rostros")

    # Mostrar el resultado
    st.image(img, channels="BGR", use_column_width=True)
