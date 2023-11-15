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

    # Resto del código de detección y procesamiento...

# Continúa con el resto del código que ya tienes

