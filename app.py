import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# --- 1. CONFIGURACIÓN BÁSICA DE STREAMLIT ---
st.set_page_config(
    page_title="Reconocimiento LSE en Tiempo Real",
    page_icon="🤟",
    layout="wide"
)

# Título Principal
st.title("🤟 Reconocimiento de Lenguaje de Señas Español (LSE)")
st.markdown("### Detección y Clasificación en Tiempo Real")

# --- 2. CARGAR EL MODELO ENTRENADO ---
MODEL_PATH = 'sign_language_mlp_model.h5' 
try:
    # Carga del modelo (st.cache_resource asegura que solo se cargue una vez)
    # Movemos la confirmación a la barra lateral para no afectar el layout principal
    @st.cache_resource
    def load_model():
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    
    model = load_model()
    st.sidebar.success("✅ Modelo cargado correctamente.")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo. Verifique la ruta '{MODEL_PATH}'.")
    st.stop()

# Definir las clases (Incluyendo la 'Ñ')
# IMPORTANTE: Asegúrate de que el índice de 'Ñ' sea el mismo que usaste en tu entrenamiento.
# Asumiendo que 'Ñ' es la última clase:
CLASSES = np.array(['A','B','C','D','E','F','G','I','K','L','M','N','O','P','Q','R','S','T','U', 'Ñ']) 

# --- 3. INICIALIZAR MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- 4. CONFIGURACIÓN DE INTERFAZ (COLUMNAS Y PLACEHOLDERS) ---

# Columna 1 (Video y Seña en el frame): 70% del ancho
# Columna 2 (Resultado Estabilizado y Confianza): 30% del ancho
col1, col2 = st.columns([7, 3])

with col1:
    st.markdown("#### Cámara en Vivo (MediaPipe)")
    frame_placeholder = st.empty() # Placeholder para el video

with col2:
    st.markdown("#### Predicción Estable")
    # Usaremos un st.subheader para hacer la letra más grande y estable
    predicted_sign_display = st.subheader("Esperando...")
    st.markdown("---")
    st.markdown("##### Confianza de la Predicción")
    confidence_bar_placeholder = st.empty()


# Inicializar el estado de la aplicación
if 'running' not in st.session_state:
    st.session_state.running = False

# --- 5. FUNCIÓN PRINCIPAL DE PROCESAMIENTO ---
def process_video():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("No se pudo acceder a la cámara. Asegúrate de que no esté en uso.")
        st.session_state.running = False
        return

    with mp_hands.Hands(
        model_complexity=1, 
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.7) as hands:

        st.sidebar.info("Cámara activa. Procesando frames...")
        
        # Variables de Estabilización (Opcional, pero mejora la experiencia)
        # Aquí puedes implementar una lógica de historial si quieres evitar el parpadeo
        
        while st.session_state.running:
            success, image = cap.read()
            if not success: continue

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True
            
            # --- LÓGICA DE DETECCIÓN Y PREDICCIÓN ---
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 1. Extracción de Landmarks (63 features)
                    landmarks_list = []
                    for landmark in hand_landmarks.landmark:
                        landmarks_list.extend([landmark.x, landmark.y, landmark.z])
                    
                    # 2. Preprocesamiento
                    input_data = np.array(landmarks_list).flatten().astype('float32')
                    input_data = np.expand_dims(input_data, axis=0)

                    # 3. Predicción
                    prediction = model.predict(input_data, verbose=0)
                    predicted_class_index = np.argmax(prediction)
                    confidence = prediction[0][predicted_class_index]
                    predicted_sign = CLASSES[predicted_class_index]
                    
                    # 4. Dibujar los landmarks
                    mp_drawing.draw_landmarks(
                        image, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), 
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    # 5. Visualización en el Frame (OpenCV - Letra Grande y Verde)
                    cv2.putText(image, 
                                f"Resultado: {predicted_sign}", 
                                (50, 50), # Posición ajustada
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1.5, # Tamaño de fuente más grande
                                (0, 255, 0), # Color Verde
                                3, # Grosor de línea
                                cv2.LINE_AA)
                    
                    # 6. Actualizar la Interfaz de Resultados (Columna 2)
                    # Mostramos solo la letra predicha, el parpadeo ya no afecta tanto el texto
                    predicted_sign_display.subheader(f"## **{predicted_sign}**")
                    confidence_bar_placeholder.progress(float(confidence), text=f"Confianza: {confidence:.2f}")

            else:
                # Si no hay mano, limpiamos el resultado en la columna 2
                predicted_sign_display.subheader("Esperando...")
                confidence_bar_placeholder.empty()

            # Mostrar el frame procesado en Streamlit (Columna 1)
            # Usamos use_container_width=True para resolver la advertencia
            frame_placeholder.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        # Fuera del bucle while: liberar recursos
        cap.release()
        st.session_state.running = False
        st.sidebar.warning("Webcam Desactivada.")
        
# --- 6. INTERFAZ DE STREAMLIT (SIDEBAR) ---
st.sidebar.header("Control y Ajustes")

# Lógica del botón de encendido/apagado (Ahora más proporcional)
if st.session_state.running:
    # Si la app está corriendo, mostrar el botón de Detener
    if st.sidebar.button("⏹️ DETENER WEBCAM", use_container_width=True, type="primary"):
        st.session_state.running = False
        st.rerun() 
    process_video() 
else:
    # Si la app está detenida, mostrar el botón de Iniciar
    if st.sidebar.button("▶️ INICIAR WEBCAM", use_container_width=True, type="primary"):
        st.session_state.running = True
        st.rerun() 
    st.sidebar.info("Presiona 'INICIAR WEBCAM' para comenzar.")

st.sidebar.markdown("---")
st.sidebar.caption("Proyecto de Reconocimiento LSE | Desarrollado con Streamlit y MediaPipe.")