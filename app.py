import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import av # Necesario para manejar frames con streamlit-webrtc

# Librería para la cámara estable en entorno web
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode 


# --- 1. CONFIGURACIÓN BÁSICA DE STREAMLIT ---
st.set_page_config(
    page_title="Reconocimiento LSE en Tiempo Real",
    page_icon="🤟",
    layout="wide"
)

st.title("🤟 Reconocimiento de Lenguaje de Señas Español (LSE)")
st.markdown("### Detección y Clasificación en Tiempo Real con WebRTC")


# --- 2. CARGAR EL MODELO ENTRENADO ---
MODEL_PATH = 'sign_language_mlp_model.h5' 
@st.cache_resource
def load_model():
    """Carga el modelo una sola vez y lo almacena en caché."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.sidebar.success("✅ Modelo cargado correctamente.")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error al cargar el modelo. Verifique la ruta '{MODEL_PATH}'.")
        st.stop()

model = load_model()

# Definir las clases (Asegúrate que 'Ñ' esté en el índice correcto según tu entrenamiento)
CLASSES = np.array(['A','B','C','D','E','F','G','I','K','L','M','N','O','P','Q','R','S','T','U', 'Ñ']) 

# --- 3. CLASE PARA PROCESAR EL VIDEO EN TIEMPO REAL (WEBRTC) ---

class HandSignProcessor(VideoProcessorBase):
    """Procesa cada frame de video, detecta landmarks y predice la seña."""
    
    def __init__(self):
        # Inicializar MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Método llamado por streamlit-webrtc para cada frame."""
        
        # Convertir el frame de AV (WebRTC) a un array NumPy (BGR)
        image = frame.to_ndarray(format="bgr24")
        
        # 1. Preprocesamiento de la imagen
        image = cv2.flip(image, 1) # Efecto espejo
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # 2. Procesar con MediaPipe
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True
        
        predicted_sign = "Esperando..."
        confidence = 0.0
        
        # 3. Lógica de Detección y Predicción
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks_list = []
                for landmark in hand_landmarks.landmark:
                    # Extraer las 63 coordenadas (x, y, z)
                    landmarks_list.extend([landmark.x, landmark.y, landmark.z])
                
                # Preprocesamiento para el modelo (63 features)
                input_data = np.array(landmarks_list).flatten().astype('float32')
                input_data = np.expand_dims(input_data, axis=0)

                # Predicción (Acceso a la variable global 'model')
                prediction = model.predict(input_data, verbose=0) 
                predicted_class_index = np.argmax(prediction)
                confidence = prediction[0][predicted_class_index]
                predicted_sign = CLASSES[predicted_class_index]
                
                # Dibujar los landmarks en la imagen
                self.mp_drawing.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), 
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

                # Visualización en el Frame (Usamos 'Sena' para evitar el error de codificación)
                cv2.putText(image, 
                            f"Sena: {predicted_sign}", 
                            (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1.5, 
                            (0, 255, 0), 
                            3, 
                            cv2.LINE_AA)
        
        # Opcional: El parpadeo en la columna 2 es inevitable sin lógica de hilo, 
        # pero es útil ver el resultado fuera del frame.
        try:
            st.session_state.latest_prediction = (predicted_sign, confidence)
        except Exception:
             # Si no hay session_state (solo en el primer frame), lo ignoramos
             pass
             
        # Devuelve el frame modificado como objeto AV
        return av.VideoFrame.from_ndarray(image, format="bgr24")


# --- 4. INTERFAZ Y WEBRTC ---

col1, col2 = st.columns([7, 3])

# La cámara se inicia automáticamente cuando se carga la página con el componente
with col1:
    st.markdown("#### Webcam en Vivo (MediaPipe)")
    webrtc_ctx = webrtc_streamer(
        key="sign_language_detector",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=HandSignProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# Columna 2: Resultados
with col2:
    st.markdown("#### Predicción Estable")
    predicted_sign_display = st.subheader("Esperando...")
    st.markdown("---")
    st.markdown("##### Confianza de la Predicción")
    confidence_bar_placeholder = st.empty()
    
    # Mostrar resultados actualizados fuera del hilo del procesador (Más estable)
    if 'latest_prediction' not in st.session_state:
        st.session_state.latest_prediction = ("Esperando...", 0.0)

    # El botón "Iniciar" del webrtc_streamer ya funciona como control,
    # pero podemos mostrar el resultado de forma más estable aquí
    
    # Si el contexto de webrtc está activo (el usuario presionó Iniciar)
    if webrtc_ctx.state.playing and 'latest_prediction' in st.session_state:
        sign, conf = st.session_state.latest_prediction
        
        # Actualizamos la visualización en la columna 2
        predicted_sign_display.subheader(f"## **{sign}**")
        confidence_bar_placeholder.progress(float(conf), text=f"Confianza: {conf:.2f}")
    elif not webrtc_ctx.state.playing:
        predicted_sign_display.subheader("Detenida.")
        confidence_bar_placeholder.empty()

# --- 5. SIDEBAR FINAL ---
st.sidebar.header("Estado del Proyecto")
st.sidebar.markdown("---")
st.sidebar.info("El botón 'Iniciar' de la cámara aparecerá en la columna izquierda, manejado por WebRTC.")
st.sidebar.caption("Proyecto de Reconocimiento LSE | Desarrollado con Streamlit y MediaPipe.")
