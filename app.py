import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import threading

st.set_page_config(
    page_title="Reconocimiento LSE en Tiempo Real",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500&display=swap');
    :root { --primary: #667eea; --secondary: #764ba2; --light: #f8f9fa; --dark: #2c3e50; }
    h1,h2,h3 { font-family: 'Poppins', sans-serif; color: #2c3e50; }
    .stButton>button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px; padding: 10px 20px; font-weight:600; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 15px rgba(102,126,234,0.3); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; padding: 30px 0;'>
    <h1 style='font-size: 3em; color: #667eea; margin: 0;'>🤟 Reconocimiento LSE</h1>
    <p style='font-size: 1.1em; color: #555; margin: 8px 0 0 0;'>Lenguaje de Señas Español en Tiempo Real</p>
</div>
""", unsafe_allow_html=True)

# --- Constantes y modelo ---
MODEL_PATH = "sign_language_mlp_model.h5"
CLASSES = np.array(['A','B','C','D','E','F','G','I','K','L','M','N','O','P','Q','R','S','T','U'])

@st.cache_resource
def load_model(path=MODEL_PATH):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo. Verifique la ruta: {path}\n{e}")
        st.stop()

model = load_model()

with st.sidebar:
    st.success("✅ Modelo cargado correctamente.")

# Video 
class HandSignProcessor(VideoProcessorBase):
    def __init__(self):
        # MediaPipe: limitar a 1 mano y reducir complejidad para velocidad
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,  # más rápido, suficiente para muchas apps en tiempo real
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.landmark_spec = self.mp_drawing.DrawingSpec(circle_radius=2, thickness=2)
        self.connection_spec = self.mp_drawing.DrawingSpec(thickness=2)
        self.lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        h, w, _ = image.shape

        # convertir a RGB para MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True

        predicted_sign = "Esperando..."
        confidence = 0.0

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            lm = np.array([[l.x, l.y, l.z] for l in hand_landmarks.landmark], dtype=np.float32)
            input_data = lm.flatten()[None, :]  

            try:
                preds = model.predict(input_data, verbose=0)
                idx = int(np.argmax(preds[0]))
                confidence = float(preds[0][idx])
                predicted_sign = CLASSES[idx]
            except Exception:
                predicted_sign = "Error modelo"
                confidence = 0.0

            # Dibujar landmarks y conexiones
            self.mp_drawing.draw_landmarks(
                image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.landmark_spec, self.connection_spec
            )

            # Calcular bounding box de la mano (en coordenadas de imagen)
            xs = (lm[:, 0] * w).astype(int)
            ys = (lm[:, 1] * h).astype(int)
            x_min, x_max = np.clip([xs.min() - 10, xs.max() + 10], 0, w)
            y_min, y_max = np.clip([ys.min() - 10, ys.max() + 10], 0, h)

            # Fondo para texto
            cv2.rectangle(image, (x_min, y_min - 80), (x_max, y_min + 10), (0, 0, 0), -1)
            cv2.rectangle(image, (x_min, y_min - 80), (x_max, y_min + 10), (0, 255, 0), 2)

            # Texto con resultado y confianza
            cv2.putText(image, f"{predicted_sign}", (x_min + 8, y_min - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(image, f"{confidence*100:.1f}%", (x_min + 8, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)

        else:
            # sin mano detectada: mensaje discreto
            cv2.rectangle(image, (20, 15), (420, 80), (60, 60, 60), -1)
            cv2.putText(image, "Buscando mano...", (35, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200,200,200), 2, cv2.LINE_AA)

        try:
            st.session_state.latest_prediction = (predicted_sign, confidence)
        except Exception:
            pass

        return av.VideoFrame.from_ndarray(image, format="bgr24")


# Interfaz 
st.markdown("---")

left_col, center_col, right_col = st.columns([1, 2, 1])

with center_col:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 12px; border-radius: 10px; margin-bottom: 10px; text-align:center;'>
        <h3 style='color: white; margin: 0;'>📹 Cámara</h3>
    </div>
    """, unsafe_allow_html=True)

    media_constraints = {
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "facingMode": "user"
        },
        "audio": False
    }

    webrtc_ctx = webrtc_streamer(
        key="lse_detector_center",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=HandSignProcessor,
        media_stream_constraints=media_constraints,
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

st.markdown("---")

st.sidebar.markdown("---")
with st.sidebar:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 12px; border-radius: 10px; color: white; margin-bottom: 16px;'>
        <h3 style='margin:0'>⚙️ Configuración</h3>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🎯 Detalles del Modelo"):
        st.write(f"**Clases:** {len(CLASSES)} letras")
        st.write(f"**Letras:** {', '.join(CLASSES)}")
        st.write(f"**Framework:** TensorFlow {tf.__version__}")

    with st.expander("💡 Consejos de Uso"):
        st.markdown("""
        - 💡 Mantén buena iluminación.
        - 📹 Posiciónate frente a la cámara.
        - 🎯 Coloca la mano en el centro del video.
        - ⏱️ Espera 1-2 segundos por cada seña.
        """)

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 10px; border-radius: 8px; background: #f0f2f6;'>
        <p style='margin: 0; font-size: 0.85em; color: #666;'>
            🤟 <strong>LSE Recognition</strong><br>Streamlit + MediaPipe + TensorFlow
        </p>
    </div>
    """, unsafe_allow_html=True)


# --- 5. SIDEBAR FINAL ---
st.sidebar.markdown("---")
st.sidebar.caption("Proyecto de Reconocimiento LSE | Desarrollado con Streamlit y MediaPipe.")
