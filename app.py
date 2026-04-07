import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from gtts import gTTS
import base64
import tempfile

# -------------------------------
# CONFIG
# -------------------------------
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("model.h5")

model = load_my_model()

# -------------------------------
# PREPROCESS
# -------------------------------
def preprocess_image(img):
    img = img.convert("RGB")
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------------
# AUTO PLAY AUDIO
# -------------------------------
def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()

    audio_html = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# -------------------------------
# MULTI-LANGUAGE MESSAGES
# -------------------------------
def get_message(predicted_class, lang):

    messages = {
        "English": {
            "Healthy": "The plant is healthy",
            "Early Blight": "The plant has early blight disease",
            "Late Blight": "The plant has late blight disease"
        },
        "Hindi": {
            "Healthy": "पौधा स्वस्थ है",
            "Early Blight": "यह अर्ली ब्लाइट रोग है",
            "Late Blight": "यह लेट ब्लाइट रोग है"
        },
        "Kannada": {
            "Healthy": "ಸಸ್ಯವು ಆರೋಗ್ಯಕರವಾಗಿದೆ",
            "Early Blight": "ಇದು ಎರ್ಳಿ ಬ್ಲೈಟ್ ರೋಗವಾಗಿದೆ",
            "Late Blight": "ಇದು ಲೇಟ್ ಬ್ಲೈಟ್ ರೋಗವಾಗಿದೆ"
        },
        "Tamil": {
            "Healthy": "தாவரம் ஆரோக்கியமாக உள்ளது",
            "Early Blight": "இது ஆரம்ப ப்ளைட் நோய்",
            "Late Blight": "இது தாமத ப்ளைட் நோய்"
        },
        "Telugu": {
            "Healthy": "మొక్క ఆరోగ్యంగా ఉంది",
            "Early Blight": "ఇది ఎర్లీ బ్లైట్ వ్యాధి",
            "Late Blight": "ఇది లేట్ బ్లైట్ వ్యాధి"
        }
    }

    return messages[lang][predicted_class]

# -------------------------------
# LANGUAGE CODE MAP
# -------------------------------
lang_code_map = {
    "English": "en",
    "Hindi": "hi",
    "Kannada": "kn",
    "Tamil": "ta",
    "Telugu": "te"
}

# -------------------------------
# SPEAK FUNCTION
# -------------------------------
def speak(text, lang_code):
    tts = gTTS(text=text, lang=lang_code)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

# -------------------------------
# UI
# -------------------------------
st.title("🌿 Potato Disease Detection with AI")

language = st.selectbox(
    "🌍 Select Language",
    ["English", "Hindi", "Kannada", "Tamil", "Telugu"]
)

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

# -------------------------------
# PREDICTION
# -------------------------------
if uploaded_file:
    img_pil = Image.open(uploaded_file)
    st.image(img_pil, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(img_pil)

    prediction = model.predict(img_array)[0]
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    predicted_class = CLASS_NAMES[class_index]

    # -------------------------------
    # RESULT
    # -------------------------------
    st.subheader("🔍 Prediction Result")

    if predicted_class == "Healthy":
        st.success(f"✅ {predicted_class}")
    else:
        st.error(f"🦠 {predicted_class}")

    st.info(f"Confidence: {confidence*100:.2f}%")

    # -------------------------------
    # GRAPH
    # -------------------------------
    st.subheader("📊 Prediction Graph")

    fig, ax = plt.subplots()
    ax.bar(CLASS_NAMES, prediction * 100)
    ax.set_ylabel("Confidence (%)")
    ax.set_title("Prediction Confidence")

    st.pyplot(fig)

    # -------------------------------
    # 🔊 AUTO VOICE OUTPUT
    # -------------------------------
    st.subheader("🔊 Voice Output")

    message = get_message(predicted_class, language)
    audio_file = speak(message, lang_code_map[language])
    autoplay_audio(audio_file)

    # -------------------------------
    # DETAILS
    # -------------------------------
    st.subheader("📋 Detailed Probabilities")

    for i, name in enumerate(CLASS_NAMES):
        st.write(f"{name}: {prediction[i]*100:.2f}%")