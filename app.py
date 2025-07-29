import streamlit as st
import cv2
import numpy as np
from fer import FER
from PIL import Image
import random
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="Emotion Analyzer", layout="centered")
st.markdown("SnapEM")

# Initialize detector
detector = FER(mtcnn=True)

# Emotion-based quotes
emotion_quotes = {
    "angry": ["Keep calm, deep breaths help too 🌬️"],
    "disgust": ["Every feeling is valid — breathe out the stress 🍃"],
    "fear": ["Fear is temporary, courage is forever 💪"],
    "happy": ["Keep smiling, it looks good on you! 😊"],
    "sad": ["This too shall pass. You are not alone 💙"],
    "surprise": ["Life is full of little surprises 🎁"],
    "neutral": ["Balance is beautiful ☯️"]
}

# Session state
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

FRAME_WINDOW = st.image([])
capture_button = st.button("📸 Capture & Analyze")
history_expander = st.expander("📖 View Emotion History")

camera = cv2.VideoCapture(0)
captured_image = None

# --- Run camera loop ---
while True:
    ret, frame = camera.read()
    if not ret:
        st.error("Unable to access webcam.")
        break

    # Flip and show frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(rgb_frame, channels="RGB")

    # When capture button pressed
    if capture_button:
        captured_image = rgb_frame.copy()

        # Detect emotion
        results = detector.detect_emotions(captured_image)
        if results:
            box = results[0]["box"]
            emotion, score = max(results[0]["emotions"].items(), key=lambda x: x[1])
            st.success(f"**Detected Emotion:** `{emotion.capitalize()}` ({int(score * 100)}%)")
            quote = random.choice(emotion_quotes.get(emotion, ["Stay positive and strong! 💖"]))
            st.info(f"💬 _{quote}_")

            # Save in history
            st.session_state.emotion_history.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Emotion": emotion.capitalize(),
                "Confidence (%)": int(score * 100)
            })
        else:
            st.warning("No face detected in the captured image.")

        break  # Stop loop only after capture

camera.release()

# Show history
if st.session_state.emotion_history:
    df = pd.DataFrame(st.session_state.emotion_history)
    with history_expander:
        st.dataframe(df, use_container_width=True)
        st.download_button("⬇️ Download History", df.to_csv(index=False).encode(), "emotion_history.csv", "text/csv")
