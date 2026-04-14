# =========================
# IMPORT LIBRARIES
# =========================
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# LOAD MODEL & TOKENIZER
# =========================
model = load_model("lstm_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ⚠️ SAME VALUE USED IN TRAINING
max_len = 50  

# =========================
# PREDICTION FUNCTION
# =========================
def predict_message(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)

    pred = model.predict(padded)[0][0]

    if pred > 0.5:
        return "🚨 Spam"
    else:
        return "✅ Ham"

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Spam Detection App", layout="centered")

st.title("📩 SMS Spam Detection using LSTM")
st.write("Enter a message to check whether it is Spam or Ham")

# Input box
user_input = st.text_area("Enter your message:")

# Predict button
if st.button("Check"):
    if user_input.strip() != "":
        result = predict_message(user_input)
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter a message")

# =========================
# SAMPLE TESTS
# =========================
st.subheader("Try these examples:")

st.write("👉 Congratulations! You have won a free iPhone")
st.write("👉 Win cash prize now!!! Limited offer")
st.write("👉 Hey, are we meeting today?")
st.write("👉 Call me when you are free")