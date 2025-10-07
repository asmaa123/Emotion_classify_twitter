import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
import nltk

# Download NLTK data if not already present
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')


# ===== Load Model, Tokenizer & LabelEncoder =====
model = load_model("lstm_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("labelencoder.pkl", "rb") as f:
    le = pickle.load(f)

MAX_LEN = 100

# ===== Page Config =====
st.set_page_config(page_title="Twitter Emotion Detector", page_icon="💬", layout="centered")

# ===== Custom CSS for Modern UI =====
st.markdown("""
    <style>
        body {
            background-color: #0f172a;
        }
        .stApp {
            background-color: #0f172a;
            color: #e2e8f0;
            font-family: 'Segoe UI', sans-serif;
        }

        /* ==== Header Bar ==== */
        .top-bar {
            background-color: #1e293b;
            padding: 15px 25px;
            border-radius: 12px;
            margin-bottom: 25px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .logo {
            font-size: 20px;
            font-weight: 600;
            color: #38bdf8;
        }
        .subtitle {
            font-size: 14px;
            color: #94a3b8;
        }

        h1 {
            color: #38bdf8 !important;
            text-align: center;
            font-size: 36px !important;
            margin-bottom: 10px !important;
        }
        .stTextArea textarea {
            background-color: #1e293b !important;
            color: #e2e8f0 !important;
            border: 1px solid #334155 !important;
            border-radius: 10px !important;
            font-size: 16px !important;
            padding: 12px !important;
        }
        .stButton>button {
            background-color: #38bdf8 !important;
            color: #0f172a !important;
            font-weight: bold;
            border-radius: 8px;
            font-size: 16px;
            padding: 8px 18px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #0ea5e9 !important;
            color: white !important;
            transition: 0.3s;
        }
        .result-box {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            font-size: 18px;
            text-align: center;
            color: #a5f3fc;
        }
    </style>
""", unsafe_allow_html=True)

# ===== Custom Header =====
st.markdown("""
    <div class="top-bar">
        <div class="logo">💬 Twitter Emotion AI</div>
        <div class="subtitle">Powered by LSTM + TF-IDF</div>
    </div>
""", unsafe_allow_html=True)

# ===== Streamlit UI =====
st.title("Twitter Emotion Detection")
st.markdown("<p style='text-align:center; font-size:18px; color:#94a3b8;'>Analyze the emotion behind any tweet instantly.</p>", unsafe_allow_html=True)

user_input = st.text_area("Enter your comment:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a comment!")
    else:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        stemmer = LancasterStemmer()
        stop_words = set(stopwords.words("english"))

        def clean_text(text):
            text = text.lower()
            text = re.sub(r'@\S+|http\S+|\.pic\S+', ' ', text)
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            words = nltk.word_tokenize(text)
            words = [stemmer.stem(w) for w in words if w not in stop_words and len(w) > 2]
            return ' '.join(words)

        cleaned_input = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned_input])
        padded = pad_sequences(seq, maxlen=MAX_LEN)
        pred = model.predict(padded)
        pred_class = pred.argmax(axis=1)[0]
        emotion = le.inverse_transform([pred_class])[0]

        st.markdown(f"<div class='result-box'>Predicted Emotion: <b>{emotion}</b></div>", unsafe_allow_html=True)

