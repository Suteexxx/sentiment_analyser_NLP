import streamlit as st
import pickle
import numpy as np

# ---------- Page Config ----------
st.set_page_config(
    page_title="Emotion Analyzer",
    page_icon="🧠",
    layout="wide"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#0f172a,#1e293b);
color:white;
}

.title{
text-align:center;
font-size:55px;
font-weight:800;
color:#38bdf8;
}

.subtitle{
text-align:center;
font-size:20px;
color:#cbd5e1;
margin-bottom:40px;
}

.result{
background-color:#1e293b;
padding:30px;
border-radius:20px;
box-shadow:0px 0px 20px rgba(56,189,248,0.4);
text-align:center;
font-size:30px;
font-weight:bold;
}

textarea{
font-size:18px !important;
}

</style>
""", unsafe_allow_html=True)

# ---------- Load Model ----------
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ---------- Header ----------
st.markdown('<div class="title">🧠 AI Emotion Analyzer</div>', unsafe_allow_html=True)

st.markdown(
'<div class="subtitle">Understand emotions hidden inside text with Machine Learning</div>',
unsafe_allow_html=True)

# ---------- Input ----------
text = st.text_area(
    "✍️ Enter your sentence",
    height=180,
    placeholder="Example: I am extremely happy after getting my internship..."
)

# Emojis
emoji_map = {
    "joy":"😄",
    "sadness":"😢",
    "anger":"😡",
    "fear":"😨",
    "love":"❤️",
    "surprise":"😲"
}

color_map = {
    "joy":"#22c55e",
    "sadness":"#3b82f6",
    "anger":"#ef4444",
    "fear":"#a855f7",
    "love":"#ec4899",
    "surprise":"#f59e0b"
}

# ---------- Prediction ----------
if st.button("🚀 Analyze Emotion", use_container_width=True):

    if text.strip() != "":

        vector = vectorizer.transform([text])

        prediction = model.predict(vector)[0]

        emoji = emoji_map.get(prediction, "🙂")
        color = color_map.get(prediction, "#38bdf8")

        st.markdown(
            f"""
            <div class="result" style="border:2px solid {color}">
            <h1>{emoji}</h1>
            <h2>Emotion Detected</h2>
            <h1 style="color:{color};">{prediction.upper()}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Confidence if available
        if hasattr(model, "predict_proba"):

            probs = model.predict_proba(vector)[0]

            st.subheader("📊 Confidence")

            classes = model.classes_

            for cls, p in zip(classes, probs):
                st.progress(float(p))
                st.write(f"**{cls}** : {p*100:.2f}%")

    else:
        st.warning("Please enter some text.")

# Footer
st.markdown("---")
st.markdown(
"<center>Built with ❤️ using Streamlit + Scikit-Learn</center>",
unsafe_allow_html=True
)