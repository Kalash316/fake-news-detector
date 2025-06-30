import streamlit as st
import pickle
from PIL import Image
import requests
from readability import Document
from bs4 import BeautifulSoup
import pandas as pd
import os

import gdown

def download_file_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

# Use file IDs only
download_file_from_drive("1DXoHmMfZb5DgJ1p8BRlCtCf0oLoSPDK5", "model/model.pkl")
download_file_from_drive("123MrAKCUZXI5KmncT6MdJCMsoFl9JMzP", "model/vectorizer.pkl")

# ----------------------------- #
# 🔧 Load model & vectorizer
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# 🖼 UI Setup
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
logo = Image.open("assets/logo.png")
st.image(logo, width=80)

st.title("📰 Fake News Detector")
st.write("Check if a news is Fake or Real using Machine Learning!")

# 🧠 Store prediction history
if "history" not in st.session_state:
    st.session_state.history = []

# 🌐 Source credibility scores
credibility = {
    "bbc": 0.9,
    "ndtv": 0.85,
    "cnn": 0.8,
    "whatsapp": 0.3,
    "random": 0.2
}

# 🔘 Input method
st.markdown("### 🧾 Choose Input Method")
input_mode = st.radio("Select input type:", ["Paste News Text", "Paste URL"])

news = ""
source = ""

# ✍️ News Text Input
if input_mode == "Paste News Text":
    col1, col2 = st.columns([3, 1])
    with col1:
        news = st.text_area("📝 Enter the News Text", height=250)
    with col2:
        source = st.text_input("🌐 Source (optional)", placeholder="e.g., BBC, WhatsApp")

# 🔗 News URL Input
elif input_mode == "Paste URL":
    url = st.text_input("🔗 Paste News Article URL")
    if url:
        try:
            response = requests.get(url, timeout=10)
            doc = Document(response.text)
            html = doc.summary()
            soup = BeautifulSoup(html, "lxml")
            news = soup.get_text()

            st.success("✅ Article text extracted successfully!")
            st.text_area("📰 Extracted News Text", value=news, height=250)
        except Exception as e:
            st.error(f"⚠️ Failed to extract article: {str(e)}")

# 🚦 Check button & Prediction
if st.button("Check"):
    with st.spinner("Analyzing the news..."):
        if news.strip() == "":
            st.warning("Please enter some text or paste a URL.")
        else:
            transformed = vectorizer.transform([news])
            prediction = model.predict(transformed)[0]

            result = "REAL" if prediction == 1 else "FAKE"
            credibility_score = credibility.get(source.lower(), 0.5) if source else "N/A"

            # Output
            if prediction == 0:
                st.error("❌ This news is likely FAKE.")
                st.image("assets/fake.png", width=300)
            else:
                st.success("✅ This news is likely REAL.")
                st.image("assets/real.png", width=300)

            # Source credibility
            if source:
                st.info(f"🔍 Source Credibility Score: {credibility_score * 100:.0f}%")

            # 📝 Save session history
            st.session_state.history.append({
                "News": news[:100] + "..." if len(news) > 100 else news,
                "Prediction": result,
                "Source": source if source else "N/A",
                "Credibility": f"{credibility_score * 100:.0f}%" if source else "N/A"
            })

# 📑 Show Session History + CSV download
if st.session_state.history:
    st.markdown("---")
    st.subheader("🧾 Session History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Session History as CSV",
        data=csv,
        file_name="news_history.csv",
        mime="text/csv"
    )

