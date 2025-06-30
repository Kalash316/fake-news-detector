import streamlit as st
import pickle
from PIL import Image
import requests
from readability import Document
from bs4 import BeautifulSoup
import pandas as pd

# Load model & vectorizer
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# Page setup
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
logo = Image.open("assets/logo.png")
st.image(logo, width=80)

st.title("📰 Fake News Detector")
st.write("Check if a news is Fake or Real using Machine Learning!")

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Source credibility dictionary
credibility = {
    "bbc": 0.9,
    "ndtv": 0.85,
    "cnn": 0.8,
    "whatsapp": 0.3,
    "random": 0.2
}

# Choose Input Type
st.markdown("### 🧾 Choose Input Method")
input_mode = st.radio("Select input type:", ["Paste News Text", "Paste URL"])

news = ""
source = ""

# Text input
if input_mode == "Paste News Text":
    col1, col2 = st.columns([3, 1])
    with col1:
        news = st.text_area("📝 Enter the News Text", height=250)
    with col2:
        source = st.text_input("🌐 Source (optional)", placeholder="e.g., BBC, WhatsApp")

# URL input
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

# Prediction
if st.button("Check"):
    with st.spinner("Analyzing the news..."):
        if news.strip() == "":
            st.warning("Please enter some text or paste a URL.")
        else:
            transformed = vectorizer.transform([news])
            prediction = model.predict(transformed)[0]

            result = "REAL" if prediction == 1 else "FAKE"
            credibility_score = credibility.get(source.lower(), 0.5) if source else "N/A"

            if prediction == 0:
                st.error("❌ This news is likely FAKE.")
                fake_img = Image.open("assets/fake.png")
                st.image(fake_img, width=300)
            else:
                st.success("✅ This news is likely REAL.")
                real_img = Image.open("assets/real.png")
                st.image(real_img, width=300)

            # Save result to history
            st.session_state.history.append({
                "News": news[:100] + "..." if len(news) > 100 else news,
                "Prediction": result,
                "Source": source if source else "N/A",
                "Credibility": f"{credibility_score * 100:.0f}%" if source else "N/A"
            })

            # Show source credibility
            if source:
                st.info(f"🔍 Source Credibility Score: {credibility_score * 100:.0f}%")

# Show session history
if st.session_state.history:
    st.markdown("---")
    st.subheader("🧾 Session History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    # CSV Download Button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Session History as CSV",
        data=csv,
        file_name="news_history.csv",
        mime="text/csv"
    )
