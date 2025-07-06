# 📰 Fake News Detection System using NLP

A machine learning web application that classifies news articles as **Real** or **Fake** using NLP techniques. Built with Scikit-learn and deployed using Streamlit for a clean and interactive user experience.

---

## 🚀 Live Demo

👉 [Click here to try the app](https://fake-news-detector-mlwnzersxdjxanr2r5bucp.streamlit.app/)

---

## 🔧 Tech Stack

- **Languages**: Python  
- **Libraries**: Pandas, NumPy, Scikit-learn  
- **NLP Tools**: TF-IDF Vectorizer, Stopword Removal  
- **Models**: Logistic Regression, Naive Bayes  
- **Deployment**: Streamlit  
- **Others**: Jupyter Notebook, Git, Streamlit Cloud

---

## 📊 Features

- Detects whether a news article is real or fake using machine learning
- Preprocessing pipeline: lowercasing, punctuation & stopword removal
- Vectorization using TF-IDF
- 95%+ model accuracy on test data
- Real-time prediction interface via Streamlit deployment

---

## 📸 Screenshots *(Optional)*

> Upload images in a `screenshots/` folder in your repo and replace the links below:

![App Interface](screenshots/interface.png)  
![Prediction Result](screenshots/result.png)

---

## 💻 Run Locally

Follow these steps to run the project on your local machine:

1. **Clone the repository**
```bash
git clone https://github.com/kalash316/fake-news-detection.git
cd fake-news-detection
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the app

bash
Copy
Edit
streamlit run app.py
Make sure the model and vectorizer files (model.pkl, tfidf.pkl) are in the correct directory.

📂 Folder Structure
kotlin
Copy
Edit
fake-news-detection/
├── app.py
├── model/
│   ├── model.pkl
│   └── tfidf.pkl
├── data/
│   └── news.csv
├── requirements.txt
├── README.md
└── screenshots/
📚 Dataset
Source: Fake and Real News Dataset – Kaggle

Description: ~44,000 labeled news articles

Classes: FAKE and REAL

✅ Future Improvements
Add deep learning models like LSTM, BERT

Deploy with a Flask or React-based frontend

Enable multilingual fake news detection

Add feedback loop to improve model over time

👤 Author
Kalash Dwivedi
📧 kalashdwivedi668@gmail.com
🔗 LinkedIn
💻 GitHub

🌟 Give a Star
If you found this project useful, consider giving it a ⭐ on GitHub — it motivates me to improve and share more projects!

yaml
Copy
Edit
