import streamlit as st
import pandas as pd
import joblib
import nltk
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def download_nltk():
    
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)

download_nltk()


model = joblib.load("Sentiment_Model.pkl")
vectorizer = joblib.load("Vectorizer.pkl")

stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))

def preprocess(text):

    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = word_tokenize(text)
    
    preprocessed_word = [

        stemmer.stem(word)
        for word in words
        if word not in stop_words

    ]

    return " ".join(preprocessed_word)


st.title("🎬 IMDb Movie Review Sentiment Analysis")
st.write("Enter a movie review and predict whether it is **Positive** or **Negative**.")

user_input = st.text_area(
    "✍️ Enter Movie Review",
    placeholder="Type your message here...",
    height=120,
    label_visibility="collapsed"
)


if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        clean_text = preprocess(user_input)
        vector = vectorizer.transform([clean_text])
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector).max()

        if prediction == 1:
            st.success(f"😊 Positive Review\n\nConfidence : {probability*100:.2f}%")
        else:
            st.error(f"😞 Negative Review\n\nConfidence : {probability*100:.2f}%")

st.markdown("---")
st.caption("NLP | TF-IDF | Machine Learning | Streamlit")