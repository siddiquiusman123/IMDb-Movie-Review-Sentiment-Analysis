import streamlit as st
import joblib
import nltk
import re

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ----------------------------------
# NLTK DOWNLOAD (CACHED)
# ----------------------------------
@st.cache_resource
def download_nltk():
    nltk.download("punkt")
    nltk.download("stopwords")

download_nltk()

# ----------------------------------
# LOAD MODEL & VECTORIZER
# ----------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("Sentiment_Model.pkl")
    vectorizer = joblib.load("Vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ----------------------------------
# NLP TOOLS
# ----------------------------------
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))

# ----------------------------------
# TEXT PREPROCESSING FUNCTION
# ----------------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)          # remove HTML tags
    text = re.sub(r"[^a-z\s]", "", text)       # remove punctuation & numbers

    tokens = word_tokenize(text)

    processed_words = [
        stemmer.stem(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    return " ".join(processed_words)

# ----------------------------------
# STREAMLIT UI
# ----------------------------------
st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 IMDb Movie Review Sentiment Analysis")
st.write("Analyze a movie review and predict whether it is **Positive** or **Negative**.")

user_input = st.text_area(
    "✍️ Enter Movie Review",
    placeholder="Type your movie review here...",
    height=150
)

# ----------------------------------
# PREDICTION
# ----------------------------------
if st.button("🔍 Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a movie review.")
    else:
        clean_text = preprocess(user_input)

        if clean_text.strip() == "":
            st.warning("⚠️ Review contains only stopwords or invalid text.")
        else:
            with st.spinner("Analyzing review..."):
                vector = vectorizer.transform([clean_text])
                prediction = model.predict(vector)[0]

                if hasattr(model, "predict_proba"):
                    probability = model.predict_proba(vector).max()
                else:
                    probability = None

            if prediction == 1:
                st.success("😊 **Positive Review**")
            else:
                st.error("😞 **Negative Review**")

            if probability:
                st.metric("Confidence", f"{probability * 100:.2f}%")

# ----------------------------------
# FOOTER
# ----------------------------------
st.markdown("---")
st.caption("NLP | TF-IDF | Machine Learning | Streamlit")
