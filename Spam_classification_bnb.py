import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import joblib
from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer
from  sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB

load_encoder = joblib.load("bnb_label.econder.pkl")
load_vectorizer = joblib.load("bnb_vectorizer.pkl")
load_model = joblib.load("bernoulli_naive_bayes_model.plk")


def clean_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
  ps = PorterStemmer()

  y = []

  for w in text:
    if w.isalnum():
       y.append(w)

  text = y[:]
  y.clear()


  for w in text:
    if w not in stopwords.words('english') and w not in string.punctuation:
       y.append(w)

  text = y[:]
  y.clear()

  for w in text:
    y.append(ps.stem(w))

  return " ".join(y)




def process_text(text):
    text = pd.Series(text). apply(clean_text)
    x = load_vectorizer.transform(text)
    return x





# App appearance

st.set_page_config(page_title="Spam Classification", page_icon="💬", layout="wide")

# Add a title and description
st.title("Email Spam Classification Using Bernoulli Naive Bayes Model")
st.write("""
This app uses a Bernoulli Naive Bayes (BNB) model to predict whether a given email text is a scam or legitimate.
You can copy and paste email text or enter the text by typing, and the model will classify it as legitimate, or scam. 
Scroll below to Enter or Type your text. Contact **Jacob Akubire** @jaakubire@gmail.com
for any concerns
""")

# Add some styling
st.markdown("""
<style>
body {
    background-color: #f0f2f6;
    color: #333;
    font-family: 'Arial', sans-serif;
}
header, footer {
    visibility: hidden;
}
.stTextInput > div > div > input {
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 4px;
}
.stButton > button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}
.stButton > button:hover {
    background-color: #45a049;
}
</style>
""", unsafe_allow_html=True)


image = Image.open("Spam_cloud.png")
st.image(image, use_column_width=True)

# Input text
text_input = st.text_input("Enter or copy and paste email text here:")

if text_input:
    # Vectorize the input text
    #text_vectorized = vectorizer.transform([text_input])

    text_clean_vectorized = process_text(text_input)

    # Predict sentiment
    prediction = load_model.predict(text_clean_vectorized)

    # Map prediction to sentiment
    #sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    #sentiment = sentiment_map[prediction[0]]

    sentiment = load_encoder.inverse_transform(prediction)[0]

    # Display the result
    if sentiment == "ham":
        st.write(f"**The email text**: {text_input} is  **predicted** as  **Legitimate**.")
    else:
       st.write(f"**The email text**: {text_input} is  **predicted** as  **Scam**.")
       

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 50px;">
    <p>
    </p>
</div>
""", unsafe_allow_html=True)
