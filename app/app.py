import streamlit as st
import joblib
from preprocess import clean_text

st.title("Product Review Sentiment Analyzer")

model = joblib.load("models/sentiment_model.pkl")

review = st.text_area("Enter a product review:")

if st.button("Predict Sentiment"):
    cleaned = clean_text(review)
    result = model.predict([cleaned])[0]
    st.success(f"Predicted Sentiment: {result}")
