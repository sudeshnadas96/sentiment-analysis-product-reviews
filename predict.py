# predict.py
import joblib
from preprocess import clean_text

model = joblib.load("models/sentiment_model.pkl")

while True:
    text = input("Enter your product review (or 'exit' to stop): ")
    if text.lower() == 'exit':
        break
    cleaned = clean_text(text)
    prediction = model.predict([cleaned])
    print("Sentiment:", prediction[0])
