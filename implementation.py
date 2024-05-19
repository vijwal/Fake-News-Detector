import joblib
import re
import string
import textstat
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st
import numpy as np

tfidf_vectorizer = joblib.load('C:/My Documents/vijwal/VIJWAL CODING/PYTHON PROJECTS/fake_news detector/models/tfidf_vectorizer.pkl')
lr_model = joblib.load('C:/My Documents/vijwal\VIJWAL CODING/PYTHON PROJECTS/fake_news detector/models/lr_model.pkl')
dt_model = joblib.load('C:/My Documents/vijwal/VIJWAL CODING/PYTHON PROJECTS/fake_news detector/models/dt_model.pkl')
gb_model = joblib.load('C:/My Documents/vijwal/VIJWAL CODING/PYTHON PROJECTS/fake_news detector/models/gb_model.pkl')

def preprocess_text(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = text.lower()
    text = re.sub('\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words)
    return text

sid = SentimentIntensityAnalyzer()

def predict_news(text):
    readability = textstat.flesch_kincaid_grade(text)
    text = preprocess_text(text)
    sentiment = sid.polarity_scores(text)['compound']
    text_tfidf = tfidf_vectorizer.transform([text])
    features = np.hstack((text_tfidf.toarray(), np.array([sentiment]).reshape(-1, 1), np.array([readability]).reshape(-1, 1)))
    lr_pred = lr_model.predict(features)
    dt_pred = dt_model.predict(features)
    gb_pred = gb_model.predict(features)
    ensemble_pred = ((lr_pred.astype(int) + dt_pred.astype(int) + gb_pred.astype(int)) > 2).astype(int)
    return ensemble_pred[0]


def main():
    st.title("News Truth Predictor")
    st.write("Enter the text of a news article to predict whether it's true or fake.")

    news_text = st.text_area("News Article")

    if st.button("Predict"):
        prediction = predict_news(news_text)
        if prediction:
            st.write("The news article appears to be true.")
        else:
            st.write("The news article is appears to be fake.")

if __name__ == "__main__":
    main()
