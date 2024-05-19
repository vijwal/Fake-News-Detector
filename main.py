import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import re
import string
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import textstat
# import nltk
# nltk.download('vader_lexicon')


data_new= pd.read_csv("C:/My Documents/vijwal/VIJWAL CODING/PYTHON PROJECTS/fake_news detector/train_mix_final.csv", dtype=str)

data= data_new
data.dropna(axis=0, inplace=True)
data['readability'] = data['text'].apply(lambda x: textstat.flesch_kincaid_grade(x))

def preprocess_text(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = text.lower()
    text = re.sub('\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words)
    return text


data['text'] = data['text'].apply(preprocess_text)

sid = SentimentIntensityAnalyzer()
data['sentiment'] = data['text'].apply(lambda x: sid.polarity_scores(x)['compound'])


X_train, X_test, y_train, y_test = train_test_split(data[['text','sentiment','readability']], data['class'], test_size=0.17, random_state=42)


tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=0.03, sublinear_tf=True, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['text'])
X_test_tfidf = tfidf_vectorizer.transform(X_test['text'])

X_train_features = np.hstack((X_train_tfidf.toarray(), np.array(X_train['sentiment']).reshape(-1, 1),np.array(X_train['readability']).reshape(-1, 1)))
X_test_features = np.hstack((X_test_tfidf.toarray(), np.array(X_test['sentiment']).reshape(-1, 1),np.array(X_test['readability']).reshape(-1, 1)))


lr_model = LogisticRegression(max_iter=5000,C=4,penalty='l2',solver='liblinear')
dt_model = DecisionTreeClassifier(max_features='auto',min_samples_leaf=1,min_samples_split=2)
gb_model = GradientBoostingClassifier(learning_rate=0.1,n_estimators=300,max_depth=5)


lr_model.fit(X_train_features, y_train)
dt_model.fit(X_train_features, y_train)
gb_model.fit(X_train_features, y_train)


lr_predictions = lr_model.predict(X_test_features)
dt_predictions = dt_model.predict(X_test_features)
gb_predictions = gb_model.predict(X_test_features)


print("Logistic Regression Classification Report:")
print(classification_report(y_test, lr_predictions))
print("Decision Tree Classification Report:")
print(classification_report(y_test, dt_predictions))
print("Gradient Boosting Classification Report:")
print(classification_report(y_test, gb_predictions))
print("Ensemble Classification Report:")
ensemble_predictions = ((lr_predictions.astype(int) + dt_predictions.astype(int) + gb_predictions.astype(int)) > 2).astype(int)
print(classification_report(y_test.astype(int), ensemble_predictions))


import joblib
save_directory = "C:\\My Documents\\vijwal\\VIJWAL CODING\\PYTHON PROJECTS\\fake_news detector\\models\\"
joblib.dump(tfidf_vectorizer, save_directory + 'tfidf_vectorizer.pkl')   
joblib.dump(lr_model, save_directory + 'lr_model.pkl')
joblib.dump(dt_model, save_directory + 'dt_model.pkl')
joblib.dump(gb_model, save_directory + 'gb_model.pkl')
