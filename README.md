# Sentiment-Analysis-of-Customer-Reviews
!pip install pandas numpy scikit-learn nltk wordcloud matplotlib streamlit
\\
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from wordcloud import WordCloud
nltk.download('stopwords')
\\
# Load Twitter Sentiment dataset directly from GitHub
url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
df = pd.read_csv(url)

# Rename for consistency
df = df.rename(columns={"tweet": "review", "label": "sentiment"})

# Show first few rows
df.head()
\\
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|[^a-zA-Z]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words
df["cleaned_review"] = df["review"].apply(clean_text)
\\
# Search for a .wav file inside extracted folder
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cleaned_review"])
y = df["sentiment"]
\\
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
\\
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
\\
def predict_sentiment(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]
    return "🟢 Positive" if pred == 1 else "🔴 Negative"

# Example:
print(predict_sentiment("I absolutely love this product!"))
