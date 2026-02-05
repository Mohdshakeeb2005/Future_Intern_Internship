import os
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "customer_support_tickets.csv")

df = pd.read_csv(file_path)

cols = [c.lower() for c in df.columns]

text_cols = [df.columns[i] for i, c in enumerate(cols) if any(k in c for k in ["subject", "description", "text", "message", "issue"])]
cat_col = next(df.columns[i] for i, c in enumerate(cols) if "category" in c or "type" in c)
pri_col = next(df.columns[i] for i, c in enumerate(cols) if "priority" in c)

df[text_cols] = df[text_cols].fillna('')
df['text'] = df[text_cols].agg(" ".join, axis=1)

def clean_text(t):
    t = t.lower()
    t = re.sub(r'[^a-zA-Z]', ' ', t)
    words = t.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)

tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['clean_text'])

y_category = df[cat_col]
X_train, X_test, y_train, y_test = train_test_split(X, y_category, test_size=0.2, random_state=42)

model_cat = MultinomialNB()
model_cat.fit(X_train, y_train)
y_pred_cat = model_cat.predict(X_test)

print("CATEGORY CLASSIFICATION")
print("Accuracy:", accuracy_score(y_test, y_pred_cat))
print(classification_report(y_test, y_pred_cat))

y_priority = df[pri_col]
X_train, X_test, y_train, y_test = train_test_split(X, y_priority, test_size=0.2, random_state=42)

model_pri = MultinomialNB()
model_pri.fit(X_train, y_train)
y_pred_pri = model_pri.predict(X_test)

print("PRIORITY PREDICTION")
print("Accuracy:", accuracy_score(y_test, y_pred_pri))
print(classification_report(y_test, y_pred_pri))
