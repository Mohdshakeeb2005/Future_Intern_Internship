import os
import pandas as pd
import nltk
import re
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "Resume.csv")

df = pd.read_csv(file_path)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['clean_resume'] = df['Resume_str'].apply(clean_text)

job_description = """
Looking for a Python developer with knowledge of machine learning,
data analysis, NLP, pandas, numpy, scikit-learn, and problem-solving skills.
"""

clean_jd = clean_text(job_description)

tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['clean_resume'].tolist() + [clean_jd])

similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

df['similarity'] = similarity_scores[0]

top_candidates = df.sort_values(by='similarity', ascending=False).head(5)

jd_words = set(clean_jd.split())

def missing_skills(resume):
    resume_words = set(resume.split())
    return jd_words - resume_words

top_candidates['missing_skills'] = top_candidates['clean_resume'].apply(missing_skills)

print(top_candidates[['Category', 'similarity', 'missing_skills']])
