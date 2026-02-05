# 📄 ResumeRank — Resume Screening & Candidate Ranking using NLP

## 📌 Overview
ResumeRank is a Machine Learning and NLP project that automatically screens resumes, compares them with a job description, ranks candidates based on role fit, and highlights missing skills.

This system simulates how modern HR-tech platforms and recruiters shortlist candidates efficiently.

---

## 🎯 Objective
- Extract meaningful information from resume text
- Compare resumes with a job description
- Score and rank candidates based on similarity
- Identify missing skills for each candidate

---

## 🛠️ Tools & Libraries
- Python
- Pandas
- NLTK
- spaCy
- Scikit-learn (TF-IDF, Cosine Similarity)

---

## 📁 Dataset
Resume Dataset from Kaggle:  
https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

> Download the dataset and place `Resume.csv` in the project folder before running.

---

## ⚙️ Workflow
1. Load resume dataset
2. Clean and preprocess resume text
3. Parse job description
4. Convert text into TF-IDF vectors
5. Compute cosine similarity between resumes and job role
6. Rank top candidates
7. Identify missing skills

---

## ✅ Features Implemented
- Resume text cleaning
- Job description parsing
- Resume-to-role similarity scoring
- Candidate ranking
- Skill gap identification

---

## 📊 Output
The system displays:
- Top ranked resumes based on similarity score
- Candidate category
- Missing skills compared to job role

---

## 🚀 How to Run
```bash
pip install -r requirements.txt
python main.py
