# 🎫 TicketTriage — Support Ticket Classification & Priority Prediction

## 📌 Overview
TicketTriage is a Machine Learning and NLP project that automatically classifies customer support tickets into categories and predicts their priority level using ticket text.  
This helps support teams save time, reduce backlog, and respond faster to urgent issues.

## 🎯 Objective
- Classify tickets into categories (Billing, Technical Issue, Refund, etc.)
- Predict ticket priority (Critical, High, Medium, Low)
- Build a real-world decision-support system for support operations

## 🛠️ Tools & Libraries
- Python
- NLTK
- Scikit-learn
- Pandas
- TF-IDF Vectorizer
- Multinomial Naive Bayes

## 📁 Dataset
Customer Support Ticket Dataset from Kaggle:
https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset

## ⚙️ Workflow
1. Load support ticket dataset
2. Clean and preprocess ticket text
3. Convert text into numerical features using TF-IDF
4. Train ML model for:
   - Ticket category classification
   - Priority prediction
5. Evaluate using accuracy, precision, recall, and F1-score

## ✅ Features Implemented
- Text cleaning and preprocessing
- TF-IDF feature extraction
- Multi-class ticket category classification
- Priority level prediction
- Model performance evaluation

## 📊 Output
The system prints:
- Accuracy of category classification
- Accuracy of priority prediction
- Detailed classification report for each class

## 🚀 How to Run
```bash
pip install pandas nltk scikit-learn
python main.py
