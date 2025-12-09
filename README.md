# 💬 Sentiment Analysis using ML & Deep Learning

An end-to-end **Sentiment Analysis** project that classifies text into different sentiment classes using both **Machine Learning** and **Deep Learning (BiLSTM)** models, and exposes the final model through a **Streamlit web application**.

---

## 📌 Project Overview

This project demonstrates a complete data science workflow:

1. **Data Loading & Cleaning**
2. **Exploratory Data Analysis (EDA)**
3. **Text Preprocessing & Feature Engineering**
4. **Model Training & Evaluation**
5. **Deep Learning with BiLSTM**
6. **Model Saving**
7. **Deployment using Streamlit**

The app allows users to input any text (e.g., review, tweet, comment) and get a predicted sentiment in real time.

---

## 🗂 Dataset

- The dataset contains text samples along with corresponding **sentiment labels**.
- It was cleaned by:
  - Converting text to lowercase  
  - Removing URLs, mentions, hashtags, special characters  
  - Removing stopwords  
  - Handling rare sentiment classes by grouping them into an `Other` class

---

## ⚙️ Tech Stack

- **Language:** Python  
- **Libraries (ML):** scikit-learn, numpy, pandas  
- **Deep Learning:** TensorFlow / Keras (BiLSTM)  
- **NLP:** NLTK, TF-IDF Vectorizer  
- **Visualization:** matplotlib, seaborn, wordcloud  
- **Deployment:** Streamlit  

---

## 🧠 Models Used

### 🔹 Classical Machine Learning
- Logistic Regression  
- Multinomial Naive Bayes  
- Linear SVM  
- Random Forest  

Feature representation: **TF-IDF (uni + bi-grams)**

### 🔹 Deep Learning
- **Bidirectional LSTM (BiLSTM)** with:
  - Embedding layer  
  - LSTM units  
  - Dropout layers  
  - Dense output layer with softmax  

---

## 🏗 Project Structure

```text
SentimentAnalysisProject/
├── app.py                         # Streamlit app
├── Sentiment Analysis Using ML-Model (1).ipynb   # Main notebook
├── sentimentdataset.csv           # Original dataset
├── Cleaned_Sentiment_Dataset.csv  # Preprocessed dataset (optional)
├── best_classical_model.pkl       # Saved best classical ML model
├── tfidf_vectorizer.pkl           # Saved TF-IDF vectorizer
├── label_encoder.pkl              # Saved label encoder
├── bilstm_sentiment_model.h5      # Saved BiLSTM model
├── tokenizer.pkl                  # Saved tokenizer for BiLSTM
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation


---

## 📷 Demo (Streamlit App Working)

![App Screenshot](https://raw.githubusercontent.com/AYUSHTIWARI7126/SentimentAnalysisProject/main/WorkingSample.png)
